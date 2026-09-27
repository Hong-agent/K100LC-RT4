#!/usr/bin/env python3
"""附件链路自检：文档提取 + 多模态消息拍平。

在 DTK 容器里运行（需要 fastapi/tokenizers/jinja2）：
  bash scripts/dsh.sh 'python3 tools/test_attachments.py'
"""
import io
import json
import os
import sys
import tempfile
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

ROOT = os.environ.get('RT_ROOT', '/rt')
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import serve  # noqa: E402


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def docx_bytes(text):
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:body><w:p><w:r><w:t>%s</w:t></w:r></w:p></w:body></w:document>' % text
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as z:
        z.writestr('word/document.xml', xml)
    return buf.getvalue()


def xlsx_bytes():
    shared = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<si><t>姓名</t></si><si><t>张三</t></si></sst>'
    )
    sheet = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheetData><row><c t="s"><v>0</v></c><c t="s"><v>1</v></c></row></sheetData>'
        '</worksheet>'
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as z:
        z.writestr('xl/sharedStrings.xml', shared)
        z.writestr('xl/worksheets/sheet1.xml', sheet)
    return buf.getvalue()


def pdf_bytes():
    stream = b'BT /F1 12 Tf 72 720 Td (Hello PDF) Tj ET'
    import zlib
    packed = zlib.compress(stream)
    head = b'%%PDF-1.4\n1 0 obj\n<< /Length %d /Filter /FlateDecode >>\nstream\n' % len(packed)
    return head + packed + b'\nendstream\nendobj\n%%EOF\n'


def main():
    text, fmt, warning = serve.extract_document('a.txt', '你好\n世界'.encode())
    check('你好' in text and fmt == 'text', '纯文本提取失败')

    html = '<h1>标题</h1><script>bad()</script><p>正文</p>'
    text, fmt, warning = serve.extract_document('a.html', html.encode())
    check('标题' in text and '正文' in text and 'bad' not in text, 'HTML 提取失败')

    text, fmt, warning = serve.extract_document('a.docx', docx_bytes('文档正文'))
    check('文档正文' in text and fmt == 'docx', 'DOCX 提取失败')

    text, fmt, warning = serve.extract_document('a.xlsx', xlsx_bytes())
    check('姓名' in text and '张三' in text and fmt == 'xlsx', 'XLSX 提取失败')

    text, fmt, warning = serve.extract_document('a.pdf', pdf_bytes())
    check('Hello PDF' in text and fmt == 'pdf', 'PDF 兜底提取失败: %r' % text)

    # ---- 技能 / 工作区 ----
    old_workspace = serve.WORKSPACE_ROOT
    serve.WORKSPACE_ROOT = tempfile.mkdtemp(prefix='rt-skills-test-')
    try:
        conv = 'test-session'
        check('= 20' in serve._execute_skill(conv, {
            'name': 'calc', 'arguments': {'expression': '(2+3)*4'}}), 'calc 失败')
        serve._execute_skill(conv, {'name': 'write_file', 'arguments': {
            'name': 'report.md', 'content': '# hello\n世界'}})
        check('世界' in serve._read_file(conv, 'report.md'), 'write/read 失败')
        serve._execute_skill(conv, {'name': 'make_csv', 'arguments': {
            'name': 'data.csv', 'columns': ['a', 'b'], 'rows': [[1, 2], [3, 4]]}})
        check(len(serve._file_list(conv)) == 2, '文件列表数量错误')
        calls = serve._tool_calls(
            '<tool_call><function=calc><parameter=expression>1+1</parameter>'
            '</function></tool_call>')
        check(calls and calls[0]['name'] == 'calc', 'XML tool call 解析失败')
        calls = serve._tool_calls(
            '<tool_call>{"name":"now","arguments":{}}</tool_call>')
        check(calls and calls[0]['name'] == 'now', 'JSON tool call 解析失败')
    finally:
        serve.WORKSPACE_ROOT = old_workspace

    old_mode = serve.VISION_MODE
    serve.VISION_MODE = 'off'
    try:
        messages = serve.prepare_messages([{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': '看这张图'},
                {'type': 'image_url', 'name': 'a.png',
                 'image_url': {'url': 'data:image/png;base64,AA=='}},
            ],
        }])
    finally:
        serve.VISION_MODE = old_mode
    check('看这张图' in messages[0]['content'], '多模态文字丢失')
    check('未启用视觉' in messages[0]['content'], '纯文本运行时降级说明缺失')

    old = serve._vision_describe
    old_base, old_model = serve.VISION_BASE_URL, serve.VISION_MODEL
    serve._vision_describe = lambda url, name: '图片里是一只猫'
    serve.VISION_BASE_URL, serve.VISION_MODEL = 'http://mock/v1', 'mock-vl'
    try:
        messages = serve.prepare_messages([{
            'role': 'user',
            'content': [{'type': 'image_url', 'name': 'cat.png',
                         'image_url': {'url': 'data:image/png;base64,AA=='}}],
        }])
    finally:
        serve._vision_describe = old
        serve.VISION_BASE_URL, serve.VISION_MODEL = old_base, old_model
    check('图片里是一只猫' in messages[0]['content'], '视觉桥描述未注入')

    class FakeVision:
        last_ms = 0.0

        def encode(self, url):
            return np.zeros((2, 5120), dtype=np.float32), [[1, 2, 2]]

    old_ready = serve._vision_local_ready
    old_encoder = serve._VISION_ENCODER
    old_base, old_model = serve.VISION_BASE_URL, serve.VISION_MODEL
    old_mode = serve.VISION_MODE
    serve._vision_local_ready = lambda: True
    serve._VISION_ENCODER = FakeVision()
    serve.VISION_BASE_URL, serve.VISION_MODEL = '', ''
    serve.VISION_MODE = 'local'
    try:
        messages, embeds = serve.prepare_messages_and_embeds([{
            'role': 'user',
            'content': [{'type': 'text', 'text': '看图'},
                        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AA=='}}],
        }])
        check(messages[0]['content'].count('<|image_pad|>') == 2, '本地视觉占位符数量错误')
        check(len(embeds) == 1 and embeds[0].shape == (2, 5120), '本地视觉 embedding 错误')
        spans = serve._image_token_spans([1, serve.IMAGE_TOKEN_ID, serve.IMAGE_TOKEN_ID, 2],
                                         embeds)
        check(spans == [(1, 2)], 'image span 定位错误：%r' % spans)
    finally:
        serve._vision_local_ready = old_ready
        serve._VISION_ENCODER = old_encoder
        serve.VISION_BASE_URL, serve.VISION_MODEL = old_base, old_model
        serve.VISION_MODE = old_mode

    messages = serve.prepare_messages([{
        'role': 'assistant', 'content': 'ok',
        'tool_calls': [{'function': {'name': 'f', 'arguments': {}}}],
    }])
    check(messages[0].get('tool_calls'), 'prepare_messages 丢失 tool_calls')

    class VisionHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get('Content-Length', '0'))
            body = json.loads(self.rfile.read(n))
            check(body['model'] == 'mock-vl', '视觉桥 model 未透传')
            parts = body['messages'][0]['content']
            check(parts[1]['type'] == 'image_url', '视觉桥未收到 image_url')
            out = json.dumps({'choices': [{'message': {'content': 'MOCK-VL-OK'}}]}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *args):
            pass

    mock = HTTPServer(('127.0.0.1', 0), VisionHandler)
    threading.Thread(target=mock.serve_forever, daemon=True).start()
    old_base, old_model = serve.VISION_BASE_URL, serve.VISION_MODEL
    serve.VISION_BASE_URL, serve.VISION_MODEL = 'http://127.0.0.1:%d' % mock.server_port, 'mock-vl'
    try:
        desc = serve._vision_describe('data:image/png;base64,AA==', 'a.png')
        check(desc == 'MOCK-VL-OK', '视觉桥 HTTP 调用失败：%r' % desc)
    finally:
        serve.VISION_BASE_URL, serve.VISION_MODEL = old_base, old_model
        mock.shutdown()
        mock.server_close()

    ids, rendered = serve.build_ids({'messages': [{'role': 'user', 'content': '你好'}]})
    check(ids and '你好' in rendered, 'chat template / tokenizer 链路失败')

    # HTTP 路由与 chat 请求的端到端烟测；用假 engine，避免测试依赖 GPU。
    from fastapi.testclient import TestClient

    class FakeEngine:
        def set_mtp(self, n):
            pass

        def prefill(self, ids):
            return 'OK prefill'

        def gen(self, n, temp, top_p, top_k, seed, stops):
            return [], 'END 0 0'

    old_engine = serve.ENGINE
    old_base, old_model = serve.VISION_BASE_URL, serve.VISION_MODEL
    old_mode = serve.VISION_MODE
    serve.ENGINE = FakeEngine()
    serve.VISION_MODE = 'off'
    try:
        client = TestClient(serve.app)
        r = client.get('/v1/capabilities')
        check(r.status_code == 200 and r.json()['document_extract'], 'capabilities 失败')
        r = client.post('/v1/extract?name=a.txt', content='附件正文'.encode())
        check(r.status_code == 200 and r.json()['text'] == '附件正文', 'extract 路由失败')
        r = client.post('/v1/extract?name=a.bin', content=b'\x00\x01\x02\x03\xff\xfe')
        check(r.status_code == 415, '不支持格式应返回 415')
        old_ws = serve.WORKSPACE_ROOT
        serve.WORKSPACE_ROOT = tempfile.mkdtemp(prefix='rt-http-skills-')
        try:
            serve._write_file('http-test', 'hello.txt', '文件内容')
            r = client.get('/v1/files?conversation_id=http-test')
            check(r.status_code == 200 and r.json()['files'][0]['name'] == 'hello.txt',
                  '文件列表路由失败')
            r = client.get('/v1/files/hello.txt?conversation_id=http-test')
            check(r.status_code == 200 and '文件内容' in r.text, '文件下载路由失败')
            r = client.delete('/v1/files/hello.txt?conversation_id=http-test')
            check(r.status_code == 200, '文件删除路由失败')
        finally:
            serve.WORKSPACE_ROOT = old_ws
        r = client.post('/v1/chat/completions', json={
            'model': 'test',
            'messages': [{'role': 'user', 'content': [
                {'type': 'text', 'text': '看图'},
                {'type': 'image_url', 'name': 'a.png',
                 'image_url': {'url': 'data:image/png;base64,AA=='}},
            ]}],
            'max_tokens': 1,
            'temperature': 0,
        })
        check(r.status_code == 200, '带图片的 chat 请求失败：%s' % r.text)
        old_ctx = serve.CTX_LIMIT
        serve.CTX_LIMIT = 1
        try:
            r = client.post('/v1/chat/completions', json={
                'model': 'test',
                'messages': [{'role': 'user', 'content': '你好'}],
                'max_tokens': 1,
            })
            check(r.status_code == 413, '超长 prompt 应返回 413')
        finally:
            serve.CTX_LIMIT = old_ctx

        serve.VISION_MODE = 'external'
        serve.VISION_BASE_URL, serve.VISION_MODEL = 'http://127.0.0.1:9/v1', 'mock-vl'
        r = client.get('/v1/capabilities')
        check(r.json()['vision'] is True, '配置视觉桥后 capabilities 未更新')
    finally:
        serve.ENGINE = old_engine
        serve.VISION_BASE_URL, serve.VISION_MODEL = old_base, old_model
        serve.VISION_MODE = old_mode

    print('附件链路自检：全部通过')


if __name__ == '__main__':
    main()
