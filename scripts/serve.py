#!/usr/bin/env python3
# OpenAI 兼容 HTTP 服务：把自研运行时（build/rt --engine）包成 /v1/chat/completions。
# 接口形态与常见 OpenAI 客户端一致，客户端不用改。
#
#   python3 scripts/serve.py --port 8080 [--ctx 131072] [--default-max-tokens 128000]
#
# 依赖（DTK 镜像里都有）：fastapi / uvicorn / tokenizers / jinja2。
import argparse
import ast
import asyncio
import csv
import datetime
import hashlib
import io
import json
import mimetypes
import os
import re
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
import zipfile
import zlib
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tok as T                                      # noqa: E402
from chat import Engine                              # noqa: E402

from fastapi import FastAPI, HTTPException, Request    # noqa: E402
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware    # noqa: E402
import uvicorn                                        # noqa: E402

EOS = [248044, 248046]          # <|endoftext|> / <|im_end|>
MODEL_NAME = os.environ.get('RT_SERVED_NAME', 'qwen38-rt')

# 附件上限只用于避免网页/接口被超大文件拖死；可按目标机内存调整。
MAX_UPLOAD_MB = int(os.environ.get('RT_MAX_UPLOAD_MB', '25'))
MAX_UPLOAD_BYTES = max(1, MAX_UPLOAD_MB) * 1024 * 1024
MAX_EXTRACT_CHARS = int(os.environ.get('RT_MAX_EXTRACT_CHARS', '40000'))
CTX_LIMIT = int(os.environ.get('RT_CTX', '131072'))
# 请求体不带 max_tokens 时用它；main() 会用 --default-max-tokens 覆盖
DEFAULT_MAX_TOKENS = int(os.environ.get('RT_DEFAULT_MAX_TOKENS', '128000'))

# 可选的外部视觉桥：当前 RT4 运行时只加载文本塔，model.visual.* 在转换时跳过。
# 配好一个 OpenAI 兼容的视觉模型后，服务端会先把图片转成文字描述，再交给 RT4；
# 没配置时图片仍可在网页中展示，但会明确告诉模型“看不到图片内容”。
VISION_BASE_URL = os.environ.get('RT_VISION_BASE_URL', '').rstrip('/')
VISION_MODEL = os.environ.get('RT_VISION_MODEL', '')
VISION_API_KEY = os.environ.get('RT_VISION_API_KEY', '')
VISION_TIMEOUT = float(os.environ.get('RT_VISION_TIMEOUT', '120'))
VISION_MODE = os.environ.get('RT_VISION_MODE', 'auto').lower()
VISION_RT4 = os.environ.get('RT_VISION_RT4', '')
VISION_PROMPT = os.environ.get('RT_VISION_PROMPT') or (
    '请详细描述这张图片的内容，包括其中的文字、表格、图表、界面元素和关键细节。'
    '如果是文档截图，请尽量完整转写可见文字。只输出描述，不要寒暄。')
_VISION_CACHE = {}
_VISION_ENCODER = None
_VISION_ENCODER_LOCK = threading.Lock()

# ---- 技能 / 工作区 ----
WORKSPACE_ROOT = os.environ.get('RT_WORKSPACE_DIR') or os.path.join(ROOT, 'workspaces')
MAX_FILE_BYTES = int(os.environ.get('RT_MAX_FILE_KB', '2048')) * 1024
MAX_WORKSPACE_BYTES = int(os.environ.get('RT_MAX_WORKSPACE_MB', '64')) * 1024 * 1024
MAX_SKILL_ROUNDS = int(os.environ.get('RT_MAX_SKILL_ROUNDS', '6'))

SKILLS = [
    {'type': 'function', 'function': {
        'name': 'now', 'description': '获取当前本地日期、时间和时区。',
        'parameters': {'type': 'object', 'properties': {}},
    }},
    {'type': 'function', 'function': {
        'name': 'calc', 'description': '计算一个不含变量的四则/乘方表达式。',
        'parameters': {'type': 'object', 'properties': {
            'expression': {'type': 'string', 'description': '例如 (2+3)*4/2'},
        }, 'required': ['expression']},
    }},
    {'type': 'function', 'function': {
        'name': 'write_file',
        'description': '在当前会话工作区创建或覆盖一个文本文件，并返回文件信息。',
        'parameters': {'type': 'object', 'properties': {
            'name': {'type': 'string', 'description': '文件名，例如 report.md'},
            'content': {'type': 'string', 'description': '完整文件内容'},
            'append': {'type': 'boolean', 'description': '是否追加而不是覆盖'},
        }, 'required': ['name', 'content']},
    }},
    {'type': 'function', 'function': {
        'name': 'read_file', 'description': '读取当前会话工作区里的文本文件。',
        'parameters': {'type': 'object', 'properties': {
            'name': {'type': 'string'},
            'max_chars': {'type': 'integer', 'description': '最多返回多少个字符'},
        }, 'required': ['name']},
    }},
    {'type': 'function', 'function': {
        'name': 'list_files', 'description': '列出当前会话工作区里的所有文件。',
        'parameters': {'type': 'object', 'properties': {}},
    }},
    {'type': 'function', 'function': {
        'name': 'delete_file', 'description': '删除当前会话工作区里的一个文件。',
        'parameters': {'type': 'object', 'properties': {
            'name': {'type': 'string'},
        }, 'required': ['name']},
    }},
    {'type': 'function', 'function': {
        'name': 'make_csv', 'description': '把二维数据写成 CSV 文件。',
        'parameters': {'type': 'object', 'properties': {
            'name': {'type': 'string'},
            'columns': {'type': 'array', 'items': {'type': 'string'}},
            'rows': {'type': 'array', 'items': {'type': 'array'}},
        }, 'required': ['name', 'columns', 'rows']},
    }},
]
SKILL_NAMES = {s['function']['name'] for s in SKILLS}

app = FastAPI()
# 局域网里的其它网页/工具直接跨域调用时，浏览器需要 CORS 预检。
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'],
                   allow_headers=['*'])
ENGINE = None
LOCK = asyncio.Lock()
STATS = {'prefill_tokens': 0, 'gen_tokens': 0, 'prefill_ms': 0.0, 'gen_ms': 0.0, 'reqs': 0}
WEB_INDEX = os.path.join(ROOT, 'web', 'index.html')


THINK_END = '</think>'


def template_kwargs(body):
    """模型自带 chat template 的开关：reasoning_effort / enable_thinking。"""
    kw = {}
    # vLLM 风格：{"chat_template_kwargs": {"enable_thinking": false}}
    ck = body.get('chat_template_kwargs') or {}
    if isinstance(ck, dict):
        for k in ('enable_thinking', 'reasoning_effort'):
            if k in ck:
                kw[k] = ck[k]
    if 'enable_thinking' in body:
        kw['enable_thinking'] = bool(body['enable_thinking'])
    if body.get('reasoning_effort'):
        kw['reasoning_effort'] = body['reasoning_effort']
    return kw


def thinking_enabled(body):
    """默认模板会思考；显式 enable_thinking=false 才关闭。"""
    return bool(template_kwargs(body).get('enable_thinking', True))


def split_thinking(text, enabled=True):
    """把 '思考...</think>回答' 拆成 (reasoning, answer)。"""
    if not enabled:
        return '', text
    i = text.find(THINK_END)
    if i < 0:
        return text.strip('\r\n'), ''
    reasoning = text[:i].strip('\r\n')
    answer = text[i + len(THINK_END):].lstrip('\r\n')
    return reasoning, answer


class ThinkSplitter:
    """流式增量 detokenize 的切分器：把「思考」与「正式回答」分成两股逐段产出。

    解码是逐 token 的，decode(acc) 每次都会变长，这里只把**新增**的部分发出去；
    正式回答要等 THINK_END 出现之后才开始，之前的内容算思考段（reasoning_content）。
    结果与 split_thinking() 对整段文本的切法一致（流式与非流式内容不会分叉）。
    """

    def __init__(self, think_enabled=True):
        self.think = think_enabled
        self.sent_reason = 0
        self.sent_content = 0
        self.tag_pos = 0 if not think_enabled else -1
        self.started = False

    def feed(self, cur):
        """返回 [(field, text), ...]，field 是 'reasoning' 或 'content'。"""
        out = []
        if not self.think:
            if len(cur) > self.sent_content:
                out.append(('content', cur[self.sent_content:]))
                self.sent_content = len(cur)
            return out
        if self.tag_pos < 0:
            i = cur.find(THINK_END)
            if i >= 0:
                self.tag_pos = i
                if i > self.sent_reason:
                    out.append(('reasoning', cur[self.sent_reason:i]))
                    self.sent_reason = i
                self.sent_content = i + len(THINK_END)
                self.started = False
            elif len(cur) > self.sent_reason:
                out.append(('reasoning', cur[self.sent_reason:]))
                self.sent_reason = len(cur)
        if self.tag_pos >= 0:
            if not self.started:
                while self.sent_content < len(cur) and cur[self.sent_content] in '\r\n':
                    self.sent_content += 1
                if self.sent_content < len(cur):
                    self.started = True
                    out.append(('content', cur[self.sent_content:]))
                    self.sent_content = len(cur)
            elif len(cur) > self.sent_content:
                out.append(('content', cur[self.sent_content:]))
                self.sent_content = len(cur)
        return out


def _decode_text_bytes(data):
    """尽量正确解码文本；中文文档常见 UTF-8/GB18030。"""
    for enc in ('utf-8-sig', 'utf-8'):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            pass
    for enc in ('gb18030', 'big5'):
        try:
            text = data.decode(enc)
            if text.count('\ufffd') == 0:
                return text
        except UnicodeDecodeError:
            pass
    return data.decode('utf-8', errors='replace')


def _clean_extracted(text):
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'[ \t]+\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


class _HTMLTextExtractor:
    """stdlib HTML 取正文，跳过 script/style。"""

    def __init__(self):
        from html.parser import HTMLParser
        class Parser(HTMLParser):
            def __init__(self):
                super().__init__(convert_charrefs=True)
                self.parts = []
                self.skip = 0

            def handle_starttag(self, tag, attrs):
                if tag in ('script', 'style', 'noscript', 'svg'):
                    self.skip += 1
                elif tag in ('p', 'div', 'br', 'li', 'tr', 'h1', 'h2', 'h3',
                             'h4', 'h5', 'h6', 'table'):
                    self.parts.append('\n')

            def handle_endtag(self, tag):
                if tag in ('script', 'style', 'noscript', 'svg') and self.skip:
                    self.skip -= 1
                elif tag in ('p', 'div', 'li', 'tr', 'h1', 'h2', 'h3',
                             'h4', 'h5', 'h6', 'table'):
                    self.parts.append('\n')

            def handle_data(self, data):
                if not self.skip:
                    self.parts.append(data)

        self.parser = Parser()

    def feed(self, text):
        self.parser.feed(text)
        return ''.join(self.parser.parts)


def _xml_text(xml_bytes, tag):
    root = ET.fromstring(xml_bytes)
    needle = '}' + tag
    vals = []
    for node in root.iter():
        if node.tag.endswith(needle) and node.text:
            vals.append(node.text)
    return ''.join(vals)


def _docx_text(data):
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        xml = z.read('word/document.xml')
    root = ET.fromstring(xml)
    ns = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
    lines = []
    for para in root.iter('{%s}p' % ns):
        parts = [node.text or '' for node in para.iter('{%s}t' % ns)]
        lines.append(''.join(parts))
    return '\n'.join(lines)


def _pptx_text(data):
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = sorted(n for n in z.namelist()
                       if n.startswith('ppt/slides/slide') and n.endswith('.xml'))
        out = []
        for n in names:
            nums = re.search(r'slide(\d+)\.xml$', n)
            out.append('# 幻灯片 %s' % (nums.group(1) if nums else n))
            out.append(_xml_text(z.read(n), 't'))
    return '\n'.join(out)


def _xlsx_text(data):
    main = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = z.namelist()
        shared = []
        if 'xl/sharedStrings.xml' in names:
            root = ET.fromstring(z.read('xl/sharedStrings.xml'))
            for si in root.iter('{%s}si' % main):
                shared.append(''.join(t.text or '' for t in si.iter('{%s}t' % main)))
        sheets = sorted((n for n in names if n.startswith('xl/worksheets/sheet')
                         and n.endswith('.xml')),
                        key=lambda s: int(re.search(r'(\d+)', s).group(1)))
        out = []
        for n in sheets:
            out.append('# %s' % n)
            root = ET.fromstring(z.read(n))
            for row in root.iter('{%s}row' % main):
                cells = []
                for c in row.findall('{%s}c' % main):
                    t = c.get('t')
                    v = c.find('{%s}v' % main)
                    inline = c.find('{%s}is' % main)
                    if t == 's' and v is not None and v.text is not None:
                        val = shared[int(v.text)]
                    elif t == 'inlineStr' and inline is not None:
                        val = ''.join(x.text or '' for x in inline.iter('{%s}t' % main))
                    elif v is not None:
                        val = v.text or ''
                    else:
                        val = ''
                    cells.append(val)
                if any(cells):
                    out.append('\t'.join(cells).rstrip())
    return '\n'.join(out)


def _pdf_unescape(raw):
    out = bytearray()
    i = 0
    while i < len(raw):
        c = raw[i]
        if c != 0x5c:
            out.append(c)
            i += 1
            continue
        i += 1
        if i >= len(raw):
            break
        c = raw[i]
        maps = {ord('n'): 10, ord('r'): 13, ord('t'): 9, ord('b'): 8,
                ord('f'): 12, ord('('): 40, ord(')'): 41, ord('\\'): 92}
        if c in maps:
            out.append(maps[c])
            i += 1
        elif 48 <= c <= 55:
            j = i
            while j < len(raw) and j < i + 3 and 48 <= raw[j] <= 55:
                j += 1
            out.append(int(raw[i:j], 8) & 0xFF)
            i = j
        else:
            i += 1
    return bytes(out)


def _pdf_text_naive(data):
    """不依赖第三方库的 PDF 文本兜底：适合简单、带 FlateDecode 文本流的 PDF。"""
    found = []
    for match in re.finditer(rb'stream\r?\n(.*?)\r?\nendstream', data, re.S):
        raw = match.group(1)
        try:
            raw = zlib.decompress(raw)
        except zlib.error:
            pass
        for item in re.finditer(rb'\((?:\\.|[^\\()])*\)', raw, re.S):
            val = _pdf_unescape(item.group(0)[1:-1])
            if not val:
                continue
            try:
                s = val.decode('utf-8')
            except UnicodeDecodeError:
                s = val.decode('latin-1', errors='replace')
            s = ''.join(ch if ch.isprintable() or ch in '\n\t' else ' ' for ch in s).strip()
            if s:
                found.append(s)
    return ' '.join(found)


def _pdf_text(data):
    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            return (_pdf_text_naive(data),
                    '容器内未安装 pypdf，已使用内置基础 PDF 提取器；复杂 PDF 可能提取不完整。')
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or '')
    return '\n\n'.join(pages), ''


def extract_document(name, data):
    """把上传文件转成纯文本，返回 (text, format, warning)。"""
    ext = os.path.splitext(name.lower())[1]
    if ext == '.pdf':
        text, warning = _pdf_text(data)
        return text, 'pdf', warning
    if ext == '.docx':
        return _docx_text(data), 'docx', ''
    if ext == '.pptx':
        return _pptx_text(data), 'pptx', ''
    if ext == '.xlsx':
        return _xlsx_text(data), 'xlsx', ''
    if ext in ('.html', '.htm', '.xhtml'):
        return _HTMLTextExtractor().feed(_decode_text_bytes(data)), 'html', ''
    if ext in ('.md', '.markdown', '.txt', '.log', '.csv', '.tsv', '.json',
               '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf', '.xml',
               '.py', '.cpp', '.cc', '.c', '.h', '.hpp', '.hip', '.cu',
               '.js', '.ts', '.sh', '.bash', '.sql', '.rst', '.tex', '.srt'):
        return _decode_text_bytes(data), 'text', ''
    if ext == '.rtf':
        text = re.sub(r'\\[a-zA-Z]+-?\d* ?', '', _decode_text_bytes(data))
        text = text.replace('{', '').replace('}', '')
        return text, 'rtf', ''
    if ext == '.odt':
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            return _HTMLTextExtractor().feed(z.read('content.xml').decode('utf-8',
                                                                          errors='replace')), 'odt', ''

    # 没有扩展名或扩展名不认识时：像文本就收，否则明确拒绝。
    text = _decode_text_bytes(data)
    if not text:
        return '', 'empty', ''
    printable = sum(ch.isprintable() or ch in '\n\t' for ch in text)
    if printable / max(1, len(text)) < 0.85:
        raise ValueError('暂不支持该文件格式：%s' % (name or '(未命名)'))
    return text, 'text', '未识别扩展名，已按纯文本读取。'


def _vision_local_file():
    return VISION_RT4 or os.path.join(
        os.environ.get('RT_MODEL_DIR', os.path.join(ROOT, 'models', 'Qwen3.8-27B-NVFP4')),
        'rt4', 'qwen38_27b_vision.rt4')


def _vision_local_ready():
    return (VISION_MODE in ('auto', 'local') and
            os.path.exists(_vision_local_file()))


def _vision_backend():
    """返回 None / 'external' / 'local'。"""
    if VISION_MODE == 'off':
        return None
    if VISION_MODE in ('auto', 'external') and VISION_BASE_URL and VISION_MODEL:
        return 'external'
    if _vision_local_ready():
        return 'local'
    return None


def _vision_ready():
    return _vision_backend() is not None


def _get_local_encoder():
    global _VISION_ENCODER
    if _VISION_ENCODER is None:
        with _VISION_ENCODER_LOCK:
            if _VISION_ENCODER is None:
                from vision_encoder import LocalVisionEncoder
                _VISION_ENCODER = LocalVisionEncoder(engine=ENGINE,
                                                     vision_rt4=_vision_local_file())
    return _VISION_ENCODER


def _vision_describe(image_url, name):
    """调用外部 OpenAI 兼容视觉模型，返回图片的文字描述。"""
    if _vision_backend() != 'external' or not isinstance(image_url, str) or not image_url:
        return None
    key = hashlib.sha256(image_url.encode('utf-8')).hexdigest()
    if key in _VISION_CACHE:
        return _VISION_CACHE[key]
    payload = {
        'model': VISION_MODEL,
        'messages': [{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': VISION_PROMPT},
                {'type': 'image_url', 'image_url': {'url': image_url}},
            ],
        }],
        'temperature': 0,
        'max_tokens': int(os.environ.get('RT_VISION_MAX_TOKENS', '768')),
    }
    headers = {'Content-Type': 'application/json'}
    if VISION_API_KEY:
        headers['Authorization'] = 'Bearer ' + VISION_API_KEY
    req = urllib.request.Request(VISION_BASE_URL + '/chat/completions',
                                 data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
                                 headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=VISION_TIMEOUT) as resp:
            out = json.loads(resp.read().decode('utf-8'))
        msg = (out.get('choices') or [{}])[0].get('message') or {}
        desc = msg.get('content') or msg.get('reasoning_content') or ''
        if isinstance(desc, list):
            desc = ''.join(p.get('text', '') for p in desc if isinstance(p, dict))
        desc = str(desc)
        if THINK_END in desc:
            desc = desc.split(THINK_END, 1)[1]
        desc = desc.strip()
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError,
            IndexError, OSError) as e:
        print(f'[serve] 视觉桥失败（{name}）: {e}', file=sys.stderr)
        return None
    if desc:
        if len(_VISION_CACHE) >= 128:
            _VISION_CACHE.pop(next(iter(_VISION_CACHE)))
        _VISION_CACHE[key] = desc
    return desc or None


def _image_part_url(part):
    if not isinstance(part, dict):
        return ''
    image = part.get('image_url')
    if isinstance(image, str):
        return image
    if isinstance(image, dict):
        return image.get('url') or ''
    image = part.get('image')
    if isinstance(image, str):
        return image
    return ''


def prepare_messages_and_embeds(messages):
    """把 OpenAI 多模态 content 拍平成 RT4 文本模板能吃的消息。

    本地视觉塔：把图片替换成 <|vision_start|> + N 个 <|image_pad|> + <|vision_end|>，
    并返回可直接覆盖这些 image_pad 位置的 f32 embedding。
    外部视觉桥：把图片先转成文字描述。
    都没有：保留明确的“未启用视觉”说明。
    """
    out = []
    embeds = []
    image_no = 0
    for raw in messages or []:
        if not isinstance(raw, dict):
            continue
        # 保留 tool_calls / reasoning_content 等模板会用到的其它字段。
        msg = dict(raw)
        msg['role'] = raw.get('role', 'user')
        content = raw.get('content', '')
        if isinstance(content, list):
            texts = []
            for part in content:
                if isinstance(part, str):
                    texts.append(part)
                    continue
                if not isinstance(part, dict):
                    continue
                if part.get('type') == 'text' or 'text' in part:
                    texts.append(str(part.get('text') or ''))
                elif (_image_part_url(part) or part.get('type') in ('image', 'image_url',
                                                                   'input_image')):
                    image_no += 1
                    name = str(part.get('name') or part.get('filename') or f'图片{image_no}')
                    url = _image_part_url(part)
                    backend = _vision_backend()
                    if backend == 'local':
                        try:
                            enc = _get_local_encoder()
                            emb, grid = enc.encode(url)
                        except Exception as e:                       # noqa: BLE001
                            raise HTTPException(400, f'图片 {name} 编码失败：{e}') from e
                        embeds.append(emb)
                        n = len(emb)
                        print(f'[serve] 本地视觉塔：{name} {grid} -> {n} token, '
                              f'{enc.last_ms:.1f} ms',
                              file=sys.stderr)
                        texts.append('<|vision_start|>' + '<|image_pad|>' * n +
                                     '<|vision_end|>')
                    elif backend == 'external':
                        desc = _vision_describe(url, name)
                        if desc:
                            texts.append(f'[用户上传的图片 {image_no}：{name}]\n{desc}\n')
                        else:
                            texts.append(
                                f'[用户上传了图片 {image_no}：{name}；视觉桥调用失败，'
                                f'无法查看图片内容。]\n')
                    else:
                        texts.append(
                            f'[用户上传了图片 {image_no}：{name}；当前运行时未启用视觉能力，'
                            f'无法查看图片内容，只能依据用户文字回答。]\n')
            msg['content'] = '\n'.join(texts)
        else:
            msg['content'] = '' if content is None else str(content)
        out.append(msg)
    return out, embeds


def prepare_messages(messages):
    """兼容旧调用：只返回拍平后的消息。"""
    return prepare_messages_and_embeds(messages)[0]


def build_ids_and_embeds(body):
    """把 OpenAI 请求变成 token id 序列 + 本地视觉 embedding 列表。"""
    if body.get('messages'):
        msgs, embeds = prepare_messages_and_embeds(body['messages'])
        # 模型自带的模板默认走 xhigh 推理强度（会先输出一段思考）；客户端可以用
        # {"reasoning_effort": "low"} 让它直接给结论。模板只认 xhigh/medium/low。
        kw = template_kwargs(body)
        if body.get('tools'):
            kw['tools'] = body['tools']
        txt = T.apply_chat(msgs, add_generation_prompt=True, **kw)
        return T.encode(txt), txt, embeds
    prompt = body.get('prompt', '')
    if isinstance(prompt, list):
        prompt = prompt[0]
    return T.encode(prompt), prompt, []


def build_ids(body):
    """兼容旧调用：只返回 token id 和渲染文本。"""
    ids, txt, _ = build_ids_and_embeds(body)
    return ids, txt


def sampling(body):
    temp = body.get('temperature')
    temp = 0.0 if temp is None else float(temp)
    top_p = body.get('top_p')
    top_p = 1.0 if top_p is None else float(top_p)
    top_k = int(body.get('top_k') or 0)
    # max_tokens：客户端没给就用 --default-max-tokens（默认 128000，即「一直生成到 EOS
    # 或者把上下文用满」）。显式给值也只是上限，装不下时按剩余上下文收窄
    # （见 clamp_max_tokens）；只有 prompt 本身就超出上下文才报 413。
    raw = body.get('max_tokens')
    n = int(raw) if raw else DEFAULT_MAX_TOKENS
    seed = int(body.get('seed') or 1234)
    return n, temp, top_p, top_k, seed


def clamp_max_tokens(prompt_len, n):
    """max_tokens 是「最多生成多少」，不是要预留一整块上下文。

    客户端（含本项目的网页）常直接发 128000，装不下时按剩余上下文收窄即可；
    只有 prompt 本身就超出上下文才报 413。
    """
    if prompt_len >= CTX_LIMIT:
        raise HTTPException(413, f'输入 {prompt_len} token 已经超出上下文 {CTX_LIMIT}；'
                                 f'请缩短文档或放宽 --ctx。')
    return max(1, min(n, CTX_LIMIT - prompt_len))


@app.get('/health')
def health():
    return {'ok': True, 'model': MODEL_NAME, 'vision': _vision_ready(),
            'vision_backend': _vision_backend(), **STATS}


@app.get('/')
@app.get('/ui')
def chat_ui():
    """项目自带的简易对话网页（同源调用 /v1/chat/completions，支持流式）。"""
    if os.path.exists(WEB_INDEX):
        return FileResponse(WEB_INDEX)
    return HTMLResponse('<h3>缺少 web/index.html</h3>', status_code=500)


@app.get('/v1/models')
def models():
    return {'object': 'list', 'data': [{'id': MODEL_NAME, 'object': 'model',
                                        'created': int(time.time()), 'owned_by': 'self'}]}


@app.get('/models')
def models_alias():
    """兼容把 Base URL 填成 http://host:8080 的客户端。"""
    return models()


@app.get('/v1/capabilities')
def capabilities():
    """网页据此决定图片是直接送模型还是只作为会话附件展示。"""
    return {
        'object': 'capabilities',
        'model': MODEL_NAME,
        'text': True,
        'vision': _vision_ready(),
        'vision_backend': _vision_backend(),
        'vision_model': (VISION_MODEL if _vision_backend() == 'external'
                         else ('qwen3.5-vision-rt4' if _vision_backend() == 'local'
                               else None)),
        'document_extract': True,
        'context': CTX_LIMIT,                 # KV cache 容量（token）
        'max_tokens': DEFAULT_MAX_TOKENS,     # 请求不带 max_tokens 时的生成上限
        'max_upload_mb': MAX_UPLOAD_MB,
        'max_extract_chars': MAX_EXTRACT_CHARS,
        'formats': ['txt', 'md', 'json', 'csv', 'html', 'xml', 'pdf', 'docx',
                    'pptx', 'xlsx', 'rtf', 'odt', '代码/日志等纯文本'],
    }


@app.post('/v1/extract')
async def extract_file(request: Request):
    """原始文件字节进、提取文本出；不依赖 python-multipart。"""
    name = (request.query_params.get('name')
            or request.headers.get('x-filename')
            or 'upload.txt')
    length = request.headers.get('content-length')
    if length and length.isdigit() and int(length) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f'文件超过 {MAX_UPLOAD_MB}MB 上限')
    data = await request.body()
    if not data:
        raise HTTPException(400, '空文件')
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f'文件超过 {MAX_UPLOAD_MB}MB 上限')
    try:
        text, fmt, warning = extract_document(name, data)
    except ValueError as e:
        raise HTTPException(415, str(e)) from e
    except (zipfile.BadZipFile, ET.ParseError, UnicodeError, KeyError) as e:
        raise HTTPException(422, f'文件解析失败：{e}') from e
    original_chars = len(text)
    if original_chars > MAX_EXTRACT_CHARS:
        text = text[:MAX_EXTRACT_CHARS]
        warning = ((warning + ' ') if warning else '') + \
                  f'内容超过 {MAX_EXTRACT_CHARS} 字符，已截断。'
    return {
        'name': name,
        'format': fmt,
        'chars': original_chars,
        'returned_chars': len(text),
        'truncated': original_chars > MAX_EXTRACT_CHARS,
        'warning': warning,
        'text': text,
    }


@app.get('/v1/files')
def files_api(conversation_id: str = 'default'):
    conv = _conversation_id({'conversation_id': conversation_id})
    return {'conversation_id': conv, 'files': _file_list(conv)}


@app.get('/v1/files/{name}')
def file_download(name: str, conversation_id: str = 'default', download: int = 0):
    conv = _conversation_id({'conversation_id': conversation_id})
    try:
        path = _file_path(conv, name)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    if not os.path.isfile(path):
        raise HTTPException(404, '文件不存在')
    mime = mimetypes.guess_type(path)[0] or 'application/octet-stream'
    if name.lower().endswith(('.md', '.txt', '.log', '.csv', '.json', '.py', '.sh')):
        mime = 'text/plain; charset=utf-8'
    return FileResponse(path, media_type=mime, filename=name,
                        content_disposition_type='attachment' if download else 'inline')


@app.delete('/v1/files/{name}')
def file_delete(name: str, conversation_id: str = 'default'):
    conv = _conversation_id({'conversation_id': conversation_id})
    try:
        _delete_file(conv, name)
    except ValueError as e:
        raise HTTPException(404, str(e)) from e
    return {'ok': True, 'deleted': name}


@app.post('/v1/chat/completions')
async def chat(body: dict):
    return await _complete(body)


@app.post('/chat/completions')
async def chat_alias(body: dict):
    """兼容只在 Base URL 后拼 /chat/completions 的客户端。"""
    return await _complete(body)


@app.post('/v1/completions')
async def completions(body: dict):
    return await _complete(body)


@app.post('/completions')
async def completions_alias(body: dict):
    return await _complete(body)


IMAGE_TOKEN_ID = 248056


def _image_token_spans(ids, embeds):
    """在 token 序列里找连续的 <|image_pad|> 段，按顺序与视觉塔输出配对。"""
    spans = []
    i = 0
    while i < len(ids):
        if ids[i] != IMAGE_TOKEN_ID:
            i += 1
            continue
        j = i
        while j < len(ids) and ids[j] == IMAGE_TOKEN_ID:
            j += 1
        spans.append((i, j - i))
        i = j
    if len(spans) != len(embeds):
        raise HTTPException(
            500, f'图片占位符 {len(spans)} 段，但视觉塔输出 {len(embeds)} 个；'
                 f'检查 chat template 是否保留了 <|image_pad|>。')
    for idx, ((_, count), emb) in enumerate(zip(spans, embeds), 1):
        if count != len(emb):
            raise HTTPException(
                500, f'第 {idx} 张图片占位符 {count} 个 token，'
                     f'视觉塔输出 {len(emb)} 个 token。')
    return spans


def _write_embeddings_file(embeds):
    import numpy as np
    fd, path = tempfile.mkstemp(prefix='rt4-vision-', suffix='.f32',
                                dir=os.environ.get('TMPDIR', '/tmp'))
    with os.fdopen(fd, 'wb') as f:
        for emb in embeds:
            f.write(np.asarray(emb, dtype='<f4').tobytes(order='C'))
    return path


async def _prefill_with(ids, embeds):
    """按需把视觉 embedding 写临时文件并执行 PREFILL / PREFILL_EMB。"""
    spans = _image_token_spans(ids, embeds) if embeds else []
    emb_path = _write_embeddings_file(embeds) if embeds else None
    try:
        if embeds:
            st = await asyncio.to_thread(ENGINE.prefill_emb, ids, emb_path, spans)
        else:
            st = await asyncio.to_thread(ENGINE.prefill, ids)
    finally:
        if emb_path:
            try:
                os.unlink(emb_path)
            except OSError:
                pass
    if not st or st.startswith('ERR'):
        raise HTTPException(500, f'运行时 prefill 失败：{st}')
    return st


# ------------------------------ 技能 / 工作区 ------------------------------
def _conversation_id(body):
    raw = str(body.get('conversation_id') or body.get('session_id') or 'default')
    safe = re.sub(r'[^A-Za-z0-9_.-]', '_', raw).strip('._')[:64]
    return safe or 'default'


def _workspace(conv):
    path = os.path.join(WORKSPACE_ROOT, conv)
    os.makedirs(path, exist_ok=True)
    return path


def _safe_name(name):
    name = os.path.basename(str(name or '')).strip()
    name = re.sub(r'[\\/:*?"<>|\x00-\x1f]', '_', name)
    if not name or name in ('.', '..'):
        raise ValueError('非法文件名')
    if len(name) > 128:
        raise ValueError('文件名过长')
    return name


def _file_path(conv, name):
    path = os.path.join(_workspace(conv), _safe_name(name))
    root = os.path.realpath(_workspace(conv))
    if os.path.realpath(path) != os.path.join(root, os.path.basename(path)):
        raise ValueError('路径越界')
    return path


def _file_list(conv):
    root = _workspace(conv)
    files = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isfile(path):
            continue
        st = os.stat(path)
        files.append({'name': name, 'size': st.st_size,
                      'modified': int(st.st_mtime),
                      'ext': os.path.splitext(name)[1].lower().lstrip('.')})
    return files


def _write_file(conv, name, content, append=False):
    path = _file_path(conv, name)
    data = str(content).encode('utf-8')
    if len(data) > MAX_FILE_BYTES:
        raise ValueError(f'文件超过 {MAX_FILE_BYTES // 1024}KB 上限')
    used = sum(os.path.getsize(os.path.join(_workspace(conv), f))
               for f in os.listdir(_workspace(conv))
               if os.path.isfile(os.path.join(_workspace(conv), f)))
    old = os.path.getsize(path) if os.path.exists(path) else 0
    if used - old + len(data) > MAX_WORKSPACE_BYTES:
        raise ValueError(f'工作区超过 {MAX_WORKSPACE_BYTES // 1024 // 1024}MB 上限')
    mode = 'ab' if append else 'wb'
    with open(path, mode) as f:
        f.write(data)
    return path


def _read_file(conv, name, max_chars=20000):
    path = _file_path(conv, name)
    if not os.path.exists(path):
        raise ValueError('文件不存在')
    with open(path, 'rb') as f:
        raw = f.read(max(1, int(max_chars)) * 4)
    return _decode_text_bytes(raw)[:int(max_chars)]


def _delete_file(conv, name):
    path = _file_path(conv, name)
    if not os.path.exists(path):
        raise ValueError('文件不存在')
    os.remove(path)


def _eval_expr(expr):
    expr = str(expr)[:500]
    ops = {
        ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b,
        ast.FloorDiv: lambda a, b: a // b, ast.Mod: lambda a, b: a % b,
        ast.Pow: lambda a, b: a ** b,
        ast.USub: lambda a: -a, ast.UAdd: lambda a: +a,
    }

    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](walk(node.operand))
        raise ValueError('表达式只支持数字和四则/乘方')

    return walk(ast.parse(expr, mode='eval'))


def _parse_param_value(value):
    value = value.strip()
    if value and (value[0] in '[{') or value.lower() in ('true', 'false', 'null'):
        try:
            return json.loads(value)
        except ValueError:
            pass
    return value


def _tool_calls(text):
    calls = []
    for block in re.findall(r'<tool_call>\s*(.*?)\s*(?:</tool_call>|$)', text, re.S):
        block = block.strip()
        if block.startswith('{'):
            try:
                obj = json.loads(block)
            except ValueError:
                continue
            fn = obj.get('name') or (obj.get('function') or {}).get('name')
            args = obj.get('arguments') or (obj.get('function') or {}).get('arguments') or {}
            if fn:
                calls.append({'name': fn, 'arguments': args})
            continue
        m = re.search(r'<function=([^>\s]+)>\s*(.*?)\s*</function>', block, re.S)
        if not m:
            continue
        args = {}
        for key, val in re.findall(r'<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>',
                                   m.group(2), re.S):
            args[key] = _parse_param_value(val)
        calls.append({'name': m.group(1), 'arguments': args})
    return calls


def _strip_tool_calls(text):
    return re.sub(r'<tool_call>.*?(?:</tool_call>|$)', '', text, flags=re.S).strip()


def _execute_skill(conv, call):
    name = call.get('name')
    args = call.get('arguments') or {}
    if name not in SKILL_NAMES:
        return f'错误：未知技能 {name}'
    try:
        if name == 'now':
            return '当前时间：' + datetime.datetime.now().astimezone().isoformat()
        if name == 'calc':
            val = _eval_expr(args.get('expression', ''))
            return f'{args.get("expression")} = {val:g}' if isinstance(val, float) else \
                   f'{args.get("expression")} = {val}'
        if name == 'write_file':
            path = _write_file(conv, args.get('name'), args.get('content', ''),
                               bool(args.get('append')))
            return json.dumps({'ok': True, 'file': os.path.basename(path),
                               'bytes': os.path.getsize(path)}, ensure_ascii=False)
        if name == 'read_file':
            return _read_file(conv, args.get('name'), int(args.get('max_chars', 20000)))
        if name == 'list_files':
            return json.dumps(_file_list(conv), ensure_ascii=False)
        if name == 'delete_file':
            _delete_file(conv, args.get('name'))
            return json.dumps({'ok': True, 'deleted': args.get('name')}, ensure_ascii=False)
        if name == 'make_csv':
            name_ = args.get('name', 'data.csv')
            cols = args.get('columns') or []
            rows = args.get('rows') or []
            if isinstance(cols, str):
                cols = [c.strip() for c in cols.split(',') if c.strip()]
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(cols)
            for row in rows:
                writer.writerow(row if isinstance(row, (list, tuple)) else [row])
            path = _write_file(conv, name_, buf.getvalue())
            return json.dumps({'ok': True, 'file': os.path.basename(path),
                               'bytes': os.path.getsize(path)}, ensure_ascii=False)
    except Exception as e:                                   # noqa: BLE001
        return f'技能执行失败：{e}'
    return '技能没有返回结果'


def _tools_for(body):
    if not body.get('skills'):
        return body.get('tools') or []
    names = body.get('skill_names')
    return [s for s in SKILLS if not names or s['function']['name'] in names]


def _workspace_note(conv):
    files = _file_list(conv)
    if not files:
        return ''
    lines = ['当前会话工作区已有文件：']
    for f in files[:50]:
        lines.append(f'- {f["name"]} ({f["size"]} bytes)')
    lines.append('需要文件内容时请调用 read_file；需要新增/修改时请调用 write_file。')
    return '\n'.join(lines)


async def _complete(body: dict):
    tools = _tools_for(body)
    if tools:
        return await _complete_with_skills(body, tools)
    n, temp, top_p, top_k, seed = sampling(body)
    # 文档/图片整理、视觉桥调用和 tokenizer 都不占 GPU；放线程里避免卡住事件循环。
    ids, text, embeds = await asyncio.to_thread(build_ids_and_embeds, body)
    if not ids:
        raise HTTPException(400, '空 prompt')
    if ENGINE is None:
        raise HTTPException(503, '模型还在加载')
    if len(ids) + n > CTX_LIMIT:
        # max_tokens 是「最多生成多少」的上限，不是要预留一整块上下文：装不下就按
        # 剩余上下文收窄（客户端发 128000 是常态）。只有 prompt 本身就超了才报错。
        n = clamp_max_tokens(len(ids), n)
    spans = _image_token_spans(ids, embeds) if embeds else []
    emb_path = None
    if embeds:
        emb_path = _write_embeddings_file(embeds)
    # 模型名不参与路由：客户端填什么就接受什么，并在响应里原样回显。
    # 这样 Open WebUI / Cline / 各类 OpenAI SDK 随便填 model 都能调用。
    req_model = str(body.get('model') or MODEL_NAME)
    stream = bool(body.get('stream'))
    rid = 'chatcmpl-' + uuid.uuid4().hex[:24]
    created = int(time.time())
    stops = list(body.get('stop') or []) + EOS
    out = []
    async with LOCK:
        if 'mtp' in body:
            try:
                ENGINE.set_mtp(max(0, min(3, int(body.get('mtp') or 0))))
            except Exception as e:                       # noqa: BLE001
                print(f'[serve] set_mtp 失败: {e}', file=sys.stderr)
        t0 = time.time()
        try:
            if embeds:
                st = await asyncio.to_thread(ENGINE.prefill_emb, ids, emb_path, spans)
            else:
                st = await asyncio.to_thread(ENGINE.prefill, ids)
        finally:
            if emb_path:
                try:
                    os.unlink(emb_path)
                except OSError:
                    pass
        if not st or st.startswith('ERR'):
            raise HTTPException(500, f'运行时不接受带视觉 embedding 的 prefill：{st}')
        t1 = time.time()
        STATS['reqs'] += 1
        STATS['prefill_tokens'] += len(ids)
        STATS['prefill_ms'] += (t1 - t0) * 1000

        if not stream:
            toks, end = await asyncio.to_thread(ENGINE.gen, n, temp, top_p, top_k, seed, stops)
            out = toks
            t2 = time.time()
            STATS['gen_tokens'] += len(toks)
            STATS['gen_ms'] += (t2 - t1) * 1000
            txt = T.decode(toks)
            reasoning, answer = split_thinking(txt, thinking_enabled(body))
            usage = {'prompt_tokens': len(ids), 'completion_tokens': len(toks),
                     'total_tokens': len(ids) + len(toks)}
            pms = (t1 - t0) * 1000
            gms = (t2 - t1) * 1000
            timings = {'prefill_tokens': len(ids), 'prefill_ms': round(pms, 1),
                       'prefill_tps': round(len(ids) / max(1e-6, t1 - t0), 1),
                       'completion_tokens': len(toks), 'gen_ms': round(gms, 1),
                       'gen_tps': round(len(toks) / max(1e-6, t2 - t1), 1),
                       'ttft_ms': None, 'max_tokens': n}
            return _resp(rid, created, answer, usage, req_model, timings=timings,
                         reasoning=reasoning)

    # 流式：每个 token 一行，增量 detokenize
    think = thinking_enabled(body)

    async def gen():
        yield _sse(rid, created, {'role': 'assistant', 'content': ''}, model=req_model)
        acc, prev = [], ''
        timings = None
        sent_reason = 0
        tag_pos = -1
        sent_content = 0
        content_started = False
        try:
            async with LOCK:
                tg0 = time.time()
                ttft = None
                st = ENGINE.gen_stream(n, temp, top_p, top_k, seed, stops)
                try:
                    while True:
                        line = await asyncio.to_thread(st.get)
                        if not line:
                            break
                        line = line.strip()
                        if line.startswith('TOK '):
                            tok = int(line[4:])
                            acc.append(tok)
                            if ttft is None:
                                ttft = time.time()
                            cur = T.decode(acc)
                            if not think:
                                # 关闭思考：全部是正式回答
                                if len(cur) > len(prev):
                                    yield _sse(rid, created, {'content': cur[len(prev):]},
                                               model=req_model)
                            else:
                                if tag_pos < 0:
                                    i = cur.find(THINK_END)
                                    if i >= 0:
                                        tag_pos = i
                                        if i > sent_reason:
                                            d = cur[sent_reason:i]
                                            yield _sse(rid, created,
                                                       {'reasoning_content': d, 'reasoning': d},
                                                       model=req_model)
                                            sent_reason = i
                                        sent_content = i + len(THINK_END)
                                        content_started = False
                                    elif len(cur) > sent_reason:
                                        d = cur[sent_reason:]
                                        yield _sse(rid, created,
                                                   {'reasoning_content': d, 'reasoning': d},
                                                   model=req_model)
                                        sent_reason = len(cur)
                                if tag_pos >= 0:
                                    if not content_started:
                                        while (sent_content < len(cur) and
                                               cur[sent_content] in '\r\n'):
                                            sent_content += 1
                                        if sent_content < len(cur):
                                            content_started = True
                                            yield _sse(rid, created,
                                                       {'content': cur[sent_content:]},
                                                       model=req_model)
                                            sent_content = len(cur)
                                    elif len(cur) > sent_content:
                                        yield _sse(rid, created,
                                                   {'content': cur[sent_content:]},
                                                   model=req_model)
                                        sent_content = len(cur)
                            prev = cur
                        elif line.startswith('END '):
                            break
                except (asyncio.CancelledError, GeneratorExit):
                    # 网页按了「停止」/ 断开连接：让引擎在本轮结束后停下，并等它
                    # 把 END 打完再释放请求锁（否则残留输出会串到下一个请求）。
                    ENGINE.stop()
                    st.join(5.0)
                    raise
                tg1 = time.time()
            pms = (t1 - t0) * 1000
            gms = (tg1 - tg0) * 1000
            timings = {'prefill_tokens': len(ids), 'prefill_ms': round(pms, 1),
                       'prefill_tps': round(len(ids) / max(1e-6, t1 - t0), 1),
                       'completion_tokens': len(acc), 'gen_ms': round(gms, 1),
                       'gen_tps': round(len(acc) / max(1e-6, tg1 - tg0), 1),
                       'ttft_ms': None if ttft is None else round((ttft - tg0) * 1000, 1),
                       'max_tokens': n}       # 收窄后的实际生成上限
            STATS['gen_tokens'] += len(acc)
            STATS['gen_ms'] += gms
            yield _sse(rid, created, {}, model=req_model,
                       extra={'timings': timings,
                              'usage': {'prompt_tokens': len(ids),
                                        'completion_tokens': len(acc),
                                        'total_tokens': len(ids) + len(acc)}})
        finally:
            yield _sse(rid, created, {}, 'stop', model=req_model)
            yield 'data: [DONE]\n\n'

    return StreamingResponse(gen(), media_type='text/event-stream')


async def _skill_events(body, tools):
    """技能循环的核心：把「多轮工具调用」跑成一条事件流（流式/非流式共用）。

    产出：
      ('delta', text)      正式回答的增量（边生成边发，页面不再等整段生成完）
      ('reasoning', text)  思考段增量
      ('reset', None)      本轮已流出的文字作废（该轮其实是工具调用），前端应清空
      ('skill', ev)        一次技能调用与结果
      ('done', payload)    收尾：完整文本 / reasoning / usage / timings / skills
      ('streamed', bool)   收尾附带：最终文本是否已经流出去过（避免重复发送）
    """
    n, temp, top_p, top_k, seed = sampling(body)
    conv = _conversation_id(body)
    think = thinking_enabled(body)
    msgs = [m for m in (body.get('messages') or []) if isinstance(m, dict)]
    note = _workspace_note(conv)
    if note:
        msgs = [{'role': 'system', 'content': note}] + msgs
    stops = list(body.get('stop') or []) + EOS
    events, final_text, final_reasoning = [], '', ''
    prefill_ms = gen_ms = 0.0
    prefill_tokens = completion_tokens = 0
    streamed = False
    if os.environ.get('RT_DEBUG_SKILL'):
        print(f'[dbg] 技能循环：think={think} max_tokens={n} tools={len(tools)}',
              file=sys.stderr, flush=True)

    async with LOCK:
        if 'mtp' in body:
            try:
                ENGINE.set_mtp(max(0, min(3, int(body.get('mtp') or 0))))
            except Exception as e:                           # noqa: BLE001
                print(f'[serve] set_mtp 失败: {e}', file=sys.stderr)
        for rnd in range(MAX_SKILL_ROUNDS):
            work = dict(body)
            work['messages'] = msgs
            work['tools'] = tools
            ids, rendered, embeds = await asyncio.to_thread(build_ids_and_embeds, work)
            if not ids:
                raise HTTPException(400, '空 prompt')
            if len(ids) + n > CTX_LIMIT:
                n = clamp_max_tokens(len(ids), n)
            t0 = time.time()
            await _prefill_with(ids, embeds)
            t1 = time.time()
            # 逐 token 读引擎输出：边生成边发，避免「等整段生成完才出现」
            toks = []
            split = ThinkSplitter(think)
            st = ENGINE.gen_stream(n, temp, top_p, top_k, seed, stops)
            try:
                while True:
                    line = await asyncio.to_thread(st.get)
                    if not line:
                        break
                    line = line.strip()
                    if line.startswith('TOK '):
                        toks.append(int(line[4:]))
                        for kind, text in split.feed(T.decode(toks)):
                            # 切分器给的是 'content'/'reasoning'，对外统一叫 'delta'
                            yield ('delta' if kind == 'content' else 'reasoning', text)
                    elif line.startswith('END '):
                        break
            except (asyncio.CancelledError, GeneratorExit):
                ENGINE.stop()          # 同普通路径：停引擎并等它收尾，再放锁
                st.join(5.0)
                raise
            t2 = time.time()
            full = T.decode(toks)
            if os.environ.get('RT_DEBUG_SKILL'):
                print(f'[dbg] 轮 {rnd}: {len(toks)} tok, 思考标记={THINK_END in full}, '
                      f'内容={split.sent_content}, 思考={split.sent_reason}',
                      file=sys.stderr, flush=True)
            if THINK_END in full:
                reasoning, answer = split_thinking(full, True)
            else:
                reasoning, answer = '', full
            calls = _tool_calls(full)
            prefill_ms += (t1 - t0) * 1000
            gen_ms += (t2 - t1) * 1000
            prefill_tokens += len(ids)
            completion_tokens += len(toks)
            if not calls:
                final_text = (_strip_tool_calls(answer) or
                              _strip_tool_calls(full) or '已完成。')
                final_reasoning = reasoning
                streamed = True                 # 这一轮的文本已经流出去了
                break
            assistant = {'role': 'assistant', 'content': answer or full}
            if reasoning:
                assistant['reasoning_content'] = reasoning
            msgs.append(assistant)
            # 这一轮是工具调用：刚流出去的是调用片段，作废掉（前端清空气泡文字），
            # 再把技能事件发出去，然后带着工具结果进入下一轮。
            yield ('reset', None)
            for call in calls:
                result = _execute_skill(conv, call)
                ev = {
                    'name': call.get('name'), 'arguments': call.get('arguments') or {},
                    'ok': not str(result).startswith('技能执行失败'),
                    'result': str(result)[:2000],
                }
                events.append(ev)
                yield ('skill', ev)
                msgs.append({'role': 'tool', 'content': str(result)})
        else:
            yield ('reset', None)
            final_text = f'已达到技能调用轮数上限（{MAX_SKILL_ROUNDS} 轮），请继续下一轮对话。'

    STATS['reqs'] += 1
    STATS['prefill_tokens'] += prefill_tokens
    STATS['gen_tokens'] += completion_tokens
    STATS['prefill_ms'] += prefill_ms
    STATS['gen_ms'] += gen_ms
    usage = {'prompt_tokens': prefill_tokens, 'completion_tokens': completion_tokens,
             'total_tokens': prefill_tokens + completion_tokens}
    timings = {'prefill_tokens': prefill_tokens, 'prefill_ms': round(prefill_ms, 1),
               'prefill_tps': round(prefill_tokens / max(1e-6, prefill_ms / 1000), 1),
               'completion_tokens': completion_tokens, 'gen_ms': round(gen_ms, 1),
               'gen_tps': round(completion_tokens / max(1e-6, gen_ms / 1000), 1),
               'ttft_ms': None, 'max_tokens': n}
    yield ('done', {'text': final_text, 'reasoning': final_reasoning, 'usage': usage,
                    'timings': timings, 'skills': events, 'streamed': streamed})


async def _complete_with_skills(body, tools):
    """技能循环入口：stream=True 走 SSE，否则收干事件流拼成一条 JSON 响应。"""
    req_model = str(body.get('model') or MODEL_NAME)
    stream = bool(body.get('stream'))
    rid = 'chatcmpl-' + uuid.uuid4().hex[:24]
    created = int(time.time())

    if not stream:
        text, reasoning, events, usage, timings = '', '', [], None, None
        async for kind, val in _skill_events(body, tools):
            if kind == 'done':
                text, reasoning = val['text'], val['reasoning']
                events, usage, timings = val['skills'], val['usage'], val['timings']
        out = _resp(rid, created, text, usage or {}, req_model,
                    timings=timings, reasoning=reasoning)
        out['skills'] = events
        return out

    async def gen():
        yield _sse(rid, created, {'role': 'assistant', 'content': ''}, model=req_model)
        sent = ''                            # 已经发给前端的正式回答
        try:
            async for kind, val in _skill_events(body, tools):
                if kind == 'delta':
                    sent += val
                    yield _sse(rid, created, {'content': val}, model=req_model)
                elif kind == 'reasoning':
                    yield _sse(rid, created,
                               {'reasoning_content': val, 'reasoning': val}, model=req_model)
                elif kind == 'reset':
                    sent = ''
                    yield _sse(rid, created, {}, model=req_model,
                               extra={'type': 'round_reset'})
                elif kind == 'skill':
                    yield _sse(rid, created, {}, model=req_model,
                               extra={'type': 'skill', 'skill': val})
                elif kind == 'done':
                    # 兜底：最终文本与已流出的内容不一致（罕见）时补发前缀差
                    if not val['streamed'] or val['text'] != sent:
                        if sent and not val['text'].startswith(sent):
                            yield _sse(rid, created, {}, model=req_model,
                                       extra={'type': 'round_reset'})
                            sent = ''
                        rest = val['text'][len(sent):] if val['text'].startswith(sent) else val['text']
                        for i in range(0, len(rest), 24):
                            yield _sse(rid, created, {'content': rest[i:i + 24]},
                                       model=req_model)
                    yield _sse(rid, created, {}, model=req_model,
                               extra={'timings': val['timings'], 'usage': val['usage'],
                                      'skills': val['skills']})
        except HTTPException as e:
            # 生成已经开始（已经发了 200 和 role 分片）不能再改状态码，用事件告诉前端
            yield _sse(rid, created, {}, model=req_model,
                       extra={'type': 'error', 'message': str(e.detail)})
        yield _sse(rid, created, {}, 'stop', model=req_model)
        yield 'data: [DONE]\n\n'

    return StreamingResponse(gen(), media_type='text/event-stream')


def _resp(rid, created, text, usage, model, timings=None, reasoning=None):
    msg = {'role': 'assistant', 'content': text}
    if reasoning:
        # 两种常见字段名都带上，兼容 Cherry Studio / Open WebUI / OpenRouter 风格客户端
        msg['reasoning_content'] = reasoning
        msg['reasoning'] = reasoning
    out = {'id': rid, 'object': 'chat.completion', 'created': created, 'model': model,
           'choices': [{'index': 0, 'message': msg,
                        'finish_reason': 'stop'}], 'usage': usage}
    if timings:
        out['timings'] = timings
    return out


def _sse(rid, created, delta, finish=None, model=None, extra=None):
    chunk = {'id': rid, 'object': 'chat.completion.chunk', 'created': created,
             'model': model or MODEL_NAME, 'choices': [{'index': 0, 'delta': delta,
                                                       'finish_reason': finish}]}
    if extra:
        chunk.update(extra)
    return 'data: ' + json.dumps(chunk, ensure_ascii=False) + '\n\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8080)
    ap.add_argument('--host', default='0.0.0.0')
    # serve.sh 一直传 --ctx，但这里以前没定义 → argparse 报错、容器反复重启
    ap.add_argument('--ctx', type=int, default=131072, help='KV cache 容量（token）')
    ap.add_argument('--default-max-tokens', type=int, default=128000,
                    help='请求体没带 max_tokens 时的生成上限（默认 128000，'
                         '会按剩余上下文收窄；显式给 max_tokens 时以请求为准）')
    ap.add_argument('--mtp-n', type=int, default=None,
                    help='MTP 草稿数 0..3（默认 3；传 0 只关闭投机但仍加载权重）')
    ap.add_argument('--no-mtp', action='store_true', help='不加载 MTP 权重')
    args = ap.parse_args()
    global ENGINE, CTX_LIMIT, DEFAULT_MAX_TOKENS
    CTX_LIMIT = args.ctx
    DEFAULT_MAX_TOKENS = args.default_max_tokens
    ENGINE = Engine(log=True, ctx=args.ctx, mtp_n=args.mtp_n, no_mtp=args.no_mtp)
    # 本地视觉前端会 import torch/transformers；后台预热，避免第一张图片白等十几秒。
    if _vision_backend() == 'local':
        threading.Thread(target=_get_local_encoder, daemon=True).start()
    print(f'[serve] 模型就绪，监听 http://{args.host}:{args.port}/v1', flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level='warning')


if __name__ == '__main__':
    main()
