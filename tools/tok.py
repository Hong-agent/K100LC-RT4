#!/usr/bin/env python3
# 分词器 + chat 模板（用容器里现成的 tokenizers/transformers；本项目不重写 BPE）。
#
#   python3 tools/tok.py encode "你好，世界"
#   python3 tools/tok.py decode 100 200 300
#   python3 tools/tok.py chat '[{"role":"user","content":"你好"}]'
#   python3 tools/tok.py serve            # stdin 逐行 JSON 协议（给 HTTP 服务用）
import json
import os
import sys

MODEL_DIR = os.environ.get('RT_MODEL_DIR',
                           os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        '..', 'models', 'Qwen3.8-27B-NVFP4'))

from tokenizers import Tokenizer                                    # noqa: E402

_tok = None
_tmpl = None


def tok():
    global _tok
    if _tok is None:
        _tok = Tokenizer.from_file(os.path.join(MODEL_DIR, 'tokenizer.json'))
    return _tok


def template():
    global _tmpl
    if _tmpl is None:
        from jinja2 import Environment
        src = open(os.path.join(MODEL_DIR, 'chat_template.jinja'), encoding='utf-8').read()
        _tmpl = Environment(trim_blocks=True, lstrip_blocks=True).from_string(src)
    return _tmpl


def apply_chat(messages, add_generation_prompt=True, **kw):
    """按模型自带的 chat_template.jinja 渲染（HF 的 apply_chat_template 语义）。"""
    return template().render(messages=messages, add_generation_prompt=add_generation_prompt,
                             **kw)


def encode(text, add_special=False):
    return tok().encode(text, add_special_tokens=add_special).ids


def decode(ids, skip_special=True):
    return tok().decode(list(ids), skip_special_tokens=skip_special)


def chat_ids(messages, add_generation_prompt=True, **kw):
    return encode(apply_chat(messages, add_generation_prompt, **kw))


def _serve():
    """stdin 一行一个 JSON 请求 → stdout 一行一个 JSON 响应。
    请求: {"op":"encode","text":...} {"op":"decode","ids":[...]}
          {"op":"chat","messages":[...],"add_generation_prompt":true}
    """
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            op = req.get('op')
            if op == 'encode':
                out = {'ids': encode(req['text'], req.get('add_special', False))}
            elif op == 'decode':
                out = {'text': decode(req['ids'], req.get('skip_special', True))}
            elif op == 'chat':
                out = {'ids': chat_ids(req['messages'], req.get('add_generation_prompt', True)),
                       'text': apply_chat(req['messages'], req.get('add_generation_prompt', True))}
            elif op == 'eos':
                out = {'eos': [248044, 248046], 'im_start': 248045, 'im_end': 248046}
            else:
                out = {'error': f'unknown op {op}'}
        except Exception as e:                       # noqa: BLE001
            out = {'error': f'{type(e).__name__}: {e}'}
        sys.stdout.write(json.dumps(out, ensure_ascii=False) + '\n')
        sys.stdout.flush()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    cmd = sys.argv[1]
    if cmd == 'encode':
        print(json.dumps(encode(sys.argv[2]), ensure_ascii=False))
    elif cmd == 'decode':
        ids = [int(x) for x in sys.argv[2:]]
        print(decode(ids))
    elif cmd == 'chat':
        msgs = json.loads(sys.argv[2])
        print(apply_chat(msgs), end='')
        print('--- ids ---')
        print(json.dumps(chat_ids(msgs)))
    elif cmd == 'serve':
        _serve()
    else:
        print(__doc__)


if __name__ == '__main__':
    main()
