#!/usr/bin/env python3
"""MTP 回滚等价性诊断（驱动引擎的 `ROLLTEST2` 协议命令）。

对保留长度 keep=1..4，比较两条路径的 logits：
  A. 逐 token 前向到 probe；
  B. 一次验证批前向（T=4）后回滚到 keep，再前向同一个 probe。
判据：max/mean 差应为 **0**（历史上这里踩过 V-tile 尺度不能跨 tile 回滚、
卷积状态 off-by-one 两个坑，见 docs/MTP.md §4）。

  bash scripts/dsh.sh 'python3 tools/mtp_rolltest.py'
  bash scripts/dsh.sh 'python3 tools/mtp_rolltest.py --ctx 8192 --probe 110827'
"""
import argparse
import os
import subprocess
import sys

ROOT = os.environ.get('RT_ROOT', '/rt')
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import tok as T                                                     # noqa: E402

DEFAULT_PROMPT = '用一句话介绍你自己。'
DEFAULT_TOKENS = '97237,96719,2005,95826'      # 上一步生成出来的 4 个 token（任意 4 个都行）
DEFAULT_PROBE = 110827


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--binary', default=os.path.join(ROOT, 'build', 'rt'))
    ap.add_argument('--ctx', type=int, default=4096)
    ap.add_argument('--prompt', default=DEFAULT_PROMPT)
    ap.add_argument('--tokens', default=DEFAULT_TOKENS, help='验证批里的 4 个候选 token')
    ap.add_argument('--probe', type=int, default=DEFAULT_PROBE, help='回滚后再前向的探针 token')
    args = ap.parse_args()

    model = os.environ.get('RT_RT4', os.path.join(
        ROOT, 'models/Qwen3.8-27B-NVFP4/rt4/qwen38_27b.rt4'))
    p = subprocess.Popen([args.binary, '--engine', '--model', model, '--json', model + '.json',
                          '--ctx', str(args.ctx)],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    while True:                       # 引擎起不来时要报错，别死等
        line = p.stdout.readline()
        if not line:
            raise SystemExit('引擎启动失败：检查权重是否存在（%s）' % model)
        if line.startswith('READY'):
            break
    ids = T.encode(T.apply_chat([{'role': 'user', 'content': args.prompt}]))
    p.stdin.write('PREFILL ' + ','.join(map(str, ids)) + '\n')
    p.stdin.flush()
    p.stdout.readline()
    p.stdin.write('ROLLTEST2 %s %s %d\n'
                  % (','.join(map(str, ids)), args.tokens, args.probe))
    p.stdin.flush()
    ok = True
    for _ in range(4):
        line = p.stdout.readline().strip()
        print(line)
        if 'max=0 ' not in line and not line.endswith('max=0 mean=0'):
            ok = False
    p.stdin.write('QUIT\n')
    p.wait(timeout=30)
    print('回滚等价性:', 'OK（全部 max diff = 0）' if ok else '不一致，见上面输出')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
