#!/usr/bin/env python3
# 解码墙钟基准（给各条优化做 A/B 用）。在 DTK 容器里跑：
#   bash scripts/dsh.sh 'python3 scripts/dec_bench.py 48'
#   bash scripts/dsh.sh 'RT_ACT4=1 python3 scripts/dec_bench.py 48'
#
# 输出：tok/s、ms/token（不含首个 token，因为它是 PREFILL 的 logits）。
import os
import subprocess
import sys
import time

ROOT = '/rt'
sys.path.insert(0, ROOT + '/tools')
import tok as T                                              # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 48
M = os.environ.get('RT_RT4', ROOT + '/models/Qwen3.8-27B-NVFP4/rt4/qwen38_27b.rt4')


def main():
    p = subprocess.Popen([ROOT + '/build/rt', '--engine', '--model', M, '--json', M + '.json'],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    while True:                       # 引擎起不来时要报错，别死等（readline 会一直返回空串）
        line = p.stdout.readline()
        if not line:
            raise SystemExit('引擎启动失败：检查权重是否存在（%s）' % M)
        if line.startswith('READY'):
            break
    ids = T.encode('你好，请用一句话介绍你自己。')
    p.stdin.write('PREFILL ' + ','.join(map(str, ids)) + '\n')
    p.stdin.flush()
    p.stdout.readline()
    t0 = time.time()
    p.stdin.write('GEN %d 0 1 0 1 248044,248046\n' % N)
    p.stdin.flush()
    cnt = 0
    while True:
        line = p.stdout.readline()
        if line.startswith('TOK '):
            cnt += 1
        elif line.startswith('END '):
            break
        elif not line:
            raise SystemExit('engine died')
    dt = time.time() - t0
    tag = ' '.join('%s=%s' % (k, v) for k, v in sorted(os.environ.items())
                   if k.startswith('RT_'))
    print('decode %3d tok: %.1f tok/s  %.2f ms/token  [%s]' % (cnt, cnt / dt, 1000 * dt / cnt, tag))
    p.stdin.write('QUIT\n')


if __name__ == '__main__':
    main()
