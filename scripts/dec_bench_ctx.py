#!/usr/bin/env python3
# 解码速率 vs 上下文长度（每 token 的墙钟）。在 DTK 容器里跑：
#   bash scripts/dsh.sh 'python3 scripts/dec_bench_ctx.py 60,600,2000'
#
# 用途：权重读取是常数开销，而注意力（现在按 64 行补齐做）随上下文线性增长；
# 这个表能一眼看出从哪个上下文开始 FA 解码成为瓶颈。
import os
import subprocess
import sys
import time

ROOT = '/rt'
sys.path.insert(0, ROOT + '/tools')
import tok as T                                              # noqa: E402

CTXS = [int(x) for x in (sys.argv[1].split(',') if len(sys.argv) > 1 else ['60', '600', '2000'])]
NGEN = int(sys.argv[2]) if len(sys.argv) > 2 else 16
M = os.environ.get('RT_RT4', ROOT + '/models/Qwen3.8-27B-NVFP4/rt4/qwen38_27b.rt4')
FILLER = ('The history of computing began with mechanical calculators and continued through '
          'the industrial revolution. ')


def main():
    p = subprocess.Popen([ROOT + '/build/rt', '--engine', '--model', M, '--json', M + '.json'],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    while True:                       # 引擎起不来时要报错，别死等
        line = p.stdout.readline()
        if not line:
            raise SystemExit('引擎启动失败：检查权重是否存在（%s）' % M)
        if line.startswith('READY'):
            break
    for n in CTXS:
        ids = T.encode(FILLER * (n // 12 + 1))[:n]
        p.stdin.write('PREFILL ' + ','.join(map(str, ids)) + '\n')
        p.stdin.flush()
        p.stdout.readline()
        t0 = time.time()
        p.stdin.write('GEN %d 0 1 0 1 248044,248046\n' % NGEN)
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
        print('ctx %5d  (%4d tok): %6.2f ms/token  %5.1f tok/s' % (n, cnt, 1000 * dt / cnt,
                                                                   cnt / dt))
    p.stdin.write('QUIT\n')


if __name__ == '__main__':
    main()
