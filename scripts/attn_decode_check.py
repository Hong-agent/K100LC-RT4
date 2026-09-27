#!/usr/bin/env python3
# 解码注意力内核的独立复核（驱动 + 对照一体）：
#   prefill 一段文本 → 走 2 步解码（后一步的 Q/K/V/out dump 会覆盖前一步）
#   → tools/attn_decode_check.py 用 fp64 重算单行 softmax 注意力做对照。
#
#   bash scripts/dsh.sh 'python3 scripts/attn_decode_check.py 3 11'
#   参数：层号（默认 3，是 4 的倍数 +1 → 全注意力层）、预填充 token 数（默认 11）
import os
import subprocess
import sys

ROOT = '/rt'
sys.path.insert(0, ROOT + '/tools')
sys.path.insert(0, ROOT + '/scripts')
import tok as T                                                    # noqa: E402

LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 3
N = int(sys.argv[2]) if len(sys.argv) > 2 else 11
M = os.environ.get('RT_RT4', ROOT + '/models/Qwen3.8-27B-NVFP4/rt4/qwen38_27b.rt4')


def main():
    env = dict(os.environ, RT_DUMP_ATTN=str(LAYER))
    # 复核的是单 token 解码内核（T=1）；显式关掉 MTP，否则 dump 会来自验证批。
    p = subprocess.Popen([ROOT + '/build/rt', '--engine', '--model', M, '--json', M + '.json',
                          '--no-mtp'],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
                         env=env)
    while True:                       # 引擎起不来时要报错，别死等
        line = p.stdout.readline()
        if not line:
            raise SystemExit('引擎启动失败：检查权重是否存在（%s）' % M)
        if line.startswith('READY'):
            break
    # 注意要按 N 铺够长度（只 [:N] 会得到 ~28 个 token，长上下文检查就白做了）
    text = '注意力内核需要一段真实的前缀文本来填出 KV 缓存，并且长度要足够。'
    ids = T.encode(text * (N // 12 + 1))[:N]
    p.stdin.write('PREFILL ' + ','.join(map(str, ids)) + '\n')
    p.stdin.flush()
    p.stdout.readline()
    # 不传 stop 词：否则模型长前缀时第一步就吐 <|im_end|>，GEN 提前退出，dump 会停在
    # prefill 那一步（曾经让这个检查在长上下文下假报错）。
    p.stdin.write('GEN 3 0 1 0 1\n')                     # 2 次 forward → n_kv = N+2
    p.stdin.flush()
    end = ''
    while True:
        line = p.stdout.readline()
        if not line or line.startswith('END'):
            end = line.strip()
            break
    p.stdin.write('QUIT\n')
    p.wait(timeout=20)
    print('GEN: %s' % end)

    # dump 来自最后一次 forward（层 LAYER，T=1）
    subprocess.run([sys.executable, ROOT + '/tools/attn_decode_check.py', str(N + 2)], check=True)


if __name__ == '__main__':
    main()
