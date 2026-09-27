#!/usr/bin/env python3
# 逐张量校验 RT4 里的 **非量化**（f16/f32）张量：与源 safetensors 反量化后逐元素比。
# 这个检查是必须的：转换器把 BF16 源写成 f32 时曾经按 4 字节原样搬，导致
# norm / A_log / dt_bias / conv1d / ssm_norm 全部错位（模型残差被放大 10 倍），
# 而"运行时 vs 同源 RT4 参考"的单层对照**根本发现不了**（两边用的是同一份坏权重）。
#
#   python3 tools/verify_plain.py <model.safetensors> <rt4> <rt4.json>
import json
import sys

import numpy as np

sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from rt4_io import RT4                        # noqa: E402
from src_weights import Src                   # noqa: E402


def main():
    st, rt4p, jsp = sys.argv[1], sys.argv[2], sys.argv[3]
    R = RT4(rt4p, jsp)
    S = Src(st)
    worst = []
    n_ok = n_bad = 0
    for name, t in R.d.items():
        if t['kind'] not in ('f16', 'f32'):
            continue
        if not S.has(name):
            print(f'  [跳过] 源里没有 {name}')
            continue
        a = R.dequant(name).reshape(-1).astype(np.float64)
        b = S.deq(name).reshape(-1).astype(np.float64)
        if a.size != b.size:
            print(f'  [形状不符] {name}: rt4={a.size} src={b.size}')
            n_bad += 1
            continue
        denom = np.linalg.norm(b)
        rel = np.linalg.norm(a - b) / denom if denom > 0 else np.linalg.norm(a - b)
        if rel < 1e-3:
            n_ok += 1
        else:
            n_bad += 1
            worst.append((rel, name))
    print(f'非量化张量：一致 {n_ok}，异常 {n_bad}')
    worst.sort(reverse=True)
    for rel, name in worst[:15]:
        print(f'  rel={rel:.3e}  {name}')
    return 1 if n_bad else 0


if __name__ == '__main__':
    sys.exit(main())
