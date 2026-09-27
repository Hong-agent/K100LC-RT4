#!/usr/bin/env python3
"""解码注意力的独立复核（fa_decode_k + split-KV）。

把 RT_DUMP_ATTN=<层> 导出的一次**解码步**的 Q/K/V 与打包尺度反解成 f32，
用 fp64 重算单行 softmax 注意力，与内核写出的 attn_out.bin 逐元素对照。
（判据：内核 vs 精确 softmax 应该在 1e-2 以内 —— 上限是 int4 Q/K/V 的量化误差。）

  bash scripts/dsh.sh 'python3 scripts/attn_decode_check.py 3 11'   # 一键跑（含 dump）
  或手工：先跑 scripts/attn_decode_check.py 生成 build/attn_*.bin，再
  python3 tools/attn_decode_check.py <n_kv>
"""
import sys

import numpy as np

H, KV, D, QG, BG = 24, 4, 256, 128, 64
KD, NQG = D // 8, D // QG
n_kv = int(sys.argv[1]) if len(sys.argv) > 1 else 0

qq = np.fromfile('build/attn_qq.bin', np.uint32)
qs = np.fromfile('build/attn_qs.bin', np.float32)
K = np.fromfile('build/attn_k.bin', np.uint32)
Ks = np.fromfile('build/attn_ks.bin', np.float32)
V = np.fromfile('build/attn_v.bin', np.uint32)
Vs = np.fromfile('build/attn_vs.bin', np.float32)
out = np.fromfile('build/attn_out.bin', np.float32)
CAP = K.size // (KV * KD)          # KV cache 的分配容量（行距）
TP = qq.size // (H * KD)
if n_kv == 0:
    n_kv = CAP
print('容量 CAP=%d  Q 补齐行数 TP=%d  参与计算的 key=%d' % (CAP, TP, n_kv))


def unpack(dw):
    """int4 打包（低半字节 = 偶数下标）→ 有符号整数值，形状 [..., dwords*8]"""
    b = dw.astype(np.uint32).view(np.uint8)
    w = np.empty(b.shape[:-1] + (b.shape[-1] * 2,), np.int16)
    w[..., 0::2] = b & 0xF
    w[..., 1::2] = (b >> 4) & 0xF
    return np.where(w >= 8, w - 16, w).astype(np.float64)


Q = np.zeros((H, D))
for h in range(H):
    sc = qs.reshape(H, NQG, TP)[h, :, 0]                 # 只有第 0 行是这一步的 query
    Q[h] = unpack(qq.reshape(H, TP, KD)[h, 0]) * np.repeat(sc, QG)

Kf = np.zeros((KV, n_kv, D))
for h in range(KV):
    sc = Ks.reshape(KV, NQG, CAP)[h, :, :n_kv].T         # [key][组] → 每 128 维一组
    Kf[h] = unpack(K.reshape(KV, CAP, KD)[h, :n_kv]) * np.repeat(sc, QG, axis=1)

Vf = np.zeros((KV, n_kv, D))
for h in range(KV):
    dw = V.reshape(KV, CAP // 8, D)[h]
    codes = np.zeros((CAP // 8, 8, D))
    for e in range(8):                                   # dword 内第 e 个 key
        c = ((dw >> (4 * e)) & 0xF).astype(np.int16)     # 必须转有符号再减 16（uint 会回绕）
        codes[:, e] = np.where(c >= 8, c - 16, c)
    codes = codes.reshape(-1, D)[:n_kv]
    sc = Vs.reshape(KV, -1, D)[h]                        # 每 64 个 key 一个 V 尺度
    Vf[h] = codes * sc[np.arange(n_kv) // BG]

errs = []
for h in range(H):
    vh = h // (H // KV)
    s = Kf[vh] @ Q[h]                                    # Q 里已经折过 1/sqrt(256)
    p = np.exp(s - s.max())
    p /= p.sum()
    o = p @ Vf[vh]
    got = out.reshape(H, TP, D)[h, 0].astype(np.float64)
    errs.append(np.abs(got - o).mean() / max(np.abs(o).mean(), 1e-12))
errs = np.array(errs)
print('解码注意力输出相对误差：mean=%.3e  max=%.3e  (头 %d 最大)'
      % (errs.mean(), errs.max(), int(errs.argmax())))
