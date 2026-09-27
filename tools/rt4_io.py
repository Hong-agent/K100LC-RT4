#!/usr/bin/env python3
# RT4 文件读取与反量化（运行时的单一事实来源的反向实现）。
# 供单层参考实现（tools/ref_hf.py）与抽查脚本共用。
import json
import numpy as np


def f16_to_f32(h):
    """把 uint16 的 f16 位模式转成 float32（不依赖 numpy 的 float16 转换）。"""
    h = np.asarray(h, dtype=np.uint16)
    sg = (h >> 15) & 1
    ex = (h >> 10) & 0x1F
    ma = h & 0x3FF
    out = np.where(ex == 0,
                   np.ldexp(ma.astype(np.float32) / 1024.0, -14),
                   np.ldexp(1.0 + ma.astype(np.float32) / 1024.0, ex.astype(np.int32) - 15))
    return np.where(sg == 1, -out, out).astype(np.float32)


class RT4:
    def __init__(self, path, js):
        man = json.load(open(js))
        ts = man['tensors'] if isinstance(man, dict) and 'tensors' in man else man
        self.d = {t['name']: t for t in ts}
        self.mm = np.memmap(path, dtype=np.uint8, mode='r')

    def names(self):
        return self.d.keys()

    def dequant(self, name, dtype=np.float32):
        """i4/f16/f32 → 数组（形状按 manifest）。"""
        t = self.d[name]
        shape = t['shape']
        if t['kind'] == 'f32':
            a = np.frombuffer(self.mm[t['q_off']:t['q_off'] + t['nbytes']], dtype=np.float32)
            return a.reshape(shape).astype(dtype, copy=True)
        if t['kind'] == 'f16':
            a = np.frombuffer(self.mm[t['q_off']:t['q_off'] + t['nbytes']], dtype=np.uint16)
            return f16_to_f32(a).reshape(shape).astype(dtype, copy=True)
        N, K = shape[0], shape[1]
        G = t['group']
        q = np.frombuffer(self.mm[t['q_off']:t['q_off'] + N * K // 2], dtype=np.uint8).reshape(N, K // 2)
        raw = np.frombuffer(self.mm[t['s_off']:t['s_off'] + N * (K // G) * 2], dtype=np.uint16)
        s = f16_to_f32(raw).reshape(N, K // G)
        w = np.empty((N, K), np.float32)
        # 0..15 → -8..7（用 int16 做中间量，避免 numpy 版本差异）
        lo = (q & 0x0F).astype(np.int16)
        hi = ((q >> 4) & 0x0F).astype(np.int16)
        w[:, 0::2] = np.where(lo >= 8, lo - 16, lo)
        w[:, 1::2] = np.where(hi >= 8, hi - 16, hi)
        w *= np.repeat(s, G, axis=1)
        return w.astype(dtype, copy=False)

    def dequant_range(self, name, n0, n1, dtype=np.float32):
        """只反量化第 [n0, n1) 行（词嵌入 / lm_head 太大，整块吃不下）。"""
        t = self.d[name]
        shape = t['shape']
        if t['kind'] == 'f32':
            a = np.frombuffer(self.mm[t['q_off'] + n0 * shape[1] * 4:
                                      t['q_off'] + n1 * shape[1] * 4], dtype=np.float32)
            return a.reshape(n1 - n0, shape[1]).astype(dtype, copy=True)
        if t['kind'] == 'f16':
            a = np.frombuffer(self.mm[t['q_off'] + n0 * shape[1] * 2:
                                      t['q_off'] + n1 * shape[1] * 2], dtype=np.uint16)
            return f16_to_f32(a).reshape(n1 - n0, shape[1]).astype(dtype, copy=True)
        N, K = shape[0], shape[1]
        G = t['group']
        nr = n1 - n0
        q = np.frombuffer(self.mm[t['q_off'] + n0 * K // 2: t['q_off'] + n1 * K // 2],
                          dtype=np.uint8).reshape(nr, K // 2)
        raw = np.frombuffer(self.mm[t['s_off'] + n0 * (K // G) * 2: t['s_off'] + n1 * (K // G) * 2],
                            dtype=np.uint16)
        s = f16_to_f32(raw).reshape(nr, K // G)
        w = np.empty((nr, K), np.float32)
        lo = (q & 0x0F).astype(np.int16)
        hi = ((q >> 4) & 0x0F).astype(np.int16)
        w[:, 0::2] = np.where(lo >= 8, lo - 16, lo)
        w[:, 1::2] = np.where(hi >= 8, hi - 16, hi)
        w *= np.repeat(s, G, axis=1)
        return w.astype(dtype, copy=False)


def read_dump(path):
    """读 runtime --dump 产生的记录：[int32 层号][int32 元素数] + float32 数据。"""
    recs = {}
    with open(path, 'rb') as f:
        while True:
            h = f.read(8)
            if len(h) < 8:
                break
            il, cnt = np.frombuffer(h, dtype=np.int32)
            recs[int(il)] = np.frombuffer(f.read(int(cnt) * 4), dtype=np.float32).copy()
    return recs
