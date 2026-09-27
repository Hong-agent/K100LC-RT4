#!/usr/bin/env python3
# 读原始 safetensors（FP8 / NVFP4 / BF16 混合精度）里的一张张量，反量化成 f32。
# 用来把「转换器的损失」和「实现的错误」分开。
import json
import struct

import numpy as np


def e4m3_lut():
    out = np.zeros(256, np.float32)
    for i in range(256):
        s, e, m = (i >> 7) & 1, (i >> 3) & 0xF, i & 7
        if e == 0:
            v = m / 8.0 * 2 ** (-6)
        elif e == 15 and m == 7:
            v = np.nan
        else:
            v = (1 + m / 8.0) * 2 ** (e - 7)
        out[i] = -v if s else v
    return out


E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], np.float32)
F8 = e4m3_lut()


class Src:
    def __init__(self, path):
        self.path = path
        self.f = open(path, 'rb')
        hlen = struct.unpack('<Q', self.f.read(8))[0]
        self.hdr = json.loads(self.f.read(hlen))
        self.base = 8 + hlen

    def has(self, name):
        return (name in self.hdr) or (name.replace('.weight', '.weight_packed') in self.hdr)

    def _rd(self, name, n0=None, n1=None):
        e = self.hdr[name]
        off = self.base + e['data_offsets'][0]
        shape = e['shape']
        if n0 is None:
            n0, n1 = 0, shape[0]
        row = int(np.prod(shape[1:])) if len(shape) > 1 else 1
        itemsz = {'BF16': 2, 'F32': 4, 'F8_E4M3': 1, 'U8': 1}[e['dtype']]
        self.f.seek(off + n0 * row * itemsz)
        raw = self.f.read((n1 - n0) * row * itemsz)
        return e, raw, n0, n1

    def deq(self, name, n0=None, n1=None):
        """返回 [n1-n0, K] 的 f32（K=prod(shape[1:])）。"""
        packed = name.replace('.weight', '.weight_packed')
        e, raw, n0, n1 = self._rd(packed if packed in self.hdr else name, n0, n1)
        nr = n1 - n0
        if e['dtype'] == 'BF16':
            v = (np.frombuffer(raw, '<u2').astype(np.uint32) << 16).view(np.float32)
            return v.reshape(nr, -1).copy()
        if e['dtype'] == 'F32':
            return np.frombuffer(raw, '<f4').reshape(nr, -1).copy()
        if e['dtype'] == 'F8_E4M3':
            K = int(np.prod(e['shape'][1:]))
            w = F8[np.frombuffer(raw, np.uint8)].reshape(nr, K)
            es, sraw, _, _ = self._rd(name + '_scale', n0, n1)
            s = (np.frombuffer(sraw, '<u2').astype(np.uint32) << 16).view(np.float32)
            return (w * s.reshape(-1, 1)).astype(np.float32)
        if e['dtype'] in ('U8',):                       # NVFP4 打包
            K = int(np.prod(e['shape'][1:])) * 2        # 每字节 2 个
            b = np.frombuffer(raw, np.uint8).reshape(nr, K // 2)
            lo = E2M1[(b & 0xF) & 7] * np.where((b & 0xF) & 8, -1, 1)
            hi = E2M1[(b >> 4) & 7] * np.where((b >> 4) & 8, -1, 1)
            w = np.empty((nr, K), np.float32)
            w[:, 0::2] = lo
            w[:, 1::2] = hi
            bs, braw, _, _ = self._rd(name + '_scale', n0, n1)
            s = F8[np.frombuffer(braw, np.uint8)].reshape(nr, K // 16)
            gs, graw, _, _ = self._rd(name + '_global_scale', 0, 1)
            g = np.frombuffer(graw, '<f4')[0]
            return (w * np.repeat(s, 16, axis=1) / g).astype(np.float32)
        raise ValueError('不认识的 dtype ' + e['dtype'])
