#!/usr/bin/env python3
"""把 Qwen3.5 视觉塔转换成独立 RT4 文件。

设计：视觉线性权重统一成 RT4 的 f16（K 补零到 32、N 补零到 64），
norm/bias/pos_embed 保留 f32。视觉塔对 int4 太敏感（27 层累积后余弦会崩），
所以这里不量化到 int4；但它仍然是 RT4 文件，由同一个运行时加载。

  python3 tools/convert_vision_rt4.py <model.safetensors> <out.rt4>
"""
import argparse
import json
import math
import os
import struct

import numpy as np


GRP = 128


def read_header(path):
    with open(path, 'rb') as f:
        hlen = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(hlen))
        base = 8 + hlen
    return header, base


def read_tensor(f, base, ent):
    f.seek(base + ent['data_offsets'][0])
    raw = f.read(ent['data_offsets'][1] - ent['data_offsets'][0])
    dt = {'BF16': np.uint16, 'F16': np.float16, 'F32': np.float32}[ent['dtype']]
    arr = np.frombuffer(raw, dtype=dt).copy().reshape(ent['shape'])
    if ent['dtype'] == 'BF16':
        u = arr.astype(np.uint32) << 16
        arr = u.view(np.float32)
    return arr.astype(np.float32, copy=False)


def pad_weight(w):
    N, K = w.shape
    Np = (N + 63) // 64 * 64
    Kp = (K + 31) // 32 * 32
    if Np == N and Kp == K:
        return w
    out = np.zeros((Np, Kp), dtype=np.float32)
    out[:N, :K] = w
    return out


def write_bytes(f, data, align=256):
    off = f.tell()
    f.write(data)
    pad = (align - (f.tell() & (align - 1))) & (align - 1)
    if pad:
        f.write(b'\0' * pad)
    return off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('source')
    ap.add_argument('out')
    args = ap.parse_args()
    header, base = read_header(args.source)
    prefix = 'model.visual.'
    keys = [k for k in header if k.startswith(prefix)]
    if not keys:
        raise SystemExit('源权重里没有 model.visual.*')

    manifest = []
    with open(args.source, 'rb') as src, open(args.out, 'wb') as out:
        for i, name in enumerate(keys, 1):
            ent = header[name]
            arr = read_tensor(src, base, ent)
            kind = 'f32'
            relerr = 0.0
            if (name.endswith('.weight') and arr.ndim >= 2 and
                    name != 'model.visual.pos_embed.weight'):
                # 5D patch_embed conv [out,in,t,ph,pw] 展平成 [out,in*t*ph*pw]
                w = arr.reshape(arr.shape[0], -1)
                logical = w.copy()
                w = pad_weight(w)
                q = w.astype(np.float16)
                q_off = write_bytes(out, q.tobytes(order='C'))
                shape = [int(w.shape[0]), int(w.shape[1])]
                nbytes = int(q.size * 2)
                # f16 舍入误差
                d = logical.astype(np.float64) - q[:logical.shape[0], :logical.shape[1]].astype(np.float64)
                relerr = math.sqrt(float(np.sum(d * d)) / float(np.sum(logical.astype(np.float64) ** 2)))
                manifest.append({
                    'name': name, 'kind': 'f16', 'shape': shape, 'group': 0,
                    'q_off': q_off, 's_off': 0, 'nbytes': nbytes,
                    'relerr': round(relerr, 5),
                    'src_rms': float(np.sqrt(np.mean(logical.astype(np.float64) ** 2))),
                })
            else:
                data = arr.astype(np.float32, copy=False).tobytes(order='C')
                q_off = write_bytes(out, data)
                manifest.append({
                    'name': name, 'kind': 'f32', 'shape': [int(x) for x in arr.shape],
                    'group': 0, 'q_off': q_off, 's_off': 0, 'nbytes': len(data),
                    'relerr': 0.0,
                    'src_rms': float(np.sqrt(np.mean(arr.astype(np.float64) ** 2))),
                })
            if i % 50 == 0 or i == len(keys):
                print(f'  视觉张量 {i}/{len(keys)}', flush=True)

    mpath = args.out + '.json'
    with open(mpath, 'w', encoding='utf-8') as f:
        json.dump({'format': 'RT4-v1', 'group': GRP, 'weight_file': args.out,
                   'tensors': manifest}, f, ensure_ascii=False, indent=1)
    print(f'写出 {args.out}: {len(manifest)} 张量, {os.path.getsize(args.out) / 1e9:.2f} GB')


if __name__ == '__main__':
    main()
