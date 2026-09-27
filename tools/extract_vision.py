#!/usr/bin/env python3
"""（可选/诊断）把 Qwen3.5 视觉塔导出成 transformers 可直接加载的目录。

运行时已统一走 tools/convert_vision_rt4.py；本工具只用于和 transformers
参考实现逐层对照，或临时做 HF 视觉桥实验。

在 DTK 容器里运行（需要 torch/safetensors/transformers）：
  python3 tools/extract_vision.py <model.safetensors> <config.json> <out_dir>
"""
import argparse
import json
import os
import struct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('source')
    ap.add_argument('config')
    ap.add_argument('out_dir')
    args = ap.parse_args()

    import numpy as np
    import torch
    from safetensors.torch import save_file

    os.makedirs(args.out_dir, exist_ok=True)
    prefix = 'model.visual.'
    tensors = {}
    dtype_map = {'BF16': np.uint16, 'F16': np.float16, 'F32': np.float32}
    with open(args.source, 'rb') as f:
        hlen = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(hlen))
        base = 8 + hlen
        keys = [k for k in header if k.startswith(prefix)]
        if not keys:
            raise SystemExit('源权重里没有 model.visual.*，确认源快照是否包含视觉塔')
        for i, key in enumerate(keys, 1):
            ent = header[key]
            f.seek(base + ent['data_offsets'][0])
            raw = f.read(ent['data_offsets'][1] - ent['data_offsets'][0])
            arr = np.frombuffer(raw, dtype=dtype_map[ent['dtype']]).copy().reshape(ent['shape'])
            t = torch.from_numpy(arr)
            if ent['dtype'] == 'BF16':
                t = t.view(torch.bfloat16)
            tensors[key[len(prefix):]] = t.contiguous()
            if i % 50 == 0 or i == len(keys):
                print(f'  读取视觉张量 {i}/{len(keys)}', flush=True)

    save_file(tensors, os.path.join(args.out_dir, 'model.safetensors'))
    cfg = json.load(open(args.config, encoding='utf-8'))['vision_config']
    cfg['architectures'] = ['Qwen3_5VisionModel']
    with open(os.path.join(args.out_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    total = sum(t.numel() * t.element_size() for t in tensors.values())
    print(f'视觉塔导出完成：{len(tensors)} 张量，{total / 1e9:.2f} GB -> {args.out_dir}')


if __name__ == '__main__':
    main()
