#!/usr/bin/env python3
# 全模型 torch 参考（逐层流式加载，宿主内存只放一层的权重）。
# 用 HF 自带的 Qwen3_5 模块 + 同一份 RT4 权重，**不含任何激活量化**（W4A16），
# 用来回答两个问题：
#   1. 这份 int4/128 的权重本身还能不能出正确的话？（对 llama.cpp Q4_K_M 做交叉验证）
#   2. 运行时在哪一层开始偏离（--dump 出每层 x，与 build/rt 的 dump 对比）。
#
#   python3 tools/ref_full.py --ids 248044,9707,... [--quant 8] [--dump build/reffull.bin] [--max-layers N]
import argparse
import gc
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rt4_io import RT4, f16_to_f32            # noqa: E402
from src_weights import Src                    # noqa: E402

import transformers.models.qwen3_5.modeling_qwen3_5 as M   # noqa: E402
from transformers import Qwen3_5TextConfig                   # noqa: E402

M.causal_conv1d_fn = None
M.causal_conv1d_update = None
M.chunk_gated_delta_rule = None
M.fused_recurrent_gated_delta_rule = None
M.FusedRMSNormGated = None

torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '24')))
PREFIX = 'model.language_model.'


def qact(x, G, bits):
    if not bits:
        return x
    T, K = x.shape
    v = x.reshape(T, K // G, G)
    amax = v.abs().amax(-1, keepdim=True)
    hi = (1 << (bits - 1)) - 1
    s = torch.where(amax > 0, amax / float(hi), torch.ones_like(amax))
    return (torch.clamp(torch.round(v / s), -hi - 1, hi) * s).reshape(T, K)


class QLinear(torch.nn.Module):
    """包装 nn.Linear：先按运行时的做法量化输入激活，再算线性层。"""

    def __init__(self, lin, G, bits):
        super().__init__()
        self.weight = lin.weight
        self.G = G
        self.bits = bits

    def forward(self, x):
        return F.linear(qact(x, self.G, self.bits), self.weight, None)


def wrap_linear(mod, G, bits):
    for name, child in list(mod.named_children()):
        if isinstance(child, torch.nn.Linear):
            setattr(mod, name, QLinear(child, G, bits))
        else:
            wrap_linear(child, G, bits)


class Full:
    def __init__(self, rt4p, jsp, bits=0, G=128, src=None):
        cfg = json.load(open(os.path.join(os.path.dirname(rt4p), '..', 'config.json')))['text_config']
        self.cfg = Qwen3_5TextConfig(**cfg)
        self.R = RT4(rt4p, jsp)
        self.S = Src(src) if src else None
        self.bits = bits
        self.G = G
        self.cache = {}

    def deq(self, name, shape, dtype=torch.float32):
        if self.S is not None:
            return torch.from_numpy(self.S.deq(name)).reshape(shape)
        return torch.from_numpy(self.R.dequant(name)).reshape(shape)

    def load_layer(self, il):
        lay = M.Qwen3_5DecoderLayer(self.cfg, il)
        pfx = f'{PREFIX}layers.{il}.'
        with torch.no_grad():
            for name, p in lay.state_dict().items():
                key = pfx + name
                if key in self.R.d:
                    p.copy_(self.deq(key, p.shape))
        if self.bits:
            wrap_linear(lay, self.G, self.bits)
        return lay

    def embed(self, ids):
        # 词嵌入表 5.1GB(f32)，只按需要的行反量化
        name = f'{PREFIX}embed_tokens.weight'
        if self.S is not None:
            rows = self.S.deq(name, min(ids), max(ids) + 1)
        else:
            rows = self.R.dequant_range(name, min(ids), max(ids) + 1)
        base = min(ids)
        return torch.from_numpy(rows[[i - base for i in ids]].copy())

    @torch.no_grad()
    def final(self, x):
        n = M.Qwen3_5RMSNorm(self.cfg.hidden_size, self.cfg.rms_norm_eps)
        n.weight = torch.nn.Parameter(self.deq(f'{PREFIX}norm.weight', (self.cfg.hidden_size,)))
        h = n(x[None])[0]
        hq = qact(h, self.G, self.bits) if self.bits else h
        last = hq[-1].detach().numpy()
        lg = np.empty(self.cfg.vocab_size, np.float32)
        for c0 in range(0, self.cfg.vocab_size, 8192):            # lm_head 分块，别整块进内存
            c1 = min(c0 + 8192, self.cfg.vocab_size)
            w = self.S.deq('lm_head.weight', c0, c1) if self.S else self.R.dequant_range('lm_head.weight', c0, c1)
            lg[c0:c1] = w @ last
            del w
        lg = torch.from_numpy(lg)
        return lg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rt4', default='models/Qwen3.8-27B-NVFP4/rt4/qwen38_27b.rt4')
    ap.add_argument('--json', default='models/Qwen3.8-27B-NVFP4/rt4/qwen38_27b.rt4.json')
    ap.add_argument('--ids', required=True)
    ap.add_argument('--quant', type=int, default=0, help='激活量化位数（0=不量化）')
    ap.add_argument('--group', type=int, default=128)
    ap.add_argument('--max-layers', type=int, default=64)
    ap.add_argument('--dump', default=None)
    ap.add_argument('--dump-every', type=int, default=1)
    ap.add_argument('--src', default=None, help='用原始 safetensors 的权重（不经 RT4）')
    ap.add_argument('--topk', type=int, default=5)
    args = ap.parse_args()
    ids = [int(x) for x in args.ids.split(',')]
    T = len(ids)

    fu = Full(args.rt4, args.json, args.quant, args.group, args.src)
    x = fu.embed(ids)
    print(f'embed: rms={x.pow(2).mean().sqrt():.5f} max={x.abs().max():.4f}', flush=True)
    if args.dump:
        open(args.dump, 'wb').close()

        def wr(tag, t):
            with open(args.dump, 'ab') as f:
                a = t.reshape(-1).detach().numpy().astype(np.float32)
                f.write(np.array([tag, a.size], np.int32).tobytes())
                f.write(a.tobytes())
        wr(-1, x)
    pos = torch.arange(T)[None]
    cos, sin = M.Qwen3_5TextRotaryEmbedding(fu.cfg)(x[None], pos)
    mask = torch.triu(torch.full((1, 1, T, T), float('-inf')), diagonal=1)
    for il in range(args.max_layers):
        lay = fu.load_layer(il)
        with torch.no_grad():
            x = lay(x[None], position_embeddings=(cos, sin), attention_mask=mask,
                    position_ids=pos)[0]
        print(f'  层 {il:2d}: rms={x.pow(2).mean().sqrt():.5f} max={x.abs().max():.4f}', flush=True)
        if args.dump and (il + 1) % args.dump_every == 0:
            wr(il, x)
        del lay
        gc.collect()
    lg = fu.final(x)
    v, i = lg.topk(args.topk)
    print('top-%d: %s' % (args.topk, ' '.join(f'{a.item()}:{b.item():.3f}' for b, a in zip(v, i))))


if __name__ == '__main__':
    main()
