#!/usr/bin/env python3
# 逐阶段对照：把运行时 dump 出来的中间量（RT_DUMP_BUF=1）与 HF 参考实现对位比较。
# 线性层的输入按运行时一样做 int4 激活量化（默认 G=128），所以差异就是**实现错误**。
#
#   RT_DUMP_BUF=1 build/rt ... --dump build/db.bin --dump-layers -1,0,1,2,3
#   python3 tools/ref_stages.py build/db.bin <rt4> <json> <层号> [--quant 128] [--noutq]
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rt4_io import RT4, read_dump          # noqa: E402
from src_weights import Src                     # noqa: E402

import transformers.models.qwen3_5.modeling_qwen3_5 as M   # noqa: E402
from transformers import Qwen3_5TextConfig                   # noqa: E402

M.causal_conv1d_fn = None
M.causal_conv1d_update = None
M.chunk_gated_delta_rule = None
M.fused_recurrent_gated_delta_rule = None
M.FusedRMSNormGated = None

HID = 5120
PREFIX = 'model.language_model.layers.'


def q4(x, G):
    """运行时的 int4 激活量化：每 G 个一组、amax/7、round、夹到 -8..7。"""
    if G == 0:
        return x
    T, K = x.shape
    v = x.reshape(T, K // G, G)
    amax = v.abs().amax(-1, keepdim=True)
    s = torch.where(amax > 0, amax / 7.0, torch.ones_like(amax))
    return (torch.clamp(torch.round(v / s), -8, 7) * s).reshape(T, K)


def q8(x, G):
    """运行时的 8bit 激活量化：每 G 个一组、amax/119、round、夹到 -128..119
    （夹到 119 是为了能精确拆成 16*h + l，见 src/k_new.hip:quant_rows_a8_k）。"""
    if G == 0:
        return x
    T, K = x.shape
    v = x.reshape(T, K // G, G)
    amax = v.abs().amax(-1, keepdim=True)
    s = torch.where(amax > 0, amax / 119.0, torch.ones_like(amax))
    return (torch.clamp(torch.round(v / s), -128, 119) * s).reshape(T, K)


class Ref:
    def __init__(self, rt4p, jsp, il, G=128, quant_ab=False, bits=4, qgroup=0, src=None):
        self.R = RT4(rt4p, jsp)
        self.S = Src(src) if src else None
        cfg = json.load(open(os.path.join(os.path.dirname(rt4p), '..', 'config.json')))['text_config']
        self.cfg = Qwen3_5TextConfig(**cfg)
        self.il = il
        self.G = G
        self.bits = bits
        self.qgroup = qgroup
        self.quant_ab = quant_ab
        self.pfx = f'{PREFIX}{il}.'
        self.W = {}
        lay = M.Qwen3_5DecoderLayer(self.cfg, il)
        for name, p in lay.state_dict().items():
            key = self.pfx + name
            if self.S is not None:
                if self.S.has(key):
                    self.W[name] = torch.from_numpy(self.S.deq(key)).reshape(p.shape)
            elif key in self.R.d:
                self.W[name] = torch.from_numpy(self.R.dequant(key))
        # 把权重真正灌进 HF 模块（q_norm / k_norm 这类子模块的权重也必须对，
        # 否则会拿初始化值算，之前就在这里踩过坑）
        with torch.no_grad():
            for name, p in lay.state_dict().items():
                if name in self.W:
                    p.copy_(self.W[name].reshape(p.shape))
        self.layer = lay

    def w(self, n):
        return self.W[n]

    def q(self, x):
        if self.bits == 8:
            return q8(x, self.qgroup or self.G)
        return q4(x, self.G)

    def stages(self, x, pos):
        il, G = self.il, self.G
        out = {}
        cfg = self.cfg
        lay = self.layer
        T = x.shape[0]
        h = M.Qwen3_5RMSNorm(HID, cfg.rms_norm_eps)
        h.weight = torch.nn.Parameter(self.w('input_layernorm.weight'))
        xb = h(x[None])[0]
        hq = self.q(xb)
        full = (il + 1) % 4 == 0
        if not full:
            qkv = hq @ self.w('linear_attn.in_proj_qkv.weight').T
            out[1] = qkv
            z = hq @ self.w('linear_attn.in_proj_z.weight').T
            out[2] = z
            cw = self.w('linear_attn.conv1d.weight').reshape(-1, 1, 4)
            conv = F.conv1d(qkv.T.unsqueeze(0), cw, padding=3, groups=cw.shape[0])[0, :, :T].T
            conv = F.silu(conv)
            out[3] = conv
            q_, k_, v_ = conv[:, :2048], conv[:, 2048:4096], conv[:, 4096:]
            q_ = F.normalize(q_.reshape(T, 16, 128), dim=-1)   # l2norm 到单位长度
            k_ = F.normalize(k_.reshape(T, 16, 128), dim=-1)
            v_ = v_.reshape(T, 48, 128)
            out[4] = q_.reshape(T, 2048)
            out[5] = k_.reshape(T, 2048)
            out[6] = v_.reshape(T, 6144)
            src = hq if self.quant_ab else xb
            a = src @ self.w('linear_attn.in_proj_a.weight').T
            b = src @ self.w('linear_attn.in_proj_b.weight').T
            out[7] = torch.cat([a, b], dim=1)
            beta = b.sigmoid()
            out[8] = beta
            g = -self.w('linear_attn.A_log').float().exp() * F.softplus(
                a.float() + self.w('linear_attn.dt_bias'))
            out[9] = g
            rep = 3
            qr = q_.repeat_interleave(rep, dim=1)
            kr = k_.repeat_interleave(rep, dim=1)
            core, _ = M.torch_chunk_gated_delta_rule(
                qr[None], kr[None], v_[None], g=g[None], beta=beta[None],
                initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=False)
            core = core[0]
            out[10] = core.reshape(T, 6144)
            cc = core.reshape(-1, 128)
            zz = z.reshape(-1, 128)
            var = cc.pow(2).mean(-1, keepdim=True)
            n = cc * torch.rsqrt(var + cfg.rms_norm_eps)
            n = n * self.w('linear_attn.norm.weight') * F.silu(zz)
            n = n.reshape(T, 6144)
            out[11] = n
            y = self.q(n) @ self.w('linear_attn.out_proj.weight').T
        else:
            a = lay.self_attn
            qg = (hq @ self.w('self_attn.q_proj.weight').T).reshape(T, 24, 512)
            out[1] = qg.reshape(T, 12288)
            gate = qg[..., 256:]
            out[3] = gate.reshape(T, 6144)
            qs = a.q_norm(qg[..., :256])
            ks = a.k_norm((hq @ self.w('self_attn.k_proj.weight').T).reshape(T, 4, 256))
            vs = (hq @ self.w('self_attn.v_proj.weight').T).reshape(T, 4, 256)
            out[4] = (hq @ self.w('self_attn.k_proj.weight').T)
            out[5] = vs.reshape(T, 1024)
            cos, sin = M.Qwen3_5TextRotaryEmbedding(cfg)(x[None], pos)
            qs = qs.transpose(0, 1)[None]
            ks = ks.transpose(0, 1)[None]
            qs, ks = M.apply_rotary_pos_emb(qs, ks, cos, sin)
            out[2] = (qs[0].transpose(0, 1) * a.scaling).reshape(T, 6144)
            out[6] = ks[0].transpose(0, 1).reshape(T, 1024)
            mask = torch.triu(torch.full((1, 1, T, T), float('-inf')), diagonal=1)
            o, _ = M.eager_attention_forward(a, qs, ks, vs.transpose(0, 1)[None], mask,
                                             scaling=a.scaling)
            o = o.transpose(1, 2)[0]                                      # [24, T, 256]
            # 运行时那边 hfa 是 [head][补齐行][D]，这里保持同样顺序（head 在前）
            out[7] = o
            ao = o.transpose(0, 1).reshape(T, 6144) * torch.sigmoid(gate.reshape(T, 6144))
            out[8] = ao
            y = self.q(ao) @ self.w('self_attn.o_proj.weight').T
        out[9 if full else 12] = y
        x2 = x + y
        h2 = M.Qwen3_5RMSNorm(HID, cfg.rms_norm_eps)
        h2.weight = torch.nn.Parameter(self.w('post_attention_layernorm.weight'))
        hbq = self.q(h2(x2[None])[0])
        g_ = hbq @ self.w('mlp.gate_proj.weight').T
        out[21] = g_
        u_ = hbq @ self.w('mlp.up_proj.weight').T
        out[22] = u_
        act = F.silu(g_) * u_
        out[23] = act
        y2 = self.q(act) @ self.w('mlp.down_proj.weight').T
        out[24] = y2
        out['out'] = x2 + y2
        return out


NAME = {
    -1: 'embed',
    1: 'mixer/in_proj_qkv|q_proj', 2: 'z|q(norm+rope+scale)', 3: 'conv|gate',
    4: 'q(l2)|k(proj)', 5: 'k(l2)|v(proj)', 6: 'v|k(norm+rope)',
    7: 'a|b (attn out)', 8: 'beta|out*gate', 9: 'g|o_proj y',
    10: 'core', 11: 'normed', 12: 'out_proj y',
    21: 'mlp gate', 22: 'mlp up', 23: 'silu(gate)*up', 24: 'mlp down y',
}


def main():
    db, rt4p, jsp, il = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    G = int(sys.argv[sys.argv.index('--quant') + 1]) if '--quant' in sys.argv else 128
    bits = 8 if '--bits8' in sys.argv else 4
    qgroup = int(sys.argv[sys.argv.index('--qgroup') + 1]) if '--qgroup' in sys.argv else 0
    quant_ab = '--quant-ab' in sys.argv
    srcp = sys.argv[sys.argv.index('--src') + 1] if '--src' in sys.argv else None
    recs = read_dump(db)
    x = torch.from_numpy(recs[il - 1].reshape(-1, HID).copy())
    T = x.shape[0]
    pos = torch.arange(T)[None]
    ref = Ref(rt4p, jsp, il, G, quant_ab, bits, qgroup, srcp)
    st = ref.stages(x, pos)
    print(f'层 {il}  T={T}  quant={bits}bit G={qgroup or G}  quant_ab={quant_ab}')
    print(f'{"stage":<28} {"运行时":>12} {"参考":>12} {"rel":>9} {"max|Δ|":>10}')
    for k in sorted([k for k in st if isinstance(k, int)]):
        tag = 1000 + il * 100 + k
        if tag not in recs:
            print(f'  {k:>3} {NAME.get(k,"?"):<24} (dump 里没有 tag {tag})')
            continue
        got = torch.from_numpy(recs[tag].copy())
        exp = st[k].reshape(-1)
        if got.numel() != exp.numel():
            # 注意力输出在运行时里是按补齐行数（TP）存的
            if k == 7:
                TP = got.numel() // (24 * 256)
                got = got.reshape(24, TP, 256)[:, :T, :].reshape(-1)
            if got.numel() != exp.numel():
                print(f'  {k:>3} {NAME.get(k,"?"):<24} 形状不符 got={got.numel()} exp={exp.numel()}')
                continue
        rms = exp.pow(2).mean().sqrt()
        d = (got - exp)
        print(f'  {k:>3} {NAME.get(k,"?"):<24} {got.pow(2).mean().sqrt():>12.4f} '
              f'{rms:>12.4f} {d.pow(2).mean().sqrt()/max(rms.item(),1e-9):>9.2e} '
              f'{d.abs().max():>10.3f}')


if __name__ == '__main__':
    main()
