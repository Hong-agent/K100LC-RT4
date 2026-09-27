#!/usr/bin/env python3
# 单层参考实现：**直接用 transformers 自带的 modeling_qwen3_5.py**（同一份 HF 语义），
# 权重从运行时用的同一个 RT4 文件反量化。用来把「实现错误」和「int4 量化误差」分开。
#
#   python3 tools/ref_hf.py <dump.bin> <rt4> <rt4.json> <层号> [--quant G] [--bf16]
#
# dump.bin 由 `build/rt --dump dump.bin --dump-layers -1,0,1,...` 生成。
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rt4_io import RT4, read_dump          # noqa: E402

import torch                               # noqa: E402
import transformers.models.qwen3_5.modeling_qwen3_5 as M   # noqa: E402
from transformers import Qwen3_5TextConfig                   # noqa: E402

torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '24')))

# 强制走 torch 参考实现（不碰 CUDA/DCU 专用 kernel）
M.causal_conv1d_fn = None
M.causal_conv1d_update = None
M.chunk_gated_delta_rule = None
M.fused_recurrent_gated_delta_rule = None
M.FusedRMSNormGated = None

PREFIX = 'model.language_model.'


def stat(t):
    return f'mean={t.abs().mean():.5f} max={t.abs().max():.4f} rms={t.pow(2).mean().sqrt():.5f}'


def quant_sim(x, G):
    """模拟运行时的 int4 激活量化（每 G 个一组、amax/7）。"""
    T, K = x.shape
    v = x.reshape(T, K // G, G)
    amax = v.abs().amax(-1, keepdim=True)
    s = torch.where(amax > 0, amax / 7.0, torch.ones_like(amax))
    return (torch.clamp(torch.round(v / s), -8, 7) * s).reshape(T, K)


def main():
    dump, rt4p, jsp, il = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    QG = 0
    if '--quant' in sys.argv:
        QG = int(sys.argv[sys.argv.index('--quant') + 1])
    recs = read_dump(dump)
    x = torch.from_numpy(recs[il - 1].reshape(-1, 5120).copy())
    ref_out = torch.from_numpy(recs[il].reshape(-1, 5120).copy())
    T = x.shape[0]

    cfg = Qwen3_5TextConfig(**__import__('json').load(open(
        os.path.join(os.path.dirname(rt4p), '..', 'config.json')))['text_config'])
    layer = M.Qwen3_5DecoderLayer(cfg, il).eval()

    R = RT4(rt4p, jsp)
    pfx = f'{PREFIX}layers.{il}.'
    missing = []
    with torch.no_grad():
        with torch.no_grad():
            for name, p in list(layer.state_dict().items()):
                key = pfx + name
                if key not in R.d:
                    missing.append(key)
                    continue
                w = R.dequant(key)
                p.copy_(torch.from_numpy(w.reshape(p.shape)))
        if missing:
            print('缺张量:', missing)
            return
        layer = layer.float()

        # 位置编码（注意力层才用）
        pos = torch.arange(T)[None]
        cos, sin = M.Qwen3_5TextRotaryEmbedding(cfg)(x, pos)
        mask = torch.full((1, 1, T, T), float('-inf'))
        mask = torch.triu(mask, diagonal=1)

        xin = x[None]
        if QG:
            # 量化模拟只作用在「线性层的输入」上：拆开手算一遍
            out = manual_forward(layer, cfg, xin, cos, sin, mask, QG)
        else:
            out = layer(xin, position_embeddings=(cos, sin), attention_mask=mask,
                        position_ids=pos)[0]
    out = out[0]

    d = (out - ref_out)
    rms = ref_out.pow(2).mean().sqrt()
    print(f'层 {il}  T={T}  {"full_attention" if (il + 1) % 4 == 0 else "linear_attention"}')
    print(f'  参考 x_in : {stat(x)}')
    print(f'  参考 out  : {stat(out)}')
    print(f'  运行时 out: {stat(ref_out)}')
    print(f'  相对 RMS  = {d.pow(2).mean().sqrt() / rms:.3e}   max|Δ| = {d.abs().max():.3e}')
    for t in range(min(T, 3)):
        dt = d[t]
        print(f'    token {t}: rel={dt.pow(2).mean().sqrt() / ref_out[t].pow(2).mean().sqrt():.3e} '
              f'max|Δ|={dt.abs().max():.3f}')


def manual_forward(layer, cfg, x, cos, sin, mask, QG):
    """把一层拆开手算，好把 int4 激活量化插在「线性层输入」上。"""
    import torch.nn.functional as F
    h = layer.input_layernorm(x)
    qh = quant_sim(h[0], QG)[None]
    if layer.layer_type == 'linear_attention':
        a = layer.linear_attn
        mixed = a.in_proj_qkv(qh).transpose(1, 2)
        z = a.in_proj_z(qh).reshape(1, x.shape[1], -1, a.head_v_dim)
        b = a.in_proj_b(qh)
        av = a.in_proj_a(qh)
        mixed = F.silu(a.conv1d(mixed)[:, :, :x.shape[1]]).transpose(1, 2)
        query, key, value = torch.split(mixed, [a.key_dim, a.key_dim, a.value_dim], dim=-1)
        query = query.reshape(1, x.shape[1], -1, a.head_k_dim)
        key = key.reshape(1, x.shape[1], -1, a.head_k_dim)
        value = value.reshape(1, x.shape[1], -1, a.head_v_dim)
        beta = b.sigmoid()
        g = -a.A_log.float().exp() * F.softplus(av.float() + a.dt_bias)
        rep = a.num_v_heads // a.num_k_heads
        if rep > 1:
            query = query.repeat_interleave(rep, dim=2)
            key = key.repeat_interleave(rep, dim=2)
        core, _ = M.torch_chunk_gated_delta_rule(
            query, key, value, g=g, beta=beta, initial_state=None,
            output_final_state=False, use_qk_l2norm_in_kernel=True)
        core = core.reshape(-1, a.head_v_dim)
        zz = z.reshape(-1, a.head_v_dim)
        core = a.norm(core, zz).reshape(1, x.shape[1], -1)
        y = a.out_proj(quant_sim(core[0], QG)[None])
    else:
        a = layer.self_attn
        qg = a.q_proj(qh).view(1, x.shape[1], -1, a.head_dim * 2)
        qs, gate = torch.chunk(qg, 2, dim=-1)
        gate = gate.reshape(1, x.shape[1], -1)
        ks = a.k_norm(a.k_proj(qh).view(1, x.shape[1], -1, a.head_dim))
        vs = a.v_proj(qh).view(1, x.shape[1], -1, a.head_dim)
        qs = a.q_norm(qs)
        qs = qs.transpose(1, 2)
        ks = ks.transpose(1, 2)
        vs = vs.transpose(1, 2)
        qs, ks = M.apply_rotary_pos_emb(qs, ks, cos, sin)
        o, _ = M.eager_attention_forward(a, qs, ks, vs, mask, scaling=a.scaling)
        o = o.transpose(1, 2).reshape(1, x.shape[1], -1)
        o = o * torch.sigmoid(gate)
        y = a.o_proj(quant_sim(o[0], QG)[None])
    x2 = x + y
    hb = layer.post_attention_layernorm(x2)
    hbq = quant_sim(hb[0], QG)[None]
    act = F.silu(layer.mlp.gate_proj(hbq)) * layer.mlp.up_proj(hbq)
    y2 = layer.mlp.down_proj(quant_sim(act[0], QG)[None])
    return x2 + y2


if __name__ == '__main__':
    main()
