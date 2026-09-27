#!/usr/bin/env python3
# 用 torch 复现 HF 的「单层前向」，拿它跟自研运行时 dump 的层输出逐元素对比。
# 权重直接从 RT4 文件反量化（与运行时**同一份权重**），所以这里查的是实现错误，
# 不掺量化误差。
#
#   python3 tools/ref_layer.py <dump.bin> <rt4> <rt4.json> <层号>
#
# dump.bin 由 `build/rt --dump dump.bin --dump-layers -1,0,1,3` 生成：
#   每条记录 = [int32 layer][int32 n_tokens] 紧跟 n_tokens*hidden 个 float32
import sys, json, struct, numpy as np, torch

HID = 5120
DBG = True
import os
QUANT = os.environ.get('REF_QUANT', '0') == '1'
QUANT_G = int(os.environ.get('REF_G', '128'))

def stat(t):
    return f'mean|x|={t.abs().mean():.5f} max={t.abs().max():.4f}'

def read_dump(path):
    recs = {}
    with open(path, 'rb') as f:
        while True:
            h = f.read(8)
            if len(h) < 8: break
            il, cnt = struct.unpack('<ii', h)
            a = np.frombuffer(f.read(cnt * 4), dtype=np.float32)
            recs[il] = torch.from_numpy(a.copy())
    return recs

class RT4:
    def __init__(self, path, js):
        man = json.load(open(js))
        ts = man['tensors'] if isinstance(man, dict) and 'tensors' in man else man
        self.d = {t['name']: t for t in ts}
        self.f = open(path, 'rb')
        self.mm = np.memmap(path, dtype=np.uint8, mode='r')

    def dequant(self, name):
        """i4/f16/f32 → float32 torch 张量（形状按 manifest）"""
        t = self.d[name]
        shape = t['shape']
        if t['kind'] == 'f32':
            a = np.frombuffer(self.mm[t['q_off']:t['q_off'] + t['nbytes']],
                              dtype=np.float32)
            return torch.from_numpy(a.copy()).reshape(shape)
        if t['kind'] == 'f16':
            a = np.frombuffer(self.mm[t['q_off']:t['q_off'] + t['nbytes']],
                              dtype=np.float16).astype(np.float32)
            return torch.from_numpy(a.copy()).reshape(shape)
        N, K = shape[0], shape[1]
        G = t['group']
        q = np.frombuffer(self.mm[t['q_off']:t['q_off'] + N * K // 2], dtype=np.uint8).reshape(N, K // 2)
        s = np.frombuffer(self.mm[t['s_off']:t['s_off'] + N * (K // G) * 2],
                          dtype=np.float16).astype(np.float32).reshape(N, K // G)
        lo = (q & 0x0F).astype(np.int8); lo = np.where(lo >= 8, lo - 16, lo).astype(np.float32)
        hi = ((q >> 4) & 0x0F).astype(np.int8); hi = np.where(hi >= 8, hi - 16, hi).astype(np.float32)
        w = np.empty((N, K), np.float32)
        w[:, 0::2] = lo; w[:, 1::2] = hi
        w *= np.repeat(s, G, axis=1)
        return torch.from_numpy(w)

def q4(x, G=128):
    """模拟运行时的 int4 激活量化（每 128 个一组、amax/7 缩放、round 到 -8..7）"""
    T_, K = x.shape
    v = x.reshape(T_, K // G, G)
    amax = v.abs().amax(-1, keepdim=True)
    s = torch.where(amax > 0, amax / 7.0, torch.ones_like(amax))
    q = torch.clamp(torch.round(v / s), -8, 7)
    return (q * s).reshape(T_, K)

def q4s(x, G=32):
    T_, K = x.shape
    v = x.reshape(T_, K // G, G)
    amax = v.abs().amax(-1, keepdim=True)
    s = torch.where(amax > 0, amax / 7.0, torch.ones_like(amax))
    q = torch.clamp(torch.round(v / s), -8, 7)
    return (q * s).reshape(T_, K)

def rmsnorm(x, w, eps=1e-6, zero_centered=True):
    v = x.float().pow(2).mean(-1, keepdim=True)
    y = x * torch.rsqrt(v + eps)
    return y * ((1.0 + w.float()) if zero_centered else w.float())

def rope(x, pos, rot=64, theta=1e7):
    # x: [T, H, D]，只旋转前 rot 维，rotate_half 约定
    T, Hn, D = x.shape
    half = rot // 2
    inv = 1.0 / (theta ** (torch.arange(0, rot, 2, dtype=torch.float32) / rot))
    a = torch.outer(torch.tensor(pos, dtype=torch.float32), inv)      # [T, rot/2]
    c = a.cos()[:, None, :]; s = a.sin()[:, None, :]
    x1 = x[..., :half]; x2 = x[..., half:rot]
    o = torch.cat([x1 * c - x2 * s, x2 * c + x1 * s, x[..., rot:]], dim=-1)
    return o

def l2norm(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + eps)

def gdn_recurrent(q, k, v, g, beta):
    """HF torch_recurrent_gated_delta_rule（q/k 已 repeat 到 v 的头数）"""
    B, H, T, D = k.shape
    scale = 1.0 / (D ** 0.5)
    q = q * scale
    S = torch.zeros(B, H, D, v.shape[-1])
    out = torch.zeros(B, H, T, v.shape[-1])
    for t in range(T):
        gt = g[:, :, t].exp().unsqueeze(-1).unsqueeze(-1)
        bt = beta[:, :, t].unsqueeze(-1)
        S = S * gt
        kv = (S * k[:, :, t].unsqueeze(-1)).sum(-2)
        d = (v[:, :, t] - kv) * bt
        S = S + k[:, :, t].unsqueeze(-1) * d.unsqueeze(-2)
        out[:, :, t] = (S * q[:, :, t].unsqueeze(-1)).sum(-2)
    return out

def main():
    dump, rt4, js, il = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    recs = read_dump(dump)
    if il - 1 not in recs or il not in recs:
        print('dump 里缺少层', il - 1, '或', il); return
    x = recs[il - 1]                       # 本层输入
    ref_out = recs[il]                     # 运行时给出的本层输出
    T = x.shape[0]
    R = RT4(rt4, js)
    P = f'model.language_model.layers.{il}.'
    def W(n): return R.dequant(P + n)
    def F(n): return R.dequant(P + n)

    def rms(t, w): return rmsnorm(t, w)

    ltype = 'full_attention' if (il + 1) % 4 == 0 else 'linear_attention'
    qf_ = (q4 if QUANT_G == 128 else q4s) if QUANT else (lambda t, g=None: t)
    qf = q4 if QUANT_G == 128 else q4s
    h = torch.nn.functional
    if ltype == 'linear_attention':
        xb = rms(x, F('input_layernorm.weight'))
        xb = qf_(xb)
        qkv = xb @ W('linear_attn.in_proj_qkv.weight').T              # [T,10240]
        z = qf_(xb) @ W('linear_attn.in_proj_z.weight').T                  # [T,6144]
        # 因果卷积（核 4，padding 3）+ SiLU，取前 T 行
        cw = F('linear_attn.conv1d.weight').reshape(10240, 1, 4)      # [C,1,K]
        conv = h.conv1d(qkv.T.unsqueeze(0), cw, padding=3, groups=10240)[0, :, :T].T
        conv = h.silu(conv)
        q_, k_, v_ = conv[:, :2048], conv[:, 2048:4096], conv[:, 4096:]
        q_ = q_.reshape(T, 16, 128); k_ = k_.reshape(T, 16, 128); v_ = v_.reshape(T, 48, 128)
        a = qf_(xb) @ W('linear_attn.in_proj_a.weight').T
        b = qf_(xb) @ W('linear_attn.in_proj_b.weight').T
        beta = torch.sigmoid(b)
        g = -F('linear_attn.A_log').exp() * torch.nn.functional.softplus(
            a + F('linear_attn.dt_bias'))
        if DBG:
            print('    [ref] qkv', stat(qkv), 'z', stat(z), 'conv', stat(conv),
                  'a', stat(a), 'beta', stat(beta))
        q_ = l2norm(q_); k_ = l2norm(k_)
        rep = 48 // 16
        q_rep = q_.repeat_interleave(rep, dim=1); k_rep = k_.repeat_interleave(rep, dim=1)
        core = gdn_recurrent(q_rep.transpose(0, 1).unsqueeze(0),
                             k_rep.transpose(0, 1).unsqueeze(0),
                             v_.transpose(0, 1).unsqueeze(0),
                             g.transpose(0, 1).unsqueeze(0),
                             beta.transpose(0, 1).unsqueeze(0))[0].transpose(0, 1)
        if DBG: print('    [ref] g', stat(g), 'core', stat(core))
        core = core.reshape(T * 48, 128)
        zn = z.reshape(T * 48, 128)
        vv = core.pow(2).mean(-1, keepdim=True)
        n = core * torch.rsqrt(vv + 1e-6)
        n = n * F('linear_attn.norm.weight') * h.silu(zn)
        y = qf_(n.reshape(T, 6144)) @ W('linear_attn.out_proj.weight').T
    else:
        xb = rms(x, F('input_layernorm.weight'))
        xb = qf_(xb)
        qg = xb @ W('self_attn.q_proj.weight').T                      # [T,12288]
        qg = qg.reshape(T, 24, 512)
        q_, gate = qg[..., :256], qg[..., 256:]
        k_ = (xb @ W('self_attn.k_proj.weight').T).reshape(T, 4, 256)
        v_ = (xb @ W('self_attn.v_proj.weight').T).reshape(T, 4, 256)
        q_ = rmsnorm(q_, F('self_attn.q_norm.weight'))
        k_ = rmsnorm(k_, F('self_attn.k_norm.weight'))
        pos = list(range(T))
        q_ = rope(q_, pos); k_ = rope(k_, pos)
        rep = 24 // 4
        k_rep = k_.repeat_interleave(rep, dim=1).transpose(0, 1)
        v_rep = v_.repeat_interleave(rep, dim=1).transpose(0, 1)
        qt = q_.transpose(0, 1)
        o = torch.zeros_like(qt)
        scale = 1.0 / (256 ** 0.5)
        for i in range(T):
            s = torch.einsum('hd,hkd->hk', qt[:, i], k_rep[:, :i + 1]) * scale
            p = torch.softmax(s, dim=-1)
            o[:, i] = torch.einsum('hk,hkd->hd', p, v_rep[:, :i + 1])
        ao = o.transpose(0, 1).reshape(T, 6144) * torch.sigmoid(gate.reshape(T, 6144))
        y = qf_(ao) @ W('self_attn.o_proj.weight').T
    if DBG: print('    [ref] y(mixer)', stat(y), ' y2(mlp)', stat(y2 if False else torch.zeros(1)))
    x2 = x + y
    xb2 = rms(x2, F('post_attention_layernorm.weight'))
    xb2q = qf(xb2, QUANT_G) if QUANT else xb2
    g_ = xb2q @ W('mlp.gate_proj.weight').T
    u_ = xb2q @ W('mlp.up_proj.weight').T
    act = h.silu(g_) * u_
    actq = qf(act, QUANT_G) if QUANT else act
    y2 = actq @ W('mlp.down_proj.weight').T
    out = x2 + y2
    if DBG: print('    [ref] x2', stat(x2), ' y2', stat(y2), ' out', stat(out), ' x', stat(x))
    d = out - ref_out
    rms_ref = ref_out.pow(2).mean().sqrt()
    bulk = ref_out.abs() < 10
    print(f'层 {il} ({ltype}): 相对 RMS = {d.pow(2).mean().sqrt() / rms_ref:.3e}  '
          f'max|Δ| = {d.abs().max():.3e}  参考 rms = {rms_ref:.4f}')
    print(f'      小元素(|ref|<10, {bulk.sum().item()}/{ref_out.numel()}): 相对 = '
          f'{d[bulk].pow(2).mean().sqrt() / ref_out[bulk].pow(2).mean().sqrt():.3e}  '
          f'大元素: 相对 = {d[~bulk].pow(2).mean().sqrt() / ref_out[~bulk].pow(2).mean().sqrt():.3e}')
    # 是不是「同一批数值但顺序不同」？排序后比一遍
    y_ref = y
    y_rt = ref_out - x
    for tag, a, b in (('mixer y', y_ref, y_rt), ):
        d2 = a - b
        as_, bs_ = a.sort(dim=-1).values, b.sort(dim=-1).values
        print(f'      {tag}: 直接 rel={d2.pow(2).mean().sqrt()/a.pow(2).mean().sqrt():.3e}  '
              f'排序后 rel={(as_-bs_).pow(2).mean().sqrt()/as_.pow(2).mean().sqrt():.3e}')
    # 逐 token 的误差
    for t in range(min(T, 3)):
        dt_ = d[t]; rt = ref_out[t]
        print(f'      token {t}: rel={(dt_.pow(2).mean().sqrt()/rt.pow(2).mean().sqrt()):.3e} '
              f'最大元素 |ref|={rt.abs().max():.2f} |Δ|={dt_.abs().max():.2f}')

if __name__ == '__main__':
    main()
