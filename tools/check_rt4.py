#!/usr/bin/env python3
"""读 RT4 文件 + manifest，反量化并与期望值（或用例自带参考）比较。

用法: check_rt4.py <out.rt4> [expect.json]
"""
import json
import struct
import sys


def f16_to_f32(h: int) -> float:
    return struct.unpack("<e", struct.pack("<H", h))[0]


def load(manifest_path, q_path, name):
    man = json.load(open(manifest_path))
    ent = None
    for t in man["tensors"]:
        if t["name"] == name:
            ent = t
            break
    if ent is None:
        raise KeyError(name)
    # 非量化张量是 1 维（norm 之类），当成长度 kb 的一行
    nb = ent["shape"][0]
    kb = ent["shape"][1] if len(ent["shape"]) > 1 else 1
    if ent["kind"] in ("f16", "f32"):
        with open(q_path, "rb") as fh:
            fh.seek(ent["q_off"])
            raw = fh.read(ent["nbytes"])
        if ent["kind"] == "f32":
            vals = list(struct.unpack(f"<{len(raw) // 4}f", raw))
        else:
            vals = [f16_to_f32(h) for h in struct.unpack(f"<{len(raw) // 2}H", raw)]
        return ent, [vals], []
    bits = 4 if ent["kind"] == "i4" else 8
    grp = ent["group"]
    with open(q_path, "rb") as fh:
        fh.seek(ent["q_off"])
        q = fh.read(nb * kb // (2 if bits == 4 else 1))
        fh.seek(ent["s_off"])
        s = fh.read(2 * nb * (kb // grp))
    scales = struct.unpack(f"<{nb * (kb // grp)}e", s)
    out = []
    for r in range(nb):
        row = []
        for i in range(kb):
            if bits == 4:
                b = q[r * (kb // 2) + (i >> 1)]
                code = (b >> 4) if (i & 1) else (b & 0xF)
                code = code - 16 if code >= 8 else code
            else:
                code = q[r * kb + i]
                code = code - 256 if code >= 128 else code
            row.append(code * scales[r * (kb // grp) + i // grp])
        out.append(row)
    return ent, out, scales


def main():
    rt4_path = sys.argv[1]
    manifest_path = rt4_path + ".json"
    expect_path = sys.argv[2] if len(sys.argv) > 2 else None
    expect = json.load(open(expect_path)) if expect_path else None

    checks = [
        ("model.language_model.layers.0.mlp.gate_proj.weight", "rt4_nvfp4"),
        ("lm_head.weight", "rt4_fp8"),
        # 回归：BF16 源 → f32 目标必须**真的转换**（曾经按 4 字节原样搬，
        # 导致 norm / A_log / dt_bias / conv1d 全错位、模型变乱码）
        ("model.language_model.norm.weight", "norm_f32"),
    ]
    bad = 0
    for name, key in checks:
        if expect is None:
            ent, vals, _ = load(manifest_path, rt4_path, name)
            print(name, ent["kind"], "first row[0:6]=", [round(v, 6) for v in vals[0][:6]])
            continue
        ent, vals, sc = load(manifest_path, rt4_path, name)
        ref = expect[key]
        if isinstance(ref[0], (int, float)):      # 1 维期望值（norm 权重）
            ref = [ref]
        grp = ent["group"] or 1
        kb = len(ref[0])
        worst = 0.0
        ties = 0
        real = []
        for r in range(len(ref)):
            for i in range(kb):
                d = abs(vals[r][i] - ref[r][i])
                worst = max(worst, d)
                # 该元素所在的量化组尺度 = 一个量化格；两组都合法的舍入只会在
                # 正好 .5 的平局点上差 1 格（C 用 roundf，Python 用 banker's）
                step = sc[r * (kb // grp) + i // grp] if sc else 0.0
                if ent["kind"] in ("f16", "f32"):     # 非量化张量只允许浮点误差
                    step = 1e-6
                if d > 1e-9:
                    if d <= 1.01 * step:
                        ties += 1
                    else:
                        real.append((r, i, d, step))
        bad += len(real)
        print(f"{name:58s} kind={ent['kind']} 声明 relerr={ent['relerr']:.4f} "
              f"max|Δ|={worst:.3e} 平局差={ties} 真错={len(real)}")
        if real:
            print("   真错样例:", real[:3])
    print("结论:", "通过（差异只有整数格平局舍入）" if bad == 0 else f"不通过，{bad} 处真错")
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
