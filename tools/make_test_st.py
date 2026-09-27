#!/usr/bin/env python3
"""造一个极小的 safetensors（NVFP4 + FP8 两种张量）用于验证 convert.c 的正确性。

输出: /tmp/rt_test/in.safetensors  +  /tmp/rt_test/expect.json（Python 侧独立算出的期望值）
"""
import json
import os
import random
import struct

OUT = "/tmp/rt_test"
N, K = 4, 256


def f32_to_bf16(x: float) -> int:
    u = struct.unpack("<I", struct.pack("<f", x))[0]
    return (u + 0x8000) >> 16 & 0xFFFF


def f32_to_f16(x: float) -> int:
    return struct.unpack("<H", struct.pack("<e", x))[0]


def f16_to_f32(h: int) -> float:
    return struct.unpack("<e", struct.pack("<H", h))[0]


def e4m3_to_f32(i: int) -> float:
    s, e, m = (i >> 7) & 1, (i >> 3) & 0xF, i & 7
    if e == 0:
        v = m / 8.0 * 2 ** (-6)
    elif e == 15 and m == 7:
        v = float("nan")
    else:
        v = (1 + m / 8.0) * 2 ** (e - 7)
    return -v if s else v


def f32_to_e4m3(x: float) -> int:
    best, bd = 0, 1e30
    for i in range(256):
        d = abs(e4m3_to_f32(i) - x)
        if d < bd:
            best, bd = i, d
    return best


E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def e2m1_to_f32(code: int) -> float:
    v = E2M1[code & 7]
    return -v if code & 8 else v


def e2m1_code(v: float) -> int:
    s = 8 if v < 0 else 0
    v = abs(v)
    best, bd = 0, 1e30
    for i, m in enumerate(E2M1):
        if abs(m - v) < bd:
            best, bd = i, abs(m - v)
    return s | best


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(1234)

    gscale = 0.0137
    codes = [[e2m1_code(rng.choice([-1, -0.5, 0, 0.5, 1, 2, 3]) * rng.uniform(0.2, 4))
              for _ in range(K)] for _ in range(N)]
    bs = [[f32_to_e4m3(rng.uniform(0.5, 2.0)) for _ in range(K // 16)] for _ in range(N)]
    packed = bytearray()
    for r in range(N):
        for i in range(0, K, 2):
            packed.append((codes[r][i] & 0xF) | ((codes[r][i + 1] & 0xF) << 4))

    # 注意：NVFP4 存的块尺度是「已乘过 global_scale」的，反量化要除回来
    ref_nv = [[e2m1_to_f32(codes[r][i]) * e4m3_to_f32(bs[r][i // 16]) / gscale
               for i in range(K)] for r in range(N)]

    fp8_codes = [[f32_to_e4m3(rng.choice([-2.0, -1.0, -0.5, 0.25, 0.5, 1.0, 2.0]) * rng.uniform(0.5, 3))
                  for _ in range(K)] for _ in range(N)]
    ch_scale = [rng.uniform(0.01, 0.05) for _ in range(N)]
    ref_fp8 = [[e4m3_to_f32(fp8_codes[r][i]) * ch_scale[r] for i in range(K)] for r in range(N)]
    norm = [rng.uniform(0.5, 1.5) for _ in range(K)]

    blobs = {}
    blobs["model.language_model.layers.0.mlp.gate_proj.weight_packed"] = ("U8", [N, K // 2], bytes(packed))
    blobs["model.language_model.layers.0.mlp.gate_proj.weight_scale"] = (
        "F8_E4M3", [N, K // 16], bytes(b for r in bs for b in r))
    blobs["model.language_model.layers.0.mlp.gate_proj.weight_global_scale"] = ("F32", [1], struct.pack("<f", gscale))
    blobs["lm_head.weight"] = ("F8_E4M3", [N, K], bytes(b for r in fp8_codes for b in r))
    blobs["lm_head.weight_scale"] = ("BF16", [N, 1],
                                     b"".join(struct.pack("<H", f32_to_bf16(s)) for s in ch_scale))
    blobs["model.language_model.norm.weight"] = ("BF16", [K],
                                                 b"".join(struct.pack("<H", f32_to_bf16(v)) for v in norm))

    header, data, off = {}, bytearray(), 0
    for name, (dtype, shape, payload) in blobs.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + len(payload)]}
        data += payload
        off += len(payload)
    hj = json.dumps(header, separators=(",", ":")).encode()
    hj += b" " * ((8 - len(hj) % 8) % 8)
    with open(f"{OUT}/in.safetensors", "wb") as fh:
        fh.write(struct.pack("<Q", len(hj)))
        fh.write(hj)
        fh.write(data)

    def rt4(row, grp=128):
        out = []
        for g in range(0, len(row), grp):
            blk = row[g:g + grp]
            amax = max(abs(v) for v in blk)
            s = amax / 7.0 if amax > 0 else 1.0
            s16 = f16_to_f32(f32_to_f16(s))
            for v in blk:
                q = max(-7, min(7, round(v / s)))
                out.append(q * s16)
        return out

    with open(f"{OUT}/expect.json", "w") as fh:
        json.dump({
            "nvfp4_dequant": ref_nv,
            "fp8_dequant": ref_fp8,
            "rt4_nvfp4": [rt4(r) for r in ref_nv],
            "rt4_fp8": [rt4(r) for r in ref_fp8],
            # 期望值要按「源是 BF16」算：RT4 里存 f32 时应当是 bf16→f32 的结果
            "norm_f32": [struct.unpack("<f", struct.pack("<I", f32_to_bf16(v) << 16))[0] for v in norm],
            "ch_scale": ch_scale,
        }, fh)
    print("wrote", f"{OUT}/in.safetensors")


if __name__ == "__main__":
    main()
