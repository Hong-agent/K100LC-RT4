#!/usr/bin/env python3
"""真实张量抽查：RT4 文件反量化 vs 源 safetensors 反量化，逐元素比对。

验证 manifest 的偏移、行序、分组对齐都没错（合成用例覆盖不到的部分）。
用法: spot_check_rt4.py <model.safetensors> <out.rt4> <hf_tensor_name> [rows]
"""
import json
import struct
import sys

import numpy as np

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def e4m3_lut():
    out = np.zeros(256, dtype=np.float32)
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


def load_header(fh):
    hlen = struct.unpack("<Q", fh.read(8))[0]
    return json.loads(fh.read(hlen)), 8 + hlen


def main():
    st_path, rt4_path, name = sys.argv[1], sys.argv[2], sys.argv[3]
    rows = int(sys.argv[4]) if len(sys.argv) > 4 else 256
    F8 = e4m3_lut()

    rt4 = json.load(open(rt4_path + ".json"))
    ent = next(t for t in rt4["tensors"] if t["name"] == name)
    N, K = ent["shape"]
    grp = ent["group"]
    rows = min(rows, N)

    with open(rt4_path, "rb") as fh:
        fh.seek(ent["q_off"])
        if ent["kind"] == "i4":
            q = np.frombuffer(fh.read(rows * K // 2), dtype=np.uint8).reshape(rows, K // 2)
            lo = (q & 0xF).astype(np.int16)
            hi = ((q >> 4) & 0xF).astype(np.int16)
            codes = np.empty((rows, K), dtype=np.int16)
            codes[:, 0::2] = lo
            codes[:, 1::2] = hi
            codes = np.where(codes >= 8, codes - 16, codes).astype(np.float32)
        else:
            q = np.frombuffer(fh.read(rows * K), dtype=np.int8).reshape(rows, K)
            codes = q.astype(np.float32)
        fh.seek(ent["s_off"])
        s = np.frombuffer(fh.read(2 * rows * (K // grp)), dtype="<f2").astype(np.float32)
        s = s.reshape(rows, K // grp)
    rt = codes * np.repeat(s, grp, axis=1)

    with open(st_path, "rb") as fh:
        hdr, base = load_header(fh)
        packed = name.replace(".weight", ".weight_packed") in hdr
        src_name = name.replace(".weight", ".weight_packed") if packed else name
        e = hdr[src_name]
        fh.seek(base + e["data_offsets"][0])
        if packed:
            raw = np.frombuffer(fh.read(rows * K // 2), dtype=np.uint8).reshape(rows, K // 2)
            lo = E2M1[(raw & 0xF) & 7] * np.where((raw & 0xF) & 8, -1, 1)
            hi = E2M1[(raw >> 4) & 7] * np.where((raw >> 4) & 8, -1, 1)
            vals = np.empty((rows, K), dtype=np.float32)
            vals[:, 0::2] = lo
            vals[:, 1::2] = hi
            es = hdr[name.replace(".weight", ".weight_scale")]
            fh.seek(base + es["data_offsets"][0])
            bs = F8[np.frombuffer(fh.read(rows * (K // 16)), dtype=np.uint8)].reshape(rows, K // 16)
            eg = hdr[name.replace(".weight", ".weight_global_scale")]
            fh.seek(base + eg["data_offsets"][0])
            g = struct.unpack("<f", fh.read(4))[0]
            src = vals * np.repeat(bs, 16, axis=1) / g
        elif e["dtype"] == "F8_E4M3":
            codes8 = np.frombuffer(fh.read(rows * K), dtype=np.uint8).reshape(rows, K)
            es = hdr[name + "_scale"]
            fh.seek(base + es["data_offsets"][0])
            cs = (np.frombuffer(fh.read(rows * 2), dtype="<u2").astype(np.uint32) << 16).view(np.float32)
            src = F8[codes8] * cs.reshape(rows, 1)
        else:
            raw = np.frombuffer(fh.read(rows * K * 2), dtype="<u2").astype(np.uint32) << 16
            src = raw.view(np.float32).reshape(rows, K)

    a, b = rt.astype(np.float64), src.astype(np.float64)
    rel = np.linalg.norm(a - b) / np.linalg.norm(b)
    cos = float(np.dot(a.ravel(), b.ravel()) / (np.linalg.norm(a) * np.linalg.norm(b)))
    row_err = np.abs(a - b).mean(axis=1)
    print(f"{name}")
    print(f"  shape[{rows},{K}] kind={ent['kind']}  相对 L2 误差={rel:.4f}  余弦={cos:.6f}")
    print(f"  每行平均绝对误差: max={row_err.max():.3e} median={np.median(row_err):.3e}")
    print(f"  样例 rt4[0,:5]={a[0,:5]}  src[0,:5]={b[0,:5]}")
    print(f"  manifest relerr 声明={ent['relerr']:.4f}  src_rms={ent['src_rms']:.5f}  实测源 rms={np.sqrt((b**2).mean()):.5f}")


if __name__ == "__main__":
    main()
