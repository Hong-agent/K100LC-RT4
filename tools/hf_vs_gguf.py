#!/usr/bin/env python3
# 逐行比对 HF safetensors（含 fp8 / NVFP4 反量化）与 BF16 GGUF 的同一张量。
# 用途：确认转换过程有没有做“看不见的变换”（例如 gated delta net 的 q/k 头重排）。
#
#   python3 hf_vs_gguf.py <safetensors> <hf名> <gguf> <gguf名> <若干行号...>
import sys, json, struct, mmap
import numpy as np

DT = {'F32': np.float32, 'F16': np.float16, 'BF16': None, 'F8_E4M3': np.uint8, 'U8': np.uint8}

def st_open(path):
    f = open(path, 'rb')
    buf = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    n = struct.unpack_from('<Q', buf, 0)[0]
    hdr = json.loads(buf[8:8+n])
    base = 8 + n
    return buf, base, hdr

def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)

def st_read(buf, base, hdr, name):
    info = hdr[name]
    dt, shape = info['dtype'], info['shape']
    o0, o1 = info['data_offsets']
    raw = buf[base+o0: base+o1]
    if dt == 'F32':   a = np.frombuffer(raw, np.float32)
    elif dt == 'F16': a = np.frombuffer(raw, np.float16).astype(np.float32)
    elif dt == 'BF16':a = bf16_to_f32(np.frombuffer(raw, np.uint16))
    elif dt == 'F8_E4M3':
        c = np.frombuffer(raw, np.uint8)
        f8 = c.view(np.dtype('float8_e4m3fn')) if hasattr(np, 'dtype') and 'float8_e4m3fn' in np.sctypeDict else None
        a = np.zeros(c.shape, np.float32)
        s = (c >> 7) & 1; e = (c >> 3) & 0xF; m = c & 7
        val = np.where(e == 0, m / 8.0 * 2.0**-6, (1 + m/8.0) * np.power(2.0, e.astype(np.float32) - 7))
        a = np.where(s == 1, -val, val).astype(np.float32)
    else:
        raise RuntimeError('dtype ' + dt)
    return a.reshape(shape), dt

def main():
    st, hfname, gp, gname = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    rows = [int(x) for x in sys.argv[5:]]
    buf, base, hdr = st_open(st)
    a, dt = st_read(buf, base, hdr, hfname)
    if hfname + '.weight_scale' in hdr:                 # fp8：逐输出通道尺度
        sc, _ = st_read(buf, base, hdr, hfname + '.weight_scale')
        a = a * sc.reshape(-1, 1).astype(np.float32)
        print(f'HF {hfname} dtype={dt} + channel scale, shape={a.shape}')
    else:
        print(f'HF {hfname} dtype={dt}, shape={a.shape}')

    sys.path.insert(0, __file__.rsplit('/', 1)[0])
    import gguf_probe as G
    gbuf, ver, kv, tensors, data_off = G.parse_header(gp)
    dims, t, off = tensors[gname]
    print(f'GGUF {gname} dims={dims} type={t}')
    gtype = {0: np.float32, 1: np.float16, 30: np.uint16}[t]
    row_len = dims[0]                                   # ne0 = 输入维（最快）
    for r in rows:
        raw = gbuf[data_off + off + r*row_len*(4 if t == 0 else 2):
                   data_off + off + (r+1)*row_len*(4 if t == 0 else 2)]
        gv = np.frombuffer(raw, gtype).astype(np.float32) if t != 30 else bf16_to_f32(np.frombuffer(raw, np.uint16))
        hv = a[r].astype(np.float32)
        num = float(np.dot(gv, hv)); den = float(np.linalg.norm(gv) * np.linalg.norm(hv))
        print(f'  row {r:5d}: cos={num/(den+1e-30):.6f}  max|Δ|={np.max(np.abs(gv-hv)):.4e}  '
              f'|hf|rms={np.sqrt((hv**2).mean()):.4f}  |gguf|rms={np.sqrt((gv**2).mean()):.4f}')

if __name__ == '__main__':
    main()
