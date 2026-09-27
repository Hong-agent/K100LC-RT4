#!/usr/bin/env python3
# GGUF 探测工具：列张量表、抽一小块权重，用来跟 HF safetensors 对齐（查转换时有没有
# 做过头维重排之类的“看不见的变换”）。纯 mmap，不把整个文件读进内存。
#
# 用法:
#   python3 gguf_probe.py list  <gguf> [子串]           # 列表
#   python3 gguf_probe.py slice <gguf> <张量名> <行起> <行数>   # 打印若干行的前几个值
import sys, struct, mmap, os

GGUF_MAGIC = 0x46554747
U8,I8,U16,I16,U32,I32,F32,BOOL,STRING,ARRAY,U64,I64,F64 = range(13)
SCALAR = {U8:('B',1), I8:('b',1), U16:('H',2), I16:('h',2), U32:('I',4), I32:('i',4),
          F32:('f',4), BOOL:('?',1), U64:('Q',8), I64:('q',8), F64:('d',8)}
GGML_TYPES = {0:('f32',4),1:('f16',2),2:('q4_0',None),3:('q4_1',None),30:('bf16',2)}

def rd(buf, off, fmt, n=1):
    sz = struct.calcsize(fmt)*n
    return struct.unpack_from(fmt, buf, off)[0] if n == 1 else struct.unpack_from(fmt, buf, off), off+sz

def rdstr(buf, off):
    n, off = rd(buf, off, 'Q')
    s = buf[off:off+n].decode('utf-8', 'replace')
    return s, off+n

def rdval(buf, off, t):
    if t in SCALAR:
        f, _ = SCALAR[t]; return rd(buf, off, f)
    if t == STRING:
        return rdstr(buf, off)
    if t == ARRAY:
        et, off = rd(buf, off, 'I')
        n, off = rd(buf, off, 'Q')
        vals = []
        for _ in range(min(n, 8)):
            v, off = rdval(buf, off, et)
            vals.append(v)
        if et in SCALAR:                 # 跳过剩下的
            _, sz = SCALAR[et]
            off += sz*(n-len(vals))
        elif et == STRING:               # 字符串数组：逐个跳
            for _ in range(n-len(vals)):
                _, off = rdstr(buf, off)
        return (f'[{n}] {vals}...' if n > 8 else vals), off
    raise RuntimeError(f'bad type {t}')

def parse_header(path):
    f = open(path, 'rb')
    buf = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    off = 0
    magic, off = rd(buf, off, 'I')
    assert magic == GGUF_MAGIC, hex(magic)
    ver, off = rd(buf, off, 'I')
    ntensors, off = rd(buf, off, 'Q')
    nkv, off = rd(buf, off, 'Q')
    kv = {}
    for _ in range(nkv):
        k, off = rdstr(buf, off)
        t, off = rd(buf, off, 'I')
        v, off = rdval(buf, off, t)
        kv[k] = v
    tensors = {}
    for _ in range(ntensors):
        name, off = rdstr(buf, off)
        nd, off = rd(buf, off, 'I')
        dims = []
        for _ in range(nd):
            d, off = rd(buf, off, 'Q'); dims.append(d)
        t, off = rd(buf, off, 'I')
        o, off = rd(buf, off, 'Q')
        tensors[name] = (dims, t, o)
    data_off = (off + 31) // 32 * 32
    return buf, ver, kv, tensors, data_off

def main():
    mode, path = sys.argv[1], sys.argv[2]
    buf, ver, kv, tensors, data_off = parse_header(path)
    if mode == 'list':
        pat = sys.argv[3] if len(sys.argv) > 3 else ''
        print(f'gguf v{ver}  tensors={len(tensors)}  kv={len(kv)}  data_off={data_off}')
        for k in list(kv)[:12]:
            print('  kv', k, '=', str(kv[k])[:90])
        for name, (dims, t, o) in tensors.items():
            if pat in name:
                print(f'  {name:58s} {GGML_TYPES.get(t,(t,None))[0]:5s} {dims}  off={o}')
    elif mode == 'slice':
        name = sys.argv[3]; r0 = int(sys.argv[4]); nr = int(sys.argv[5])
        dims, t, o = tensors[name]
        assert t in (0, 1, 30), f'only f32/f16/bf16 supported, got {t}'
        elem = 4 if t == 0 else 2
        row = dims[0]
        for r in range(r0, min(r0+nr, dims[1] if len(dims) > 1 else 1)):
            off = data_off + o + r*row*elem
            raw = buf[off:off+row*elem]
            if t == 0:
                vals = struct.unpack_from(f'<{row}f', raw)
            elif t == 1:
                vals = [struct.unpack_from('<e', raw, 2*i)[0] for i in range(row)]
            else:
                import numpy as np
                vals = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
                vals = vals.view(np.float32)
            if len(vals) > 8:
                print(f'  row {r}: {vals[0]:.6f} {vals[1]:.6f} {vals[2]:.6f} {vals[3]:.6f} | '
                      f'{vals[-4]:.6f} {vals[-3]:.6f} {vals[-2]:.6f} {vals[-1]:.6f}   sum={sum(vals):.4f}')
            else:
                print(f'  row {r}: {list(vals)}')

if __name__ == '__main__':
    main()
