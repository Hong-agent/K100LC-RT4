#!/usr/bin/env python3
"""用 BF16 原模型（GGUF）核对 RT4 manifest 里的 src_rms，验证源反量化公式没搞错。

需要 numpy（在本机 DTK 容器里跑）。
用法:
  verify_vs_gguf.py --list <gguf...>
  verify_vs_gguf.py --check <manifest.json> <gguf...> [层数上限]
"""
import json
import struct
import sys
import numpy as np

GGML_SIZES = {0: 4, 1: 2, 30: 2, 12: None, 23: None, 2: None}  # 只支持 F32/F16/BF16 求 rms


class Gguf:
    def __init__(self, path):
        self.path = path
        self.fh = open(path, "rb")
        magic, self.version = struct.unpack("<II", self.fh.read(8))
        assert magic == 0x46554747, f"bad magic {magic:x}"
        self.n_tensors, self.n_kv = struct.unpack("<QQ", self.fh.read(16))
        self.alignment = 32
        for _ in range(self.n_kv):
            key = self._str()
            t = struct.unpack("<I", self.fh.read(4))[0]
            if key == "general.alignment":
                self.alignment = struct.unpack("<I", self.fh.read(4))[0]
                continue
            self._skip_value(t)
        self.tensors = {}
        for _ in range(self.n_tensors):
            name = self._str()
            nd = struct.unpack("<I", self.fh.read(4))[0]
            dims = struct.unpack(f"<{nd}Q", self.fh.read(8 * nd))
            ttype, off = struct.unpack("<IQ", self.fh.read(12))
            self.tensors[name] = (dims, ttype, off)
        pos = self.fh.tell()
        self.data_start = pos + (-pos % self.alignment)

    def _str(self):
        n = struct.unpack("<Q", self.fh.read(8))[0]
        return self.fh.read(n).decode("utf-8")

    def _skip_value(self, t):
        if t == 8:
            self._str()
        elif t == 9:
            et = struct.unpack("<I", self.fh.read(4))[0]
            n = struct.unpack("<Q", self.fh.read(8))[0]
            if et == 8:
                for _ in range(n):
                    self._str()
            else:
                sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
                self.fh.read(sizes[et] * n)
        else:
            sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
            self.fh.read(sizes[t])

    def rms(self, name, max_elems=40_000_000):
        dims, ttype, off = self.tensors[name]
        n = 1
        for d in dims:
            n *= d
        n_read = min(n, max_elems)
        if ttype == 30:
            self.fh.seek(self.data_start + off)
            raw = np.frombuffer(self.fh.read(2 * n_read), dtype="<u2").astype(np.uint32) << 16
            vals = raw.view(np.float32)
        elif ttype == 1:
            self.fh.seek(self.data_start + off)
            vals = np.frombuffer(self.fh.read(2 * n_read), dtype="<f2").astype(np.float32)
        elif ttype == 0:
            self.fh.seek(self.data_start + off)
            vals = np.frombuffer(self.fh.read(4 * n_read), dtype="<f4")
        else:
            return None, n, ttype
        return float(np.sqrt(np.mean(vals.astype(np.float64) ** 2))), n, ttype


HF2GGUF = [
    ("model.language_model.embed_tokens.weight", "token_embd.weight"),
    ("lm_head.weight", "output.weight"),
]


def hf_to_gguf(hf):
    for a, b in HF2GGUF:
        if hf == a:
            return b
    parts = hf.split(".")
    if len(parts) > 3 and parts[1] == "language_model" and parts[2] == "layers":
        n = parts[3]
        tail = ".".join(parts[4:]).replace(".weight", "")
        m = {
            "mlp.gate_proj": "ffn_gate",
            "mlp.up_proj": "ffn_up",
            "mlp.down_proj": "ffn_down",
            "self_attn.q_proj": "attn_q",
            "self_attn.k_proj": "attn_k",
            "self_attn.v_proj": "attn_v",
            "self_attn.o_proj": "attn_output",
            "linear_attn.in_proj_qkv": "attn_qkv",
            "linear_attn.in_proj_z": "attn_gate",
            "linear_attn.out_proj": "ssm_out",
            "linear_attn.in_proj_a": "ssm_alpha",
            "linear_attn.in_proj_b": "ssm_beta",
            "input_layernorm": "attn_norm",
            "post_attention_layernorm": "post_attention_norm",
        }
        if tail in m:
            return f"blk.{n}.{m[tail]}.weight"
    return None


def main():
    mode = sys.argv[1]
    if mode == "--list":
        for p in sys.argv[2:]:
            g = Gguf(p)
            for name, (dims, ttype, off) in g.tensors.items():
                print(f"{p}\t{name}\t{dims}\ttype={ttype}")
        return

    manifest = json.load(open(sys.argv[2]))
    glist = sys.argv[3:-1] if sys.argv[-1].isdigit() else sys.argv[3:]
    limit = int(sys.argv[-1]) if sys.argv[-1].isdigit() else None
    ggs = [Gguf(p) for p in glist]

    def find(name):
        for g in ggs:
            if name in g.tensors:
                return g
        return None

    checked = 0
    for t in manifest["tensors"]:
        if t["kind"] not in ("i4", "i8"):
            continue
        gname = hf_to_gguf(t["name"])
        if not gname:
            continue
        g = find(gname)
        if not g:
            continue
        r, n, ttype = g.rms(gname)
        if r is None:
            continue
        ratio = r / t["src_rms"] if t["src_rms"] else 0
        flag = "OK " if 0.96 < ratio < 1.04 else "!! "
        print(f"{flag}{t['name'][:70]:70s} gguf_rms={r:.5f} rt4_src_rms={t['src_rms']:.5f} "
              f"ratio={ratio:.4f} relerr={t['relerr']:.4f} type={ttype}")
        checked += 1
        if limit and checked >= limit:
            break
    print(f"checked {checked} tensors")


if __name__ == "__main__":
    main()
