#!/usr/bin/env python3
"""Qwen3.5 视觉塔的 CPU 前向（NumPy）。

权重直接读 `qwen38_27b_vision.rt4`：线性层 f16，norm/bias/pos_embed f32。
实现逐层对齐 `src/model.cpp` 的 VisionModel / `src/k_vision.hip`，不依赖
torch、transformers 或 HIP；图像预处理也按 Qwen2VLImageProcessor 的规则实现。
"""
import json
import mmap
import os
import threading
import time
from collections import OrderedDict

import numpy as np
from PIL import Image


H = 1152
HEADS = 16
HD = 72
INTER = 4304
MERGE_IN = 4608
OUT_H = 5120
PATCH_DIM = 3 * 2 * 16 * 16
POS_N = 2304
POS_SIDE = 48
DEPTH = 27
EPS = 1e-6

_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def _smart_resize(height, width, factor=32, min_pixels=3136, max_pixels=1003520):
    if max(height, width) / min(height, width) > 200:
        raise ValueError('图片长宽比超过 200')
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = np.sqrt((height * width) / max_pixels)
        h_bar = max(factor, int(np.floor(height / beta / factor)) * factor)
        w_bar = max(factor, int(np.floor(width / beta / factor)) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = np.sqrt(min_pixels / (height * width))
        h_bar = int(np.ceil(height * beta / factor)) * factor
        w_bar = int(np.ceil(width * beta / factor)) * factor
    return h_bar, w_bar


def preprocess_image(image, min_pixels=3136, max_pixels=1003520, patch_size=16,
                     temporal_patch_size=2, merge_size=2):
    """返回 (pixel_values [N,1536] f32, image_grid_thw [1,gh,gw])。"""
    width, height = image.size
    factor = patch_size * merge_size
    h_bar, w_bar = _smart_resize(height, width, factor, min_pixels, max_pixels)
    resample = getattr(Image, 'Resampling', Image).BICUBIC
    image = image.resize((w_bar, h_bar), resample)

    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    arr = arr.transpose(2, 0, 1)[None, None, :, :, :]  # [1,1,C,H,W]
    if arr.shape[1] % temporal_patch_size:
        last = arr[:, -1:]
        arr = np.concatenate([arr, np.repeat(last, temporal_patch_size - arr.shape[1],
                                             axis=1)], axis=1)

    grid_t = arr.shape[1] // temporal_patch_size
    gh, gw = h_bar // patch_size, w_bar // patch_size
    arr = arr.reshape(1, grid_t, temporal_patch_size, 3,
                      gh // merge_size, merge_size, patch_size,
                      gw // merge_size, merge_size, patch_size)
    arr = arr.transpose(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
    patches = arr.reshape(1, grid_t * gh * gw,
                          3 * temporal_patch_size * patch_size * patch_size)
    return np.ascontiguousarray(patches[0], dtype=np.float32), [1, gh, gw]


def _erf(x):
    """Abramowitz & Stegun 7.1.26，最大误差约 1.5e-7。"""
    sign = np.sign(x)
    ax = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                - 0.284496736) * t + 0.254829592) * t * np.exp(-ax * ax)
    return sign * y


def _gelu(x, exact=False):
    if exact:
        return 0.5 * x * (1.0 + _erf(x * 0.7071067811865476))
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 *
                                    (x + 0.044715 * x * x * x)))


class CpuVisionEncoder:
    def __init__(self, rt4_path, min_pixels=None, max_pixels=None, cache_max=16):
        self.rt4_path = rt4_path
        self.min_pixels = int(min_pixels or os.environ.get('RT_VISION_MIN_PIXELS', '3136'))
        self.max_pixels = int(max_pixels or os.environ.get('RT_VISION_MAX_PIXELS', '1003520'))
        self.cache_max = int(os.environ.get('RT_VISION_CACHE', str(cache_max)))
        self.cache = OrderedDict()
        self.lock = threading.Lock()
        self.last_ms = 0.0
        self._load()

    def _load(self):
        if not os.path.exists(self.rt4_path):
            raise FileNotFoundError(self.rt4_path)
        with open(self.rt4_path + '.json', encoding='utf-8') as f:
            manifest = json.load(f)
        self.tensors = {t['name']: t for t in manifest['tensors']}
        self.fp = open(self.rt4_path, 'rb')
        self.mm = mmap.mmap(self.fp.fileno(), 0, access=mmap.ACCESS_READ)

        self.patch_w = self.tensor('model.visual.patch_embed.proj.weight')
        self.patch_b = self.tensor('model.visual.patch_embed.proj.bias')
        self.pos_embed = self.tensor('model.visual.pos_embed.weight')
        self.layers = []
        for il in range(DEPTH):
            p = f'model.visual.blocks.{il}.'
            self.layers.append({
                'n1w': self.tensor(p + 'norm1.weight'), 'n1b': self.tensor(p + 'norm1.bias'),
                'n2w': self.tensor(p + 'norm2.weight'), 'n2b': self.tensor(p + 'norm2.bias'),
                'qkvw': self.tensor(p + 'attn.qkv.weight'), 'qkvb': self.tensor(p + 'attn.qkv.bias'),
                'projw': self.tensor(p + 'attn.proj.weight'), 'projb': self.tensor(p + 'attn.proj.bias'),
                'fc1w': self.tensor(p + 'mlp.linear_fc1.weight'),
                'fc1b': self.tensor(p + 'mlp.linear_fc1.bias'),
                'fc2w': self.tensor(p + 'mlp.linear_fc2.weight'),
                'fc2b': self.tensor(p + 'mlp.linear_fc2.bias'),
            })
        self.mnw = self.tensor('model.visual.merger.norm.weight')
        self.mnb = self.tensor('model.visual.merger.norm.bias')
        self.mfc1w = self.tensor('model.visual.merger.linear_fc1.weight')
        self.mfc1b = self.tensor('model.visual.merger.linear_fc1.bias')
        self.mfc2w = self.tensor('model.visual.merger.linear_fc2.weight')
        self.mfc2b = self.tensor('model.visual.merger.linear_fc2.bias')

    def tensor(self, name):
        t = self.tensors[name]
        dt = np.float16 if t['kind'] == 'f16' else np.float32
        count = int(np.prod(t['shape']))
        arr = np.frombuffer(self.mm, dtype=dt, count=count, offset=int(t['q_off']))
        return arr.reshape([int(x) for x in t['shape']])

    @staticmethod
    def _layernorm(x, w, b):
        mean = x.mean(axis=1, keepdims=True, dtype=np.float32)
        var = x.var(axis=1, keepdims=True, dtype=np.float32)
        return (x - mean) / np.sqrt(var + EPS) * w + b

    @staticmethod
    def _linear(x, w16, b=None, rows=None):
        if rows is not None:
            w16 = w16[:rows]
        # GPU 内核把激活也先转成 f16，再以 f32 累加；这里保持一致。
        x32 = x.astype(np.float16).astype(np.float32)
        w32 = np.asarray(w16, dtype=np.float32)
        y = x32 @ w32.T
        if b is not None:
            y += b
        return y

    @staticmethod
    def _lin_value(i, n):
        return 0.0 if n <= 1 else (i * (POS_SIDE - 1) / (n - 1)).astype(np.float32)

    def _pos(self, gh, gw, n):
        idx = np.arange(n, dtype=np.int64)
        mw = idx % 2
        t = idx // 2
        mh = t % 2
        t //= 2
        gc = t % (gw // 2)
        gr = t // (gw // 2)
        row = gr * 2 + mh
        col = gc * 2 + mw
        hf = self._lin_value(row, gh)
        wf = self._lin_value(col, gw)
        h0 = np.floor(hf).astype(np.int64)
        w0 = np.floor(wf).astype(np.int64)
        h1 = np.minimum(h0 + 1, POS_SIDE - 1)
        w1 = np.minimum(w0 + 1, POS_SIDE - 1)
        dh = (hf - h0)[:, None]
        dw = (wf - w0)[:, None]
        pe = self.pos_embed
        p00 = pe[h0 * POS_SIDE + w0]
        p01 = pe[h0 * POS_SIDE + w1]
        p10 = pe[h1 * POS_SIDE + w0]
        p11 = pe[h1 * POS_SIDE + w1]
        return ((1 - dh) * (1 - dw) * p00 + (1 - dh) * dw * p01 +
                dh * (1 - dw) * p10 + dh * dw * p11)

    def _rope(self, gh, gw, n):
        nf = 18
        inv = 1.0 / (10000.0 ** (2.0 * np.arange(nf) / 36.0))
        ft = np.arange(max(gh, gw), dtype=np.float32)[:, None] * inv[None, :]
        idx = np.arange(n, dtype=np.int64)
        mw = idx % 2
        t = idx // 2
        mh = t % 2
        t //= 2
        gc = t % (gw // 2)
        gr = t // (gw // 2)
        row = gr * 2 + mh
        col = gc * 2 + mw
        c = np.empty((n, HD), dtype=np.float32)
        s = np.empty((n, HD), dtype=np.float32)
        c[:, :nf] = np.cos(ft[row])
        s[:, :nf] = np.sin(ft[row])
        c[:, nf:2 * nf] = np.cos(ft[col])
        s[:, nf:2 * nf] = np.sin(ft[col])
        c[:, 2 * nf:3 * nf] = c[:, :nf]
        s[:, 2 * nf:3 * nf] = s[:, :nf]
        c[:, 3 * nf:] = c[:, nf:2 * nf]
        s[:, 3 * nf:] = s[:, nf:2 * nf]
        return c, s

    @staticmethod
    def _apply_rope(qkv, c, s):
        half = HD // 2
        c = c[:, :half][:, None, :]
        s = s[:, :half][:, None, :]
        for part in (0, 1):
            x0 = qkv[:, part, :, :half].copy()
            x1 = qkv[:, part, :, half:].copy()
            qkv[:, part, :, :half] = x0 * c - x1 * s
            qkv[:, part, :, half:] = x1 * c + x0 * s
        return qkv

    @staticmethod
    def _attention(qkv, n):
        scale = 1.0 / np.sqrt(float(HD))
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        out = np.empty((n, HEADS, HD), dtype=np.float32)
        for h in range(HEADS):
            scores = (q[:, h, :] @ k[:, h, :].T) * scale
            scores -= scores.max(axis=1, keepdims=True)
            np.exp(scores, out=scores)
            scores /= scores.sum(axis=1, keepdims=True)
            out[:, h, :] = scores @ v[:, h, :]
        return out.reshape(n, H)

    def encode_patches(self, patches, gh, gw):
        n = patches.shape[0]
        if n <= 0 or n != gh * gw:
            raise ValueError(f'patch 数 {n} 与 grid {gh}x{gw} 不一致')
        if n % 4:
            raise ValueError('patch 数必须是 4 的倍数')

        x = self._linear(patches, self.patch_w, self.patch_b)
        x += self._pos(gh, gw, n)
        cos, sin = self._rope(gh, gw, n)
        for il, L in enumerate(self.layers):
            y = self._layernorm(x, L['n1w'], L['n1b'])
            qkv = self._linear(y, L['qkvw'], L['qkvb'])
            qkv = self._apply_rope(qkv.reshape(n, 3, HEADS, HD), cos, sin)
            attn = self._attention(qkv, n)
            x += self._linear(attn, L['projw'], L['projb'])

            y = self._layernorm(x, L['n2w'], L['n2b'])
            h = self._linear(y, L['fc1w'], L['fc1b'], rows=INTER)
            h = _gelu(h, exact=False)
            x += self._linear(h, L['fc2w'][:, :INTER], L['fc2b'])

        x = self._layernorm(x, self.mnw, self.mnb)
        m = x.reshape(n // 4, MERGE_IN)
        m = self._linear(m, self.mfc1w, self.mfc1b)
        m = _gelu(m, exact=True)
        return self._linear(m, self.mfc2w, self.mfc2b)

    def encode_image(self, image):
        patches, grid = preprocess_image(image, self.min_pixels, self.max_pixels)
        t0 = time.perf_counter()
        out = self.encode_patches(patches, grid[1], grid[2])
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        return out, grid

    def encode(self, image, cache_key=None):
        with self.lock:
            if cache_key and cache_key in self.cache:
                self.cache.move_to_end(cache_key)
                return self.cache[cache_key]
            out = self.encode_image(image)
            if cache_key:
                self.cache[cache_key] = out
                while len(self.cache) > self.cache_max:
                    self.cache.popitem(last=False)
            return out

    @property
    def name(self):
        return 'qwen3.5-vision-cpu'
