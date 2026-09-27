#!/usr/bin/env python3
"""本地视觉塔前端：图片预处理 + 调用自研 RT4 视觉编码器。

真正的 27 层视觉块在 build/rt 里执行（加载 RT4 权重）；这里只做
Qwen2VLImageProcessor 等价的预处理，把 patch 写成 f32 文件交给引擎。
"""
import base64
import hashlib
import io
import os
import tempfile
import threading
import urllib.request
from collections import OrderedDict

import numpy as np
from PIL import Image, ImageOps


def _load_image(image_url):
    if image_url.startswith('data:'):
        if ',' not in image_url:
            raise ValueError('不支持的数据 URL')
        raw = base64.b64decode(image_url.split(',', 1)[1])
    elif image_url.startswith('file://'):
        with open(image_url[7:], 'rb') as f:
            raw = f.read()
    elif image_url.startswith(('http://', 'https://')):
        with urllib.request.urlopen(image_url, timeout=30) as r:
            raw = r.read()
    else:
        raw = base64.b64decode(image_url)
    return ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert('RGB')


class LocalVisionEncoder:
    """预处理图片并驱动引擎的 IMG_EMB 命令。"""

    def __init__(self, engine=None, vision_rt4=None):
        from transformers.models.qwen2_vl import Qwen2VLImageProcessor
        self.engine = engine
        self.processor = Qwen2VLImageProcessor(
            patch_size=16,
            temporal_patch_size=2,
            merge_size=2,
            min_pixels=int(os.environ.get('RT_VISION_MIN_PIXELS', '3136')),
            max_pixels=int(os.environ.get('RT_VISION_MAX_PIXELS', '1003520')),
        )
        self.vision_rt4 = vision_rt4
        self.cache = OrderedDict()
        self.cache_max = int(os.environ.get('RT_VISION_CACHE', '16'))
        self.lock = threading.Lock()
        self.last_ms = 0.0

    def encode(self, image_url):
        """返回 (embeds f32 [N,5120], grid_thw list)。"""
        if not image_url:
            raise ValueError('空图片')
        if self.engine is None:
            raise RuntimeError('引擎未就绪')
        key = hashlib.sha256(image_url.encode('utf-8')).hexdigest()
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]

        image = _load_image(image_url)
        with self.lock:
            batch = self.processor(images=[image], return_tensors='pt')
            patches = batch['pixel_values'].numpy().astype('<f4')
            grid = batch['image_grid_thw'].tolist()[0]
            _, gh, gw = grid
            with tempfile.TemporaryDirectory(prefix='rt4-vision-') as td:
                patch_path = os.path.join(td, 'patches.f32')
                out_path = os.path.join(td, 'emb.f32')
                patches.tofile(patch_path)
                n, ms = self.engine.image_embed(patch_path, out_path, gh, gw)
                embeds = np.fromfile(out_path, dtype='<f4').reshape(n, 5120)
                self.last_ms = ms
        if not np.isfinite(embeds).all():
            raise RuntimeError('视觉塔输出含 NaN/Inf')
        self.cache[key] = (embeds, grid)
        while len(self.cache) > self.cache_max:
            self.cache.popitem(last=False)
        return embeds, grid

    @property
    def name(self):
        return 'qwen3.5-vision-rt4'
