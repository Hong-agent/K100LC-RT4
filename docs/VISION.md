# 本地视觉塔接入（Qwen3.5 / K100_LC）

## 1. 结论

网页上传图片现在是**端到端可用**的：图片经 Qwen3.5 视觉塔编码后，作为
`<|image_pad|>` 位置的词嵌入注入自研文本运行时，主模型直接基于图像 embedding 回答。
视觉塔本身也是 RT4：线性层 f16，norm/bias/pos_embed f32，27 层视觉块全部在
自研 C++/HIP 运行时里执行；Python 只做图片预处理。

实测（K100_LC，单卡）：

| 用例 | 结果 |
|---|---|
| 上传写着 `7788` 的图片，问“识别四位数字” | 回答 `7788` |
| 一次上传 `12`、`34` 两张图，问按顺序读出 | 回答 `12\n34` |
| 单图视觉编码（91 token，RT4/C++） | 约 242 ms/图 |
| 流式输出 | 正常，SSE 增量中能看到 `7 / 7 / 8 / 8` |
| MTP3 开启时 | 图片 prefill 后仍正常投机，实测接受 2~3 token/轮 |
| 纯文本请求 | 不受影响，仍走原 `PREFILL` |

## 2. 权重精度选择

源模型文本塔是 NVFP4/FP8，转换后是 RT4 int4；视觉塔 `model.visual.*` 原本是
BF16，只有 0.92GB，27 层。实测把视觉线性层也压到 int4/128 后，单层误差
12%~19%，27 层累积会把最终 embedding 余弦打到 0.001，图片语义完全丢失。

所以 RT4 视觉权重采用 **f16 线性层 + f32 norm/bias/pos_embed**：

* 仍然是独立的 RT4 文件，由同一个 C++ 运行时加载；
* 视觉塔 27 层（patch embed、qkv/proj、MLP、merger）全部在 HIP 运行时执行；
* 与 transformers 参考逐层对齐：layer0/1 余弦 1.0，最终 embedding
  余弦 0.999998、相对误差约 0.18%。

文本权重保持 13.91GB int4，视觉 RT4 约 0.93GB；32k KV cache 与 MTP 同时开启
也在 64GB HBM 预算内。

## 3. 权重导出

```bash
bash scripts/prepare_vision.sh
```

它把源 `model.safetensors` 里的 333 个 `model.visual.*` 张量写成：

```
models/Qwen3.8-27B-NVFP4/rt4/
├── qwen38_27b_vision.rt4
└── qwen38_27b_vision.rt4.json
```

线性层：N 补到 64、K 补到 32，存 f16；norm/bias/pos_embed 存 f32。
导出器逐张量顺序读取，不 mmap 22.57GB 源文件。

## 4. 图像预处理

使用 transformers 的 `Qwen2VLImageProcessor`，参数与视觉塔配置一致：

* `patch_size=16`
* `temporal_patch_size=2`
* `merge_size=2`
* 默认 `min_pixels=3136`、`max_pixels=1003520`（可用
  `RT_VISION_MIN_PIXELS` / `RT_VISION_MAX_PIXELS` 调整）

输出 `pixel_values` 为 `[t*h*w, 3*2*16*16]`，`image_grid_thw` 为 `[1,h,w]`。
视觉塔返回 `pooler_output`，即空间合并后的 `[num_vision_tokens, 5120]`。

## 5. 运行时协议

服务端先调用引擎：

```
IMG_EMB <patch_file> <out_file> <gh> <gw>
```

引擎在自己的 RT4 视觉塔上跑 27 层，把 `[num_vision_tokens, 5120]` f32 写到
`out_file`，回 `OK image <N>`。

文本模板把每张图片替换为：

```
<|vision_start|><|image_pad|> × N<|vision_end|>
```

`N` 等于视觉塔输出的 embedding 行数（通常几十到几百，取决于图片分辨率）。
服务端再把所有图片 embedding 按顺序拼成一个 f32 文件，然后发送：

```
PREFILL_EMB <ids_csv> <emb_file> <start:count,...>
```

C++ 侧在 `k_embed` 之后、第一层之前，把对应行覆盖成视觉 embedding；分块
prefill 时按全局 token 下标取交集，因此图片段跨 512-token chunk 也不会错位。
详细实现见 `src/model.cpp` 的 `EmbSpan` 和 `PREFILL_EMB` 分支。

## 6. 开关

* 默认 `RT_VISION_MODE=auto`：有 `rt4/qwen38_27b_vision.rt4` 就用本地视觉塔；
  同时配置了 `RT_VISION_BASE_URL` + `RT_VISION_MODEL` 时优先用外部视觉桥。
* `RT_VISION_MODE=local`：强制本地。
* `RT_VISION_MODE=off`：关闭视觉，图片只作为会话附件展示。
* 相关变量：`RT_VISION_RT4`（自定义视觉 RT4 路径）、`RT_VISION_MAX_PATCHES`
  （默认 4096）、`RT_VISION_CACHE`（默认缓存 16 张图）。

## 7. 已知边界

* 视觉线性层是通用 f16 GEMM（64x64x32 tile），没有使用矩阵核心；后续可以
  针对视觉形状继续调 tile / 双缓冲。
* 图片分辨率上限默认约 1MP；超大图会被 `Qwen2VLImageProcessor` 按长宽比缩放。
* 多图 token 会占上下文：一张约 90 token 的图，prefill token 数增加约 90-1。
* 复杂版式 OCR、细小文字仍取决于原模型视觉塔本身的能力。
