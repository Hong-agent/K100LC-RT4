# 从「全是乱码」到「能对话」——2026-09-27 深夜的一轮排错

结论先行：**模型现在能出连贯的话了**（中文/英文都对），
`bash scripts/serve.sh` 起 OpenAI 兼容服务，`python3 scripts/chat.py` 直接命令行对话。
这一轮一共修掉 **7 个真 bug**，其中一个是致命的：**权重转换器把 449 张非量化张量写坏了**。

装好后的实测（1.65k 预填充 + 64 token 解码，单卡）：

| 项 | 实测 |
|---|---|
| 预填充 | 64 tok 146 t/s（含预热）/ 417 tok 258 t/s / 1649 tok 273 t/s |
| 解码（贪心） | **12 tok/s**（83 ms/token，还没优化，见文末） |
| 生成质量 | 中文自我介绍/量子纠缠解释/数字比较都通顺；英文常识对（"the lazy → dog"、"hydrogen and → oxygen"、"The capital of France is" 的 top-2 是 ` ` 空格和 ` Paris`，token 级 int4 权重的残余误差让 top-1 偶尔偏一个位置） |

---

## 1. 致命 bug：转换器把 BF16 源写成「f32」时按 4 字节原样搬

`tools/convert.c: convert_plain()` 原来长这样：

```c
} else {                                  /* 目标格式是 f32 */
    fwrite(g_data + t->off, 4, n, g_out); /* ← 直接把源字节按 4 字节搬过去 */
}
```

源权重里这些张量是 **BF16**（2 字节/元素），而 RT4 里标成 `f32`（4 字节/元素）。
结果：

* 数组前半 = 把相邻两个 bf16 当成一个 f32（低 16 位尾数是**下一个数**的位），
* 数组后半 = **越界读到下一张量的数据**。

受影响的是 353 张 f32 张量：**每层的 input/post layernorm、A_log、dt_bias、conv1d、
ssm_norm（门控 RMSNorm）、最终 norm** —— 也就是把这个模型的归一化与 SSM 门控整个搞坏：
实测第 0 层输出 RMS 从 **0.228（正确权重）变成 2.177（坏权重）**，残差流被放大 10 倍，
64 层之后完全变成乱码。

修完之后 `tools/verify_plain.py` 全绿：

```
$ python3 tools/verify_plain.py model.safetensors rt4/qwen38_27b.rt4 rt4/qwen38_27b.rt4.json
非量化张量：一致 449，异常 0
```

**为什么之前的对照没发现**：`tools/ref_layer.py` / `ref_stages.py` 都是拿 *同一份 RT4*
当参考权重，两边用同一份坏权重，自然「对得上」（逐阶段 1~3%）。要发现这类 bug，
参考端**必须回到源 safetensors**（`--src` 选项，见下）。

复现/防止回归：

```bash
bash scripts/convert_weights.sh          # 重新转换（约 5 分钟）
bash scripts/dsh.sh 'python3 tools/verify_plain.py /rt/models/Qwen3.8-27B-NVFP4/model.safetensors \
  /rt/models/Qwen3.8-27B-NVFP4/rt4/qwen38_27b.rt4 /rt/models/Qwen3.8-27B-NVFP4/rt4/qwen38_27b.rt4.json'
```

## 2. 同一节发现的第二个转换问题（已确认无误）

源模型是**混合精度**：

* `group_0`：self_attn q/k/v/o、linear_attn in_qkv/in_z/out_proj、lm_head、56~63 层的 MLP
  —— **FP8(e4m3) 权重 + per-channel 尺度**，激活 FP8 per-token；
* `group_1`：0~55 层 mlp gate/up/down —— **NVFP4（4bit，每 16 一个 e4m3 尺度）**；
* 其余（embedding、norm、conv1d、in_proj_a/b、A_log、dt_bias）—— BF16。

RT4 把所有 4/8bit 权重统一重编码成「int4 + 每 128 组 f16 尺度」，逐张量相对 L2 误差
**12%~17%**（RTN 的固有代价，绝不是转换错误：与源对照 cosine 0.993）。
这层损失是真实的、也是后面 12 t/s 之外质量还有余量的原因（现在能对话，
但 top-1 有时偏，例如 "The capital of France is" 排序是 `...` > ` Paris`）。

## 3. 其余 6 个 bug（都在 src/ 与 kernels/）

| # | 位置 | 症状 | 修法 |
|---|---|---|---|
| 3 | `kernels/gemv_w4a4_core.h` | 权重尺度读越过行尾（K 不是 STEP 整数倍时），末行读到 Inf/NaN → **整个 5119 维变 NaN**，之后每层的 mixer 输出恒为 0（残差流冻住） | 尺度读也加 `ok` 谓词 |
| 4 | `src/model.cpp` `k_gather_heads` | q_proj 每头是 `[q(256), gate(256)]`，头间距应是 2D，代码里按 D 取 → **各头的 q/gate 交叉错位** | 新增 `head_stride` 参数，传 `2*D` |
| 5 | `kernels/flash_attn_core.h` | KV cache 的行距是**分配容量 max_ctx**，内核按 `n_kv` 算头偏移 → GQA 的 vh≥1 全读错 | 内核加 `kv_cap` 参数 |
| 6 | `kernels/flash_attn_core.h` | K 的**尺度**数组同样按 `n_kv` 索引 → 只有第一个 128 维组尺度对 | 索引改用 `kv_cap` |
| 7 | `src/model.cpp` Q 量化 | Q 的尺度数组按 `T` 行写、内核按 `TP`（补齐到 64）行读 → T 不是 64 倍数时整段错位 | 量化按 `TP` 行做 |
| 8 | `src/k_new.hip` `kv_append_v_k` | V cache 暂存是 `[H][64][D]`，T>64 时越界 → **GPU VM fault**（100 token 预填充必崩） | 按 tile 分段循环 |
| 9 | `src/model.cpp` `k_scatter_heads` | 注意力输出是 `[head][TP][D]`，回填时行距用了 `T` | 用 `src_pad` |

## 4. 激活精度：int4 → int8（W4A8）

`tools/ref_stages.py` 配合当时的量化消融脚本（临时脚本，未随仓库保留）跑出的结果
（第 0 层，相对 W4A16 参考）：

| 激活量化 | 每层相对误差 |
|---|---|
| int4，每 32 一组 | **11.8%** |
| int4，每 128 一组 | **15.9%** |
| int8，每 128 一组 | **1.4%** |

int4 激活在 64 层上会彻底糊掉，所以运行时改成 **W4A8**：

* 解码（M≤4）：`k_gemv_w4a8`（`v_dot4_i32_i8`，权重只读一遍，激活 8bit 不额外花带宽）；
* 预填充（M>4）：把 8bit 激活**精确拆成两个 int4 数字** `q8 = 16*h + l`，
  用同一个 int4×int4 GEMM 内核跑两遍再相加（`k_quant_rows_a8` /
  `k_gemm_i4_a8`），代价是 2 倍 dot8 与 2 倍权重读，换来 8bit 的精度。
* `RT_ACT4=1` 可以退回纯 int4（只用于性能对照）。
* 实测预填充因此从 ~550 掉到 ~273 t/s，**这是质量换来的，后面再优化**。

## 5. 调试工具（这一轮新加的，下次直接用）

| 工具 | 用途 |
|---|---|
| `tools/verify_plain.py` | **必跑**：逐张量校验 RT4 里所有非量化(f16/f32)张量与源一致 |
| `tools/ref_full.py --src <safetensors>` | 全模型 torch 参考，权重可来自**源 safetensors**或 RT4，逐层流式（7GB 内存也能跑 64 层，约 9 分钟） |
| `tools/ref_stages.py ... --src <safetensors>` | 单层逐阶段对照，参考端可选源权重（发现转换类 bug 的唯一办法） |
| 运行时 `--dump-layers a,b,c` + `RT_DUMP_BUF=1` | dump 每层 x 与层内中间量（tag=1000+层号*100+阶段号） |
| 运行时 `--engine` | stdin/stdout 行协议（RESET/PREFILL/GEN/QUIT），给 serve.py 当模型进程 |
| `tools/attn_check.py` + `RT_DUMP_ATTN=<层号>` | 把 FA 的 Q/K/V 打包原样导出，逐元素核对 |
| `tools/tok.py` | 分词 + chat 模板（`encode`/`decode`/`chat`/`serve`） |

## 6. 还没做的（按价值排序）

1. ~~**解码提速**：现在 83 ms/token（12 t/s）。~~ **已完成**：单行解码注意力（不再按 64 行补齐）、
   采样器去全排序、MTP3 投机上线后，解码 25.5 → **48.8 t/s**（约 20.5 ms/token，
   短上下文贪心）。实测过程见 [PERF-DECODE.md](PERF-DECODE.md) 与 [MTP.md](MTP.md)。
   现在的头号开销变成校验批（M=4）的 GEMV 与逐行解码注意力。
2. **权重精度**：int4/128 的 12% 误差还在（top-1 偶尔错）。可选
   (a) 每 32/16 组 + 非对称(scale+min) 量化；(b) 对源里本就是 FP8 的张量直接存 int8；
   (c) 用 `tools/ref_full.py --src` 量化消融先定方案，再改转换器与内核。
3. **注意力精度**：FA 里 P 现在是 4bit（`p*7` 取整），消融显示它单独就贡献 41% 的
   注意力输出误差（Q/K/V 各 7~11%）。把 P 也换成 8bit（同样用 `16h+l` 两遍）收益最大。
4. **MTP 链**、分页 KV、128k 上下文（现在服务默认 32768，`--ctx` 可调）。
5. `scripts/test_tools.sh` 的合成用例要补 **BF16→f32** 这一路（这次就是它没覆盖到）。
