# RT4 运行格式与自研框架设计（K100_LC / gfx926）

目标模型：**`unsloth/Qwen3.8-27B-NVFP4`**（镜像自魔搭官方同名仓库，落在 `models/Qwen3.8-27B-NVFP4/`）。
硬指标（用户指定）：**128k 上下文；填充 1000 token；开 MTP3；输出 100 token**，
即验收跑法等价于 `llama-bench -d 131072 -p 1000 -n 100` 再加 MTP 3 草稿，
两个吞吐指标分别按 1000 t/s（预填充）与 100 t/s（吐字）考核。
约束：不借用其他推理框架（vLLM / SGLang / llama.cpp），自研运行框架 + 自研算子。

---

## 1. 硬件通路（本机实测，见 `DESIGN-INT4.md` / `bench/bench_isa2.cpp`）

| 通路 | 实测峰值 | 折算 1000 t/s 预填充需要的 52 GMAC/token |
|---|---|---|
| `v_pk_fma_f16`（打包 fp16 FMA） | 33.7 TFLOPS = 16.9 TMAC/s | 不可能（差 3 倍） |
| `v_dot2_f32_f16` | 16.9 TMAC/s | 不可能 |
| `v_dot4_i32_i8`（int8 点积） | 37.9 TMAC/s | 137%，不可能 |
| **`v_dot8_i32_i4`（int4 点积）** | **75.5 TMAC/s** | **69%，唯一可行通路** |

MFMA / WMMA 在 gfx926 上不存在（汇编器全部拒绝）。显存可用带宽实测 **804 GB/s**（理论 896）。
结论：**权重必须统一 int4、激活也要 int4（W4A4）才有余量**——这决定了下面的 RT4 格式。

## 2. 源权重结构（读自 22.57GB safetensors 头部 + config.json）

文本部分 26.9B 参数、21.65GB：

* `embed_tokens.weight` BF16 `[248320,5120]` = 2.54GB（源里未量化）
* `lm_head.weight` FP8 `[248320,5120]` + 逐通道 BF16 scale = 1.27GB
* 56 层 MLP：NVFP4 `[17408,2560]`(U8 打包) + `[17408,320]` fp8 块尺度(每 16) + 全局/输入 scale
  = 7.49GB（14.97B 参数）
* 8 层 MLP（第 56~63 层）：FP8 = 2.14GB
* 48 层线性注意力（gated delta net）：`in_proj_qkv[10240,5120]`、`in_proj_z[6144,5120]`、
  `out_proj[6144,5120]` FP8 = 5.56GB，另加 f32 的 `A_log`/`dt_bias`/`conv1d`
* 16 层全注意力（每 4 层一个）：`q_proj[12288,5120]`（含输出门，24 头×256×2）、
  `k,v[1024,5120]`、`o_proj[6144,5120]` FP8 = 1.68GB
* `model.visual.*` 0.92GB（视觉塔）——单独转成 RT4（线性层 f16，
  norm/bias/pos f32），由运行时自己的 27 层视觉块加载执行
* 结构要点：64 层，`linear:full = 3:1`；MRoPE `partial_rotary_factor=0.25`（只旋转 64/256 维）；
  `attn_output_gate=true`；`mtp_num_hidden_layers=1`；`kv_cache_scheme` 为 8bit（自带 k_scale/v_scale）

## 3. 源精度 → RT4 的换算

源里两套量化共存，反量化公式（已按此实现）：

```
NVFP4 : w = e2m1(code) * (fp8_e4m3(block_scale) * weight_global_scale)   # 块大小 16
FP8   : w = fp8_e4m3(code) * bf16(channel_scale)                         # 逐输出通道
```

统一重量化成 **int4（每 128 个 K 一组，f16 尺度）**：

```
s = amax(|w|) / 7 ;  q = clamp(round(w / s), -8, 7) ;  w' = q * s
```

为什么必须这么做：源的 NVFP4(E2M1) 只能走 int8 通路（37.9 TMAC/s），1000 t/s 需要它的 137%
——**源精度在本卡上到不了目标**；统一 int4 才有 69% 的余量。代价是二次量化误差，
逐张量实测 `relerr` 会写进 manifest，并用 BF16 原模型（GGUF；不是本仓库内容，
`RT_BF16_GGUF_DIR` 指向它即可）逐张量对照 `src_rms` 复核反量化公式没搞错。

## 4. RT4 文件格式（`tools/convert.c` 产出）

一个权重文件 + 一个 JSON manifest；所有张量 256B 对齐，可 `mmap` 后按偏移异步拷到显存。

```
2D 线性权重 [N, K]，kind="i4"：
  q 区: N * (K/2) 字节   有符号 int4，沿 K 两个一组打包，**低半字节 = 偶数 k**
  s 区: N * (K/128) * 2  f16 尺度，w = q * s
kind="i8"：q 区 N*K 字节（int8），尺度同上
kind="f16"/"f32"：逐元素原样
```

manifest 条目：`name, kind, shape[N,K], group, q_off, s_off, nbytes, relerr, src_rms`。

精度策略（`convert.c` 默认）：

* MLP / 注意力 / 线性注意力投影 / embed：**i4**（group 128）
* `lm_head`：默认 **i4**，`--int8-lmhead` 可切 int8（+0.64GB，护输出质量，随时可切）
* `*norm.weight` / `A_log` / `dt_bias` / `conv1d.weight` / bias：**f32**（SSM 里参与 exp，半精度会掉点）
* `in_proj_a/b`：f16

预期体积：`26.9B/2 + 26.9B/128*2 ≈ 13.9 GB`（源文本权重 21.6GB）。

## 5. 运行期激活量化（自研）

| 阶段 | 激活格式 | 指令 | 理由 |
|---|---|---|---|
| 解码（M=1..4，含 MTP 校验批） | 逐 token int8，每 32/64 一组 + 逐 token scale | `v_dot4_i32_i8` | 解码是带宽瓶颈不是算力瓶颈，int8 更省事更准 |
| 预填充（M 大） | 逐 token int4，每 16/32 一组 + scale，必要时加 Hadamard 旋转 | `v_dot8_i32_i4` | 只有它在 128k 上撑得到 1000 t/s |

KV cache：按 **q8 + 逐(层, kv 头, 通道) scale** 实现（源模型自带 8bit KV 尺度）；
128k 全量 KV = 16 层 × 4 kv 头 × 256 维 × 2(KV) × 131072 = **4.30 GB**（int4 KV 则 2.15GB）。
SSM 状态：48 层 ×（卷积状态 + delta-net 递归状态）≈ 150MB，与上下文长度无关。

## 6. 每步带宽预算（决定能不能到 100 t/s）

解码一步要读的量（MTP3：一步校验最多 4 个 token，权重只读一遍）：

| 项 | int4 权重 + q8 KV | 备注 |
|---|---|---|
| 主模型权重 | 13.87 GB | 全部线性层 |
| KV cache @128k | 4.30 GB | 16 个全注意力层 |
| MTP 头 ×3 步 | 2.55 GB | 0.85GB/步 ×3（若也 int4 则 1.29GB） |
| **合计** | **20.7 GB** | 804 GB/s ⇒ **25.8 ms/步 ⇒ 38.8 步/s** |

要 100 t/s 需要 **2.6 个 token/步**（4 个候选平均接受 2.6 个，接受率 ≈0.7）。
把 KV 与 MTP 头都压到 int4 后降到 17.3GB/步（21.5ms ⇒ 46.5 步/s ⇒ 只需 2.15 token/步，接受率 ≈0.55）。
**这是 MTP3 路线的关键杠杆。**

## 7. 预填充预算（128k 深度）

| 部分 | GMAC/token | 说明 |
|---|---|---|
| MLP（64 层） | 17.1 | 大头，走 int4 |
| 线性注意力投影（48 层） | 5.6 | 走 int4 |
| 全注意力投影（16 层） | 1.7 | 走 int4 |
| lm_head | 1.3 | |
| **注意力打分/PV @128k** | **25.8** | 16 层 × 131072 键 × 6144 维 × 2 |
| **合计** | **52.1** | 1000 t/s ⇒ **52 TMAC/s = dot8 上限的 69%** |

两条硬结论：① 128k 下注意力与线性层各占一半，**QK^T / PV 也必须量化到 int4/int8**
（fp16 通路只有 16.9 TMAC/s，独占都不够）；② 预填充 GEMM 至少要跑到 dot8 上限的 69%
（1000 t/s × 52 GMAC/token = 52 TMAC/s）。

**（00:30 实测补充）用现有 GEMM 原型量了两种形状，结果把风险点暴露出来了：**

| 形状 | 用途 | 实测 | 占 dot8 上限 |
|---|---|---|---|
| M=1000 N=17408 K=5120 | FFN / 线性层 | **48.60 TMAC/s** | 64.4% ✓ |
| M=1000 N=131072 K=256 | **128k 注意力打分（每个头 K=256）** | **18.66 TMAC/s** | **24.7%** ✗ |

按这两个数字折算 1000 token 的预填充：
线性 26 TMAC / 48.6 = 0.54s，注意力 25.8 TMAC / 18.66 = 1.38s，合计 **1.93s ⇒ 519 t/s**。
**也就是说：直接用通用 GEMM 打注意力，预填充只能到 ~520 t/s。**
原因是 K=256 的窄 K 形状让分块流水完全摊不开（每行只有 32 个 dword）。
所以预填充的关键路径不是"再压 GEMM 那 5 个点"，而是**写一个真正的 int4 FlashAttention**：
把 head_dim 256 整个放进内层、Q 块常驻寄存器/共享内存、外层沿 131072 个 key 长流水，
这样才可能把注意力拉到 35~45 TMAC/s（46~60%），1000 t/s 才有戏。

**（2026-09-27 晚更新）int4 FlashAttention 已经写出来并实测**：
`kernels/flash_attn_int4.hip` 在同一个形状上 **25.49 TMAC/s（33.8%）**，
折算 1000 token 的预填充 = 注意力 1.01s + 线性 0.53s = **1.54s ⇒ 约 650 t/s**
（老路线 521 t/s）。**没有到 35~45 那一档**，差距的性质已经定位：
dot8 只占每线程指令槽的 ~50%（软最大那 ~150 条是大头），且 245 VGPR 只够
4 wave/CU（O 与 PV 的累加器各占 64 个 VGPR）。完整消融与后续路径见
[FLASH-ATTN.md](FLASH-ATTN.md)。

## 8. 算子清单（全部针对 gfx926 手写）

| 算子 | 状态 | 说明 |
|---|---|---|
| int4×int4 GEMM（大 M，预填充） | **完成**（原型 61%；QG=512 版 69%） | 内核 `kernels/gemm_core.h`，基准 `bench/int4_gemm2.cpp` |
| W4A16 GEMM（对照用） | 完成但只有上限 28% | `bench/w4a16_gemm.cpp` |
| int4/int8 GEMV（M=1..4，解码 + MTP 校验） | **完成**（现役 W4A8；实测 M=1 566 / M=4 390 GB/s，上限 620/692） | `kernels/gemv_w4a8_core.h` + `gemv_int4.hip`，见 GEMV.md |
| int4 FlashAttention（预填充，128k） | **完成**（25.49 TMAC/s，33.8%） | `kernels/flash_attn_int4.hip`；QK^T/PV 都走 dot8，见 FLASH-ATTN.md |
| q8/int4 解码注意力 | 单行 + split-KV **完成**；分页 / int4 KV 未做 | `src/k_fa.hip`（`fa_decode_k`） |
| gated delta net 扫描 | **完成**（递推 + 逐 token 快照回滚）；预填充未块化 | `src/k_new.hip` 的 `gdn_k2`（旧版留作 A/B：`RT_GDN_OLD=1`） |
| RMSNorm / MRoPE(0.25) / SiLU / 残差 | 独立 kernel **完成**；融合进 GEMM 流水未做 | `src/k_new.hip`；融合点见 PERF-DECODE.md §5 |
| MTP 头（1 层）+ 投机循环 | **完成** | 复用主模型算子；权重取 `rt4/qwen38_27b_mtp.rt4`，贪心接受率 2.37 token/轮 |
| 采样器 / 分词器 / 服务层 | **完成** | `src/model.cpp` 的 Sampler + `scripts/serve.py` + `web/index.html`（batch=1 单流） |

## 9. 现状与结构

本节的旧版曾在这里复制一份「项目结构 + 里程碑表」，与 README/RESUME 各维护一份必然飘，
已删除。结构见 [README.md](../README.md) 的目录树，最新实测与下一步见
[RESUME.md](../RESUME.md)、[PERF-DECODE.md](PERF-DECODE.md)、[MTP.md](MTP.md)。

已落地：`src/`（`model.cpp` 64 层图执行 + MTP3 投机 + 引擎协议 / `k_new.hip` /
`k_gemv.hip` / `k_gemm.hip` / `k_fa.hip`）、`scripts/serve.py` + `web/index.html`
的服务层、`scripts/make_dist.sh` 的离线包生成。
