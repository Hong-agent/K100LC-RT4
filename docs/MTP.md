# MTP3 投机解码：实现、等价性与实测

目标：在自研运行时里打开模型自带的 1 层 MTP（Qwen3.5 NextN），贪心解码下
一次草拟 3 个 token，主模型只读一遍权重完成校验。

权重：`models/Qwen3.8-27B-NVFP4/rt4/qwen38_27b_mtp.rt4`（15 张量 / 0.22GB）。
运行时代码：`src/model.cpp` 的 `Mtp` / `mtp_layer_rows` / `mtp_extend_context` /
`rollback_state`，引擎 `GEN` 的 MTP 分支。

> **收益取决于草稿接受率，而接受率随 prompt 剧烈变化**（交付前实测：中文短 prompt
> 21% → 25.2 t/s，英文历史文本 43% → 34.1 t/s，英文 filler 100% → 57.0 t/s）。
> 本文件后面引用的 2.37 token/轮、48.8 t/s 是当时那个长 prompt 的结果，不是通用承诺；
> 不接受投机时解码稳定在 25~26 t/s。完整对照表见 `REPRODUCE.md` §8。

## 1. 语义

MTP 层与主模型的一层 full attention 同构：

```
[norm(embedding(token_k)) | norm(hidden_k)] → mtp.fc
  → RMSNorm → q/k/v + q/k norm + partial RoPE → softmax attention
  → * sigmoid(gate) → o_proj → + residual
  → RMSNorm → gate/up/down MLP → + residual
  → mtp.norm → 共享主模型 lm_head
```

关键是 `hidden_k` 的来源：

* 第一层深度：主模型在位置 `k` 的**最终 norm 隐藏态**（`d.h_norm`）。
* 后续深度：上一草稿步的 MTP 输出隐藏态（链式）。
* MTP 自己的 KV 也必须维护：预填充时按「token 错一位」写入
  `(token_{k+1}, hidden_k)`，位置 `k`；生成时草稿每一步追加一条。
* 全接受时，最后一个草稿 token 还要补一个 MTP 条目，使
  `mtp_len == seq_len - 1`。

## 2. 一轮投机

当前状态：主模型缓存 `seq_len=L`，`lg` 是位置 `L-1` 的 logits，
`h_prev` 是位置 `L-1` 的最终 norm 隐藏态。

1. `t0 = sample(lg)`（bonus token，必定接受）。
2. 用 `(t0, h_prev)` 跑 MTP，得到 `d1`；再用 `(d1, MTP hidden)` 得到 `d2`/`d3`。
3. 主模型一次前向 `[t0,d1,d2,d3]`，得到 4 行 logits；
   逐行比较 `argmax(row_i)` 与 `d_{i+1}`，得到接受前缀 `acc`。
4. 状态回滚到接受前缀，输出 `t0` 和接受的草稿；下一轮的 `lg` 取第 `acc` 行。

## 3. 状态回滚（最容易错的部分）

主模型有两种状态：

* full attention 的 K/V cache：只需要把 `seq_len` 截短；后续写入会覆盖被拒绝位置。
* 48 层 GDN 的递推状态 + 卷积状态：不可逆，必须保存快照。

验证批前向时（`forward(..., snap_states=true)`）：

* `k_gdn_snap` 把每个 token 之后的 `[Hv][D][D]` 递推状态写入
  `[4][Hv][D][D]`。
* `k_conv_state_update_snap` 把每个 token 之后的 `[K-1][C]` 卷积窗口写入
  `[4][K-1][C]`。
* V cache 的尺度是按 64-key tile 共享的，跨 tile 回滚会读到新 tile 的 stage；
  因此主模型和 MTP 的 V-stage 也逐 token 快照，回滚时一起恢复。

回滚 `keep` 个候选（`keep = 1 + 接受草稿数`）时，取第 `keep-1` 行快照恢复。

## 4. 三个已修的等价性坑

1. **验证批不能用预填充 FA。**
   预填充 `fa_int4` 的 P 是 4bit，单 token 解码是 fp32 P；两者不是同一算术。
   现在 `snap_mode && T<=4` 时逐行走解码内核，保证校验 logits 与普通解码一致。
   后续若要提速，需要一个真正 fp32 P 的小 batch 注意力，而不是复用预填充内核。
2. **V tile 尺度不能跨 64 回滚。**
   已在主模型 / MTP 的 V-stage 上做逐 token 快照。
3. **卷积状态快照起点差一位。**
   处理完第 `t` 个 token 后，`state[r]` 的窗口起点是 `t-(K-2)+r`（不是 `t-(K-1)+r`）。

## 5. 实测

命令（1649 token 预填充 + 64 token 贪心解码）：

```bash
bash scripts/dsh.sh 'python3 scripts/bench_rt.py'
```

| 项 | 不开 MTP | MTP3 | 变化 |
|---|---|---|---|
| 预填充 @12k | 423.9 t/s | 417.9 t/s | -1.4% |
| 解码 | 25.5 t/s | **48.8 t/s** | **1.91×** |
| MTP 接受 | — | 78.9%（2.368 token/轮） | — |

预填充的上下文衰减：不开 MTP 时 1k/12k/32k = 431.5/423.9/409.5 t/s
（12k -1.8%，32k -5.1%）；开 MTP3 后 12k 为 417.9 t/s。

短提示（5 token prompt）接受率会低一些（约 45%~75%），但仍不慢于普通解码；
请求级可用 `"mtp":0` 关掉投机。

正确性回归：

* 同一 prompt、同一 seed 的贪心输出与 `--no-mtp` 逐 token 一致（短/长提示、32/64 token）。
* 引擎调试命令 `ROLLTEST2 <prompt> <4tok> <probe>`：对 keep=1..4 比较
  「验证批+回滚」与逐 token 前向的 logits，全部 max diff = 0。
  一键复跑：`bash scripts/dsh.sh 'python3 tools/mtp_rolltest.py'`。
* `tools/verify_plain.py` 449/449 一致；`scripts/test_tools.sh` 通过。

## 6. 校验批 GEMV 的第一轮提速

W4A8 的 M=3/4 校验批原来一个 warp 只处理 1 行权重，激活加载占了大头。把 `gemv_w4a8`
的每 warp 行数做成模板参数 `NROW`，M≥2 用 `NROW=2`，两行共享同一份激活加载：

| M | NROW=1 | NROW=2 | 变化 |
|---|---|---|---|
| 2 | 391 GB/s | 418 GB/s | +7% |
| 3 | 322 GB/s | **379 GB/s** | +18% |
| 4 | 260 GB/s | **317 GB/s** | +22% |

NROW≥3 寄存器溢出，掉到 ~20 GB/s。M=1 继续 NROW=1（NROW=2 在单 token 解码更慢）。
另外测了 M=5/6：NROW=2 只有 266/235 GB/s，所以 K=4 的校验不划算，默认维持 K=3。

## 7. 下一步

* M=4 的激活尝试共享内存暂存，看看能否进一步压住 L1 流量。
* 自适应草稿数：用 MTP top-1 概率/历史接受率动态调 K。
* KV int4 后重测 128k 下的 MTP 收益。

## 8. 校验批注意力：一次算完 T 行（2026-09-27）

上一节第 1 条已经做完。校验批原来把 T≤4 行**逐行**喂给单行解码内核，于是整段 KV
被读 T 遍，而且 T 次启动在长上下文下是串行延迟。现在 `src/k_fa.hip` 的
`fa_decode_rows_k` 把「行」放进 `grid.z`，一次启动算完 T 行：每行仍由**同一个
device 函数**（`fa_decode_body`）按同一套 split 规则计算，所以逐行结果与单行内核
逐位相同，只是并行而不是串行。启动点见 `src/model.cpp` 的 `snap_mode && T <= 4`
分支（`RT_VERIFY_SEQ=1` 可以退回逐行版本做 A/B）。

### 8.1 两个必须小心的点

1. **V 的 tile 尺度是「按本行 prefix」定的**。V cache 每 64 个 key 共享一个尺度，
   而尺度取的是 prefix 内 `amax/7`；同一个 tile 在第 0 行眼里（prefix 短）和第 3 行
   眼里（prefix 长）量化结果不同。所以「先全部 append、再统一算注意力」会让前面几行
   读到按后面几行的 prefix 量化的 V —— 这就是一开始 rolltest 出现 max≈3.7 的原因。
   修法：内核多收一个 `vstage_snap`（每行 append 之后的 fp32 V tile 暂存，本来就有，
   回滚要用），对**本行的最后一格**用它按本行 prefix 重量化一次，其余格子读打包缓存
   （那些格子早就是完整状态，与逐行路径一致）。
2. **split-KV 暂存的按行切片要乘头数**。第一次实现里行距写成 `sc_stride * HD`，
   少了 `n_heads`，相邻两行的暂存互相踩：表现是结果错乱且**每次运行都不一样**
   （实测同一份输入两次运行 1.7 万个元素不同）。这类 bug 用「看 logits 差多少」很难
   定位，靠的是算子级自检：`bash scripts/run_fa_rows.sh`（合成输入，逐行 vs 一次算完
   逐位比较 + 同输入重复运行）。

### 8.2 实测（单卡，MTP3，贪心）

`scripts/dec_bench_ctx.py <ctx> 32`，每档跑两次取代表值（同一台空载机器）：

| 上下文 | 逐行（`RT_VERIFY_SEQ=1`） | 一次算完 | 变化 |
|---|---|---|---|
| 1k | 56.3 / 55.5 t/s | 57.4 / 58.4 t/s | +2.5% |
| 4k | 54.8 / 53.6 t/s | 55.6 / 56.2 t/s | +2.3% |
| 8k | 50.8 / 49.1 t/s | 52.6 / 49.3 t/s | +1.5% |
| 32k | 37.0 t/s | **38.1 t/s** | +3.0% |

收益比预期小：解码一轮里 `flash-attn` 只占 ~13%（`linear` 才是 50%），而 32k 上下文
下整个 KV 也只有 512MB/层组，4 行同时读同一片 KV 会在 L2 命中，省下的 DRAM 流量
本就有限。真正的意义在 128k：那时 KV 流量按 4 倍增长，而权重是常数。

**等价性照旧**：`tools/mtp_rolltest.py` 四个 keep 全部 `max = 0`；同一 prompt 的
贪心输出与不开 MTP 仍然逐 token 一致（64 token 全等）。

### 试过但更慢的：校验批 argmax 搬到设备侧

校验批每轮要拿 (K+1) 行 logits 的 argmax 来比对草稿。现在是在主机做：回读
(K+1)×1MB logits + CPU 扫全词表，`RT_PROF=1` 实测 **0.82 ms/轮**（`logits-copy`）。
改成设备侧逐行 `k_argmax`（只回传 4 个 int）后反而变成 **2.30 ms/轮**（`sample`）——
`k_argmax` 是单 block 归约，一个 block 读 1MB 是**延迟受限**的（256 线程 × 970 次
串行加载），跑 4 次就是 4 倍的裸延迟。要赢得多 block 两级归约（`row × 32 block`
再合并），收益上限也只有 ~0.7 ms/轮（约 1%），暂不做。
