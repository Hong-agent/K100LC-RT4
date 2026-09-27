# 解码提速：12.7 → 26.2 t/s（2026-09-28）

本轮只做一件事：**把「能对话」变成「能顺畅对话」**。用户给的下一步里，解码提速排第一
（预填充与吐字质量都已达标，但 12 t/s 的吐字速度在长回答上很折磨）。

同口径实测（`scripts/bench_rt.py`，单卡，1.65k 预填充 + 64 token 解码）：

| 项 | 本轮开始 | 现在 | 倍数 |
|---|---|---|---|
| 预填充 1649 tok | 280.7 t/s | **419.6 t/s** | 1.50× |
| 预填充 417 tok | 259.9 t/s | 375.6 t/s | 1.45× |
| 解码（贪心） | 12.7 t/s（78.9 ms/token） | **26.2 t/s（38.2 ms/token）** | 2.06× |
| 解码（temp=0.7） | 10.1 t/s（99.0 ms/token） | **25.2 t/s（39.7 ms/token）** | 2.50× |

上下文变长时的解码（`scripts/dec_bench_ctx.py`，本轮的注意力改动在这里收益最大）：

| 上下文 | 本轮开始 | 现在 |
|---|---|---|
| 60 | 38.6 ms/token | 36.5 |
| 1000 | 41.2 | 36.2 |
| 4000 | 48.3 | 36.3 |
| 8000 | — | 37.3 |

即：**解码速率从「随上下文线性恶化」变成基本持平**（原来 4k 上下文要多花 9.7 ms/token，
按那个斜率到服务默认的 32k 上下文会掉到 ~11 t/s）。

---

## 1. 先修量测：分阶段计时怎么做才不骗自己

`RT_PROF=1` 给运行时加了分阶段计时（`src/model.cpp` 顶部的 `ProfTick`）：每个作用域
进入/退出各记一个 `hipEvent`，**中途不同步**，最后只同步一次，再按相邻事件的时间戳之差
把时间归到当时最内层的作用域。中途同步会抽干流水线，把小内核放大好几倍。

但这里还有个更隐蔽的坑，值得写下来：**`hipEventRecord` 本身要花 GPU 时间**。
`bench/launch_probe.cpp` 实测：

| 探针 | 每次耗时 |
|---|---|
| 1000 × 空内核（20 块 × 256 线程） | 2.24 µs |
| 1000 × 小内核（5120 个元素） | 3.26 µs |
| 1000 × 同一个内核写同一地址（串行依赖） | 3.25 µs |
| 1000 × 小内核 + 每个作用域 2 个 `hipEventRecord` | **17.43 µs** |

也就是**每个作用域要多算 ~14 µs**。解码每 token 有约 1000 个作用域，分阶段表总共被抬高
约 10 ms（实测：开 `RT_PROF` 后整步 47.8 ms，关掉是 38.2 ms）。所以本文所有结论都
「先扣掉 作用域数 × 14 µs」，并且最终判断一律用**墙钟**（`scripts/dec_bench.py`）。

另一个探针结论同样重要：**内核启动本身只要 2~3 µs**（不管有没有依赖）。所以
「启动次数太多」不是主要矛盾，「每个作用域要等前一个内核的尾巴 + 访存延迟」才是。

## 2. 轮廓：时间到底在哪

（下表是 `RT_PROF=1` 原始值，未扣除事件开销；右侧是扣掉后的大致真实值。）

| 类目 | 原始 ms/token | 作用域数 | 真实 ms/token |
|---|---|---|---|
| linear（权重 GEMV + 激活量化） | 28.0 | 400 | ~22.4 |
| norm/elem（rmsnorm/silu/gather/rope…） | 5.3 | ~200 | ~2.5 |
| kv-quant（Q 量化） | 1.4 | 16 | ~1.2 |
| gdn-ssm（a/b 投影 + 门控） | 1.4 | 48 | ~0.7 |
| gdn-recur（Δ 递推） | 1.9 | 48 | ~1.2 |
| gdn-norm（split/l2norm/门控 rmsnorm） | 1.6 | 96 | ~0.3 |
| gdn-conv（因果卷积 + 状态） | 1.1 | 96 | ~0.2 |
| head（最终 norm + lm_head） | 0.9 | 2 | ~0.9 |
| flash-attn | 0.5 | 16 | ~0.3 |
| logits-copy / sample / embed | 0.7 | 25 | ~0.7 |
| **合计** | **43.7** | | **~30（墙钟 38.2）** |

结论：**权重 GEMV 是唯一的大头**。每 token 必须读 13.22 GB 权重（13.86 GB 的 int4
权重里，embedding 表 636 MB 只按需 gather），实测 ~500 GB/s，而 M=1 形状的纯读探针
上限是 620 GB/s（见 `docs/GEMV.md`）——即 GEMV 已经用掉 80% 的可用带宽。
剩下 10~12 ms 全是「小内核的延迟 + 启动」，没有单项超过 3 ms。

## 3. 改了什么（按收益排序）

### 3.1 `k_gemv_f32` 每个 block 只有 1 个线程（-17 ms/token）

`ssm_alpha` / `ssm_beta` 两个 `[48][5120]` 的 f32 权重走的是
`gemv_f32_k<<<dim3(N, M), 1>>>` —— **一个 block 一个线程**，解码时整卡只有 96 个活跃
lane，每层 0.4 ms。改成「一个 block 管一行输出、128 线程跨 K 步进 + 块内归约」。

> 后续：这两个 f32 GEMV 连同 `sigmoid` / `ssm_gate` 一共 4 个内核已被 §3.2 的
> `k_ssm_ab_gate` 完全取代，`k_gemv_f32` / `k_sigmoid_inplace` / `k_ssm_gate`
> 已从 `src/k_new.hip` 删除。

### 3.2 SSM 的 4 个内核合成 1 个（gdn-ssm 19.07 → 1.39 ms）

`a = W_a·x`、`b = W_b·x`、`beta = sigmoid(b)`、`g = -exp(A_log)·softplus(a+dt_bias)`
原来是 4 次启动 × 48 层。合成 `k_ssm_ab_gate`：grid = (Hv, T)，一个 block 管一个
(head, token)，一次读进 a/b 两行的权重。新增 `d.gbeta` 保存 beta，**保持原来 `db()`
阶段 7/8/9 的 dump 语义不变**（否则逐层对照工具会对不上）。

### 3.3 注意力 Q 量化：24 次小启动 × 64 行补齐（-10 ms/token）

原来每个头调一次 `k_quant_rows`：解码时 T=1，却按 TP=64 行量化，而且是
「64 个 block × 2 个线程」的小启动，一层 24 次，实测 0.72 ms/层。
新增 `k_attn_q_quant`：一次启动、`grid = ceil(H*T*(D/G)/128)`，**只量化真实的 T 行**。
补齐行的 Q 保持 `reset_state()` 给的 0 数据 / 1 尺度（FA 逐行独立，补齐行的输出会被
`k_scatter_heads` 丢掉，所以安全且不再白算）。

### 3.4 每层一次 `hipMalloc` + 阻塞 H2D 拷贝 + `hipFree`（-7 ms/token）

`attention_layer` 里为了 RoPE 的位置，每层都：`hipMalloc` 一个 int 数组 → 同步
`hipMemcpy` 上传 → `hipFree`。这是**每层一次设备级同步**，16 层/token。
改成 `k_rope(..., pos=nullptr, pos0=seq_len, ...)`，位置在核内算；`forward()` 里的
token id 上传也改成常驻缓冲 + `hipMemcpyAsync`。attn-pre 一类的小项因此掉到 1/3。

### 3.5 采样器对 248320 个 logit 做全排序（temp>0 时 -20 ms/token）

原来 `top_k=0, top_p=1` 时会对全词表 `std::sort`（实测 temp=0.7 比贪心慢 20 ms）。
现在：无截断 → **两遍 O(V) 精确采样，完全不排序**；有截断 → `nth_element` 选前 K
（K = top_k 或 2048）再排序。temp=0.7 与贪心的差距从 25% 降到 3%。

### 3.6 GDN 递推：状态布局错了，访存完全不合并（gdn-recur 4.42 → 1.85 ms）

状态布局是 `[h][col][i]`（i 连续），而 kernel 是「一线程一列」：相邻线程的地址相差
D=128 个 float，**一个 warp 的每次 load 要碰 32 条 cache line，只用 4/32 字节**；
同时每线程 128 个活跃寄存器、96 个「64 线程」的块 → 卡上几乎没占用率。

改成 `[h][i][col]`（列连续）+ 把 i 切成 4 段（`GDN_SPLIT=4`，一个 warp 管一段、
一个 block 管 32 列），跨段归约走共享内存。访存完全合并、每线程 32 个寄存器。
旧内核用 `RT_GDN_OLD=1` 仍可跑，方便对照。

### 3.7 解码注意力：去掉「64 行补齐」的 64 倍浪费 + split-KV（4k 上下文 -12 ms/token）

`kernels/flash_attn_core.h` 的 `fa_int4` 是给预填充写的（BM=64 行 query 一起算）。
解码时只有 1 行真实 query，却按 64 行补齐 —— 白算 64 倍，而且越长的上下文亏得越多。

新增 `src/k_fa.hip` 的 `fa_decode_k`（单行专用）：

* 一个 block 一个 head、256 线程 = head_dim 一个线程；
* **QK**：thread t 负责 key = t/4、dword 段 = (t%4)*8，4 个 lane 归约出一个 score。
  这 4 个 lane 正好分别落在两个 128 维量化组里，**各自的 scale 在归约前乘上**；
* **PV**：thread t 负责输出维度 t，按 `[key/8][D]` 布局读 V（同一时刻 32 个 lane 读
  连续地址）。P 直接用 fp32 —— 比预填充路径的 4bit P **更准**；
* **split-KV**：key 方向再切 8 段（`grid.y`），各算 (局部 max, 局部 sum, 未归一化输出)，
  再由 `fa_decode_comb_k` 按 flash-decoding 的公式合并。只切 1 段时 24 个 block 只
  用到 120 个 CU 的 1/5，是纯粹的延迟受限（实测每 1000 key 要 1.24 ms，而带宽只要
  0.02 ms）。切 8 段后基本贴到带宽。

### 3.8 调参：`A8_ROWS`（-2.5 ms/token）

`kernels/gemv_w4a8_core.h` 的「每个 warp 处理几行」：实测 1 行 38.0 ms/token、
2 行 40.5、**4 行 252.7**（4 行时一个 warp 要反复读 4 行权重，L1 完全不够用）。
默认值从 2 改成 1。

### 3.9 试过但**更慢**的：把激活搬进共享内存

直觉：每个 warp 每迭代要发 3 个 128-bit 全局读（权重 1、激活 2），4 个 warp 在重复
读同一份激活。实际把激活协作搬进 LDS 后 **41.7 vs 38.2 ms/token —— 更慢**
（多了一次块内 barrier 和共享内存占用，而那份激活本来就命中 L1/L2）。
代码留在 `gemv_w4a8<M, SMEM>` 里，`RT_SMEM=1` 才启用。

### 3.10 顺手修的两个服务层问题

* **`scripts/serve.py` 不认识 `--ctx`**：`serve.sh` 一直传 `--ctx 32768`，而
  serve.py 的 argparse 只定义了 `--port/--host` → 容器启动即退出、被
  `--restart unless-stopped` 反复拉起（`docker ps` 里是 Restarting）。现在
  `--ctx` 会透传给引擎（`chat.Engine(ctx=...)`）。
* **推理强度可调**：模型自带的 chat 模板默认按 `xhigh` 注入一段系统提示（所以回答
  前面会先输出一段思考）。现在客户端可以在请求体里带
  `{"reasoning_effort": "low"}` 让模板换成简短指令（模板只认 xhigh/medium/low）。

## 4. 正确性怎么保证（本轮的每一步都过了）

| 检查 | 命令 | 结果 |
|---|---|---|
| 权重链路（非量化张量逐张量） | `tools/verify_plain.py` | 一致 449，异常 0 |
| 转换器合成用例 | `bash scripts/test_tools.sh` | 通过 |
| 单层逐阶段（GDN 层，源权重） | `tools/ref_stages.py ... 0 --bits8` | 阶段 8/9/10 = 5e-08 / 8e-08 / 2e-05 |
| 单层逐阶段（注意力层） | `tools/ref_stages.py ... 3 --bits8` | 见 `docs/QUALITY-FIX.md` 的注意力误差（P 4bit 是主项，与本轮无关） |
| **解码注意力（短上下文）** | `scripts/attn_decode_check.py 3 11` | **1.7e-07**（内核 vs fp64 参考） |
| **解码注意力（长上下文 + split）** | `scripts/attn_decode_check.py 3 2100` | **9.8e-08** |
| 改 GDN 前后生成一致 | `chat.py` 贪心 20 token | 前 19 个 token 完全相同，第 20 个起是浮点求和顺序差异 |

**关于「改动后 logits 变了」**：同一段 1649 token 预填充，顶位 logit 从 10.63 变成
11.11（top-3 的 token 完全一样）。这不是精度退化——新旧 GDN 内核相对 torch 参考的
误差都是 **1.56e-05 / 1.57e-05**，只是 fp32 求和顺序不同（旧版一线程串行累加、
新版 4 段归约）。64 层 int4 模型对这点舍入本来就敏感（每层权重量化误差 12%~17%），
放大到 logit 上是 0.5 量级；生成文本无差别。**用 `RT_GDN_OLD=1` 可以复现对照。**

`scripts/attn_decode_check.py` 是本轮新加的独立复核：跑一次「prefill + 2 步解码」，
把解码步的 Q/K/V 与打包尺度 dump 出来，用 fp64 重算单行 softmax 注意力再逐元素对照。
（顺带修了一个老问题：`attn_out.bin` 以前是在注意力**之前** dump 的，落盘的是上一次的
陈旧值。）

## 5. 还没做的（按杠杆排序）

1. **减少每 token 的权重读取**——这是唯一能把解码再提一个数量级的方向。
   13.22 GB/token 在 620 GB/s 下就是 21.3 ms 的硬地板。要么上 **MTP/投机解码**
   （一次前向吐 2~4 个 token，权重只读一遍；`rt4/qwen38_27b_mtp.rt4` 已转好），要么
   在服务层做**并发批请求**（M 路请求共享一次权重读取）。注意投机解码要能回滚被拒绝
   token 的 KV（现在 KV 是追加式、没有回滚接口）。
2. **GEMV 再挤一点**：现在 ~500 GB/s，纯读探针 620。3.9 那条负结果说明「激活的
   重复读」不是瓶颈；可以考虑 `__ldcs`（流式读，不占 L2）、更大的 `uint4` 预取、
   或把 A8_ROWS=1 的尾部（K=5120 时 2.5 个迭代）处理得更整齐。
3. **元素级小内核合并**：norm/elem 现在 ~2.5 ms、`gdn-norm` ~0.3 ms，单项都不大，
   但「残差加法 + 下一层 rmsnorm」可以合成一个内核（每 token 省 64 次启动），
   `split_qkv + 2×l2norm` 也可以合成一个。
4. **预填充**：419 t/s，目标 1000。现在的瓶颈回到 `docs/FLASH-ATTN.md` 说的注意力
   指令密度（P 是 4bit，改 8bit 会更准也更值得做）；另外预填充的 `linear` 里
   `k_gemm_i4_a8` 要跑两遍（8bit 激活拆成两个 int4），是质量换来的 2 倍代价。
5. 128k 上下文、分页 KV、int4 KV。

## 6. 复现

```bash
cd K100LC-RT4
bash scripts/build_rt.sh                 # 编译（FLAGS="-DA8_ROWS=1" 可覆盖内核调参）
bash scripts/dsh.sh 'python3 scripts/bench_rt.py'            # 预填充 + 解码总表
bash scripts/dsh.sh 'python3 scripts/dec_bench.py 48'        # 解码墙钟（A/B 用）
bash scripts/dsh.sh 'python3 scripts/dec_bench_ctx.py 60,1000,4000,8000'   # 解码 vs 上下文
bash scripts/dsh.sh 'RT_PROF=1 python3 scripts/chat.py --prompt 你好 --n 24'  # 分阶段计时
bash scripts/dsh.sh 'python3 scripts/attn_decode_check.py 3 2100'            # 解码注意力复核
```

计时相关的环境变量：`RT_PROF=1` 开分阶段计时；`RT_ATTN_OLD=1` / `RT_GDN_OLD=1` 回退到
旧内核做对照；`RT_NSPLIT=<n>` 强制解码注意力的 split 数；`RT_SMEM=1` 打开（更慢的）
共享内存版 GEMV；`RT_NOQ=1` 跳过激活量化内核（只为计时，结果错）。
