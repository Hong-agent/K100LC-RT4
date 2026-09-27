# 接续说明（自研运行时线）

## 状态（2026-09-27）：**网页继续修：max_tokens 不再 413、停止按钮真能停** ✅

用户反馈「输入 10854 token + 最多生成 128000 token 超过上下文 131072」。根因是我上一轮
把 `max_tokens` 当成了「要预留的上下文」——客户端（含本网页）默认就发 128000，于是
任何超过 3k 的提示词都会被拒。现在 `max_tokens` 只当**生成上限**：
`clamp_max_tokens()` 按剩余上下文收窄（实际值随 `timings.max_tokens` 返回），
只有 prompt 本身就超上下文才 413。实测：9.4k / 18.8k token 提示词 + 128000 均返回
200；20 万 token 提示词仍 413（正确）。

顺手补上「停止」的真实语义（128000 的生成上限让这件事变得必须）：

* **引擎侧**（`src/model.cpp`）：GEN 循环里非阻塞看一眼 stdin，收到 `STOP` 就在
  当前轮（MTP 一轮 ~75ms）退出，打印 `END <n> <ms> stopped`；停止后引擎照常接活。
* **服务侧**（`scripts/serve.py` / `chat.py`）：客户端断开（网页按停止 / 关闭页面）时
  发 `STOP` 并等引擎收尾再释放请求锁——否则引擎会继续生成、锁被占住，后续请求要等
  几十分钟。读取改成「每请求一个专属线程 + 队列」（`Engine.gen_stream`/`GenStream`），
  避免被取消的协程把残留输出串到下一个请求。
* 实测：长请求 6 秒断开 → 引擎 `stopped` 收尾 → 紧接着的请求 0.4~1.4 s 正常返回，
  内容正确；技能路径同样验证。

网页还补了两处小修：工具轮重置时把上一轮的思考块一起清掉；状态栏提示实际生成上限
（鼠标悬停在 token 数上）。

## 状态（2026-09-27）：**网页「看不到输出」修好了：技能路径边生成边发** ✅

现象：网页发消息后长时间空白。实测定位到三个叠在一起的原因：

1. **技能路径整段缓冲**：`_complete_with_skills` 原来把整段生成跑完才建 SSE 流，
   于是页面要等全部生成结束（实测 13.6 s）才一次性出现文字。现在把工具循环做成
   事件流（`_skill_events`）：逐 token 发 `delta`/`reasoning`，某轮判定是工具调用时
   发 `round_reset`（前端丢弃该轮草稿）再发 `skill` 事件，然后进入下一轮。
   流式与非流式共用同一条路径（非流式只是把事件流收干）。
2. **新增的 `delta` 分支漏了映射**：切分器产出 `content`，外层只认 `delta`，
   正式回答的增量被丢掉、最后靠兜底按 24 字补发。修成 `content → delta` 后
   实测回答分片：技能开+关思考 67 片、技能关 39 片（首字 0.3~3.3 s）。
3. **网页不显示思考段**：默认开着思考时，正式回答要等思考结束才出现；
   现在网页把 `reasoning_content` 渲染成气泡里的灰色 `💭` 小字，并加了 `⏳` 计时，
   等待期间不再是「死的」。

顺带：`RT_DEBUG_SKILL=1 bash scripts/serve.sh` 可以打印每轮的 token 数、思考/回答
分片数、是否出现思考结束标记，方便以后定位这类问题。

复现验收（`REPRODUCE.md` 第 13 项）：技能开+关思考 67 片/首字 3.3 s；技能开+低思考
思考 100 片（首字 3.2 s）+ 回答 48 片（首字 6.5 s）；技能关 39 片/首字 0.3 s；
多轮技能（`now` → `write_file`）每轮都有 `round_reset` + `skill` 事件，最终回答
逐步流出。`tools/test_attachments.py` 仍通过。

## 状态（2026-09-27）：**MTP 校验批注意力改为「一次算完 T 行」** ✅

MTP 的验证批（T≤4 行、每行 KV 长度差 1）原来逐行调用单行解码内核：KV 读 4 遍，
而且 4 次启动在长上下文下串行。现在 `src/k_fa.hip` 的 `fa_decode_rows_k` 把行放进
`grid.z`，一次启动算完，**每行仍走同一个 device 函数、同一套 split 规则**，
所以逐行结果与单行内核逐位相同（`RT_VERIFY_SEQ=1` 可退回逐行版做 A/B）。

实测（MTP3 贪心，`scripts/dec_bench_ctx.py`）：1k/4k/8k 为 57.4/55.6/52.6 t/s，
对应 +2.3%/+2.3%/+1.5%；32k 从 37.0 → **38.1 t/s（+3.0%）**。
收益小的原因是解码一轮里注意力只占 13%、KV 在 32k 也只有 512MB/层组（4 行会命中
L2）；真正受益的是 128k。**等价性不变**：rolltest 四个 keep 全 `max = 0`，
贪心输出与不开 MTP 仍逐 token 一致。

这一轮同时补了一个算子级自检 `bash scripts/run_fa_rows.sh`（合成输入，「逐行 vs
一次算完」逐位比较 + 同输入重复运行），它当场抓到一个真 bug：split-KV 暂存的按行
切片少乘了 `n_heads`，相邻两行互相踩，表现是结果错乱**且不可重复**。详见
[docs/MTP.md](docs/MTP.md) 第 8 节。

## 状态（2026-09-27）：**网页技能 + 多轮文件工作区** ✅

网页默认开启“技能”：模型可在一次回答里连续调用 `now` / `calc` /
`write_file` / `read_file` / `list_files` / `delete_file` / `make_csv`。
每个 `conversation_id` 有独立工作区，文件在输入框上方列出，可打开预览或下载；
多轮任务会把工作区文件列表注入下一轮上下文，模型可继续读取/修改文件。
文件接口：`GET/DELETE /v1/files`。默认单文件 2MB、单工作区 64MB。

## 状态（2026-09-27）：**本地视觉塔已接入，图片能看懂** ✅

网页上传图片现在走本地 Qwen3.5 视觉塔，而且视觉塔本身也统一成 RT4：
`scripts/prepare_vision.sh` 从源 safetensors 导出 333 张 `model.visual.*` 为
独立 `qwen38_27b_vision.rt4`（线性层 f16，norm/bias/pos f32，0.93GB），27 层
视觉块在自研 C++/HIP 运行时里执行。文本运行时用新协议 `PREFILL_EMB` 把图像
embedding 覆盖到 `<|image_pad|>` 位置；Python 只保留图片预处理。

实测：上传写着 `7788` 的图片能回答 `7788`；两张 `12`/`34` 能按顺序读出；
流式、MTP3、多图都通过。设计、协议和权重导出见 **[docs/VISION.md](docs/VISION.md)**。

## 状态（2026-09-27）：**MTP3 投机解码已接入** ✅

权重同目录存在 `qwen38_27b_mtp.rt4` 时默认启用；贪心路径一次草拟 3 个 token，主模型
一次前向校验并接受前缀。实测（单卡；预填充行是 12k 上下文，解码行是 1649 预填充 + 64 生成）：

| 项 | 不开 MTP | MTP3 | 变化 |
|---|---|---|---|
| 预填充 @12k | 423.9 t/s | 417.9 t/s | -1.4% |
| 解码 | 25.5 t/s | **48.8 t/s** | **1.91×** |
| MTP 接受 | — | 78.9%（2.368 token/轮） | — |

> 注意：这行的 48.8 t/s 对应**78.9% 接受率的长 prompt**。MTP 的收益就等于接受率，
> 换成低接受率的 prompt 会退化到 ~25 t/s（与不开 MTP 持平）。交付前用仓库自带命令
> 复测的完整对照见 `REPRODUCE.md` §8。

预填充随上下文的衰减很小：不开 MTP 时 1k/12k/32k = 431.5/423.9/409.5 t/s
（12k 仅 -1.8%，32k 才 -5.1%）。

**等价性**：同一 prompt 的贪心输出与不开 MTP **逐 token 完全一致**（短/长提示、32/64 token
都验过）；引擎里的 `ROLLTEST2` 对保留长度 1..4 分别比较「验证批 + 回滚」与逐 token 前向的
logits，全部 max diff = 0。

**这一轮踩到的三个真坑**（都已修，修错任何一个都会导致长序列悄悄漂移）：

1. **验证批不能用预填充 FA**：预填充 FlashAttention 的 P 是 4bit，单 token 解码是 fp32 P。
   用预填充 FA 校验会让“目标 logits”本身和普通解码不是同一算术，几轮后轨迹分叉。
   现在验证批逐行走单行解码内核（`snap_mode && T<=4` 分支）。
2. **V cache 的 64-key tile 尺度不能跨 tile 回滚**：V 的尺度按 tile 共享，验证批跨过
   64 边界再回滚到旧 tile 时会读到新 tile 的 stage。现在主模型与 MTP 的 V-stage 都做
   逐 token 快照，回滚时一并恢复。
3. **卷积状态的逐 token 快照差了一位**：`state[r]` 对应的是 `t-(K-2)+r`，不是
   `t-(K-1)+r`。这个 off-by-one 会让 GDN 回滚后的状态比正确状态“老一格”，数轮后分叉。

另外 MTP 的 KV 是独立的 1 层注意力 KV：预填充时按「token 错一位 + 主模型最终 norm
隐藏态」建立，草稿时链式复用上一轮 MTP 隐藏态；全部接受时还要补一个条目，使
`mtp_len = seq_len-1`。请求级开关：OpenAI 请求体加 `"mtp": 0..3`；命令行 `--mtp-n`；
`RT_NO_MTP=1` / `--no-mtp` 完全不加载权重。

校验批的 GEMV 已做第一轮提速：一个 warp 处理 2 行权重、共享激活，M=4 带宽
260 → **317 GB/s**（M=3 322 → **379**），解码 43.8 → **48.8 t/s**。M=5 只有 266 GB/s，
所以 K=4 不划算。

**下一步**：~~验证注意力逐行跑解码内核~~ → 已完成「一次算完 T 行」
（`fa_decode_rows_k`，逐行逐位等价，32k +3.0%，见 [docs/MTP.md](docs/MTP.md) 第 8 节）；
接着是 M=4 的共享内存激活暂存、KV int4 与 128k 验收。

## 状态（2026-09-28 晚）：**能对话，而且吐字快了 2 倍** ✅

上一轮解决了「模型能不能说人话」（见下）；这一轮解决「说得快不快」。同口径实测
（单卡，1.65k 预填充 + 64 token 解码）：

| 项 | 上一轮 | 现在 | 倍数 |
|---|---|---|---|
| 预填充 1649 tok | 280.7 t/s | **419.6 t/s** | 1.50× |
| 解码（贪心） | 12.7 t/s（78.9 ms/token） | **26.2 t/s（38.2 ms/token）** | 2.06× |
| 解码（temp=0.7） | 10.1 t/s | **25.2 t/s** | 2.50× |
| 解码 @ 4k 上下文 | 20.7 t/s（48.3 ms） | **27.6 t/s（36.3 ms）** | 1.33× |

**完整报告见 [docs/PERF-DECODE.md](docs/PERF-DECODE.md)（这一轮必读）**，里面有：
分阶段计时怎么做才不骗自己（`hipEventRecord` 本身每次要 ~14 µs，会把小内核放大）、
每一项优化的实测数字、试过但更慢的方案、以及正确性验证矩阵。

### 这一轮改了什么（按收益）

| # | 位置 | 症状 | 改法 | 收益 |
|---|---|---|---|---|
| 1 | `src/k_new.hip` `gemv_f32_k` | **每个 block 只有 1 个线程**，解码时整卡 96 个活跃 lane | 一个 block 一行输出 + 128 线程归约 | 19.1 → 1.4 ms |
| 2 | `src/k_new.hip` `k_ssm_ab_gate`（新） | a/b 投影 + sigmoid + ssm_gate 是 4 次启动/层 | 合成一个内核（保持 db 阶段 7/8/9 语义） | 见下 |
| 3 | `src/k_attn_q_quant`（新） | 每个头一次小启动，且按 TP=64 行补齐量化 | 单次启动、只量化真实的 T 行 | 11.6 → 1.4 ms |
| 4 | `src/model.cpp` `attention_layer` | 每层 `hipMalloc` + **阻塞 H2D** + `hipFree`（RoPE 位置） | `k_rope(..., pos=nullptr, pos0=seq_len)` | 总时间 -7 ms |
| 5 | `src/model.cpp` `Sampler` | `top_k=0/top_p=1` 时对 248320 个 logit **全排序** | 无截断走两遍 O(V) 精确采样；有截断用 `nth_element` | temp>0 -20 ms |
| 6 | `src/k_new.hip` `gdn_k2`（新） | 状态布局 `[h][col][i]`，warp 每次 load 碰 32 条 cache line | 布局改 `[h][i][col]` + i 切 4 段 | 4.4 → 1.9 ms |
| 7 | `src/k_fa.hip` `fa_decode_k`（新） | 解码走预填充内核，按 64 行补齐 = 白算 64 倍 | 单行专用内核 + split-KV 合并 | 4k 上下文 -12 ms |
| 8 | `kernels/gemv_w4a8_core.h` | `A8_ROWS=2` | 实测 1/2/4 → 38.0/40.5/252.7 ms，默认改 1 | -2.5 ms |
| 9 | `scripts/serve.py` | 不认识 `--ctx`（serve.sh 一直在传）→ 容器反复重启 | 加 `--ctx` 并透传给引擎 | 服务能起 |

**验证**：`tools/verify_plain.py` 449/449 一致；`scripts/test_tools.sh` 通过；
单层逐阶段对照（GDN 层 / 注意力层）全绿；新写的
`scripts/attn_decode_check.py` 用 fp64 重算单行注意力，短上下文 1.7e-07、
长上下文 + split 9.8e-08；GDN 改写前后贪心生成前 19 个 token 完全相同。
服务端到端：`bash scripts/serve.sh` 起来后 58 提示 + 55 生成 = 2.09 s（26.3 t/s），
内容正确；请求体加 `{"reasoning_effort":"low"}` 可换短推理。

### 下一步（按杠杆排序，详见 docs/PERF-DECODE.md 第 5 节）

1. **减少每 token 的权重读取**——唯一能再提一个数量级的方向。13.22 GB/token 在
   620 GB/s 下就是 21.3 ms 的硬地板。要么 **MTP/投机解码**（权重只读一遍、一次前向
   吐 2~4 个 token；`rt4/qwen38_27b_mtp.rt4` 已备好，但需要给 KV 加回滚），要么在
   服务层做**并发批请求**（多路请求共享一次权重读取，改动最小、收益最直接）。
2. **权重精度**：int4/128 的 12%~17% 残余误差还在（top-1 偶尔偏一位）。
   用 `tools/ref_full.py --src` 做量化方案消融（每 32/16 组、非对称、FP8 张量直接存
   int8），定下来再改转换器+内核。注意源模型本来是混合精度：attention/linear_attn/
   lm_head/56~63 层 MLP 是 **FP8**，0~55 层 MLP 才是 NVFP4。
3. **预填充 1000 t/s**：现在 419。瓶颈回到 `docs/FLASH-ATTN.md` 的注意力指令密度
   （寄存器压力 + 非 dot8 指令），以及 W4A8 预填充要跑两遍 int4 GEMM 的 2 倍代价。
4. **FA 的 P 精度**：P 现在 4bit（`p*7` 取整），消融显示它单独贡献注意力输出 41%
   的误差；改 8bit（`16h+l` 跑两遍）收益最大。解码路径已经是 fp32 P 了，只剩预填充。
5. 元素级小内核合并（残差加法 + 下一层 rmsnorm、split_qkv + l2norm）、
   128k 上下文、分页 KV、int4 KV。

## 上上轮状态（2026-09-28 凌晨）：**已经能对话了** ✅

**模型现在能出连贯的话**（中文自我介绍、量子纠缠解释、9.11 与 9.9 比较都对；
英文 "the lazy → dog"、"hydrogen and → oxygen" 也对），对外提供 OpenAI 兼容接口：

```bash
bash scripts/serve.sh                     # 起服务（8080 端口，局域网可直连）
python3 scripts/chat.py --prompt "你好" --n 128   # 或直接命令行对话（容器里跑）
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"你好"}],"max_tokens":64}'
```

实测（单卡，1.65k 预填充 + 64 token 解码）：预填充 **273 t/s**、解码 **12 t/s**。
质量与速度都还有余量，但**「能对话/能接 API」这一里程碑已经达成**。

**这一轮修掉 7 个真 bug，其中一个致命**：`tools/convert.c` 把 BF16 源张量写进 RT4 的
`f32` 槽时**按 4 字节原样搬**，449 张非量化张量（每层 norm / A_log / dt_bias / conv1d /
ssm_norm / 最终 norm）全部错位——第 0 层输出 RMS 从 0.228 变成 2.177，残差流被放大 10 倍，
64 层后就是乱码。**完整清单、根因、以及为什么以前的对照发现不了，见
[docs/QUALITY-FIX.md](docs/QUALITY-FIX.md)（这一轮必读）**。

回归必跑：

```bash
bash scripts/build_rt.sh
bash scripts/dsh.sh 'python3 tools/verify_plain.py /rt/models/Qwen3.8-27B-NVFP4/model.safetensors \
  /rt/models/Qwen3.8-27B-NVFP4/rt4/qwen38_27b.rt4 /rt/models/Qwen3.8-27B-NVFP4/rt4/qwen38_27b.rt4.json'
```

### 下一步（按价值排序，详见 docs/QUALITY-FIX.md 第 6 节）

> 这一节的第 1 条（解码提速）已经在「2026-09-28 晚」那一轮做完，见
> [docs/PERF-DECODE.md](docs/PERF-DECODE.md)。第 2~4 条仍然成立。

1. ~~**解码提速**~~ → 已完成：12.7 → 26.2 t/s（单行解码注意力 + split-KV、
   采样器去掉全排序、GDN/SSM 内核重写、去掉每层的临时分配）。
2. **权重精度**：RT4 的 int4/128 逐张量 12%~17% 误差还在（top-1 偶尔错一位）。
   先用 `tools/ref_full.py --src <safetensors>` 做量化方案消融（每 32/16 组、非对称、
   FP8 张量直接存 int8……），定下来再改转换器+内核。注意源模型本来是混合精度：
   attention/linear_attn/lm_head/56~63 层 MLP 是 **FP8**，0~55 层 MLP 才是 NVFP4。
3. **FA 的 P 精度**：P 现在是 4bit（`p*7` 取整），消融（tools 里的 attn 误差分解）显示
   它单独贡献注意力输出 41% 的误差；改成 8bit（同样 `16h+l` 跑两遍）收益最大。
4. MTP 链、128k 上下文、分页 KV。

## 更早的状态（2026-09-27 晚，已过时）

**独立项目**：2026-09-27 从内部早期项目 `k100LC-perf/rt/` 拆出来，自成一个项目。
当时同机还跑着一条 **llama.cpp 性能线**（`k100LC-perf/`，只作为对照基线，不在本仓库），
本项目是**自研框架与定制算子**这条线；两条线共享同一张卡，动手前会先看一眼对方有没有在跑。

## 状态（2026-09-27 晚）

目标：K100_LC(gfx926) 单卡、128k 上下文、**预填充 1000 t/s + MTP3 吐字 100 t/s**，
不借用其他框架。两条硬指标**都还没达成**，但卡点已经实测定位：

| 环节 | 现状 | 目标 | 差距 |
|---|---|---|---|
| 预填充：注意力形状（K=256） | **25.49 TMAC/s（33.8%）** int4 FA 已跑通 | ≥55 TMAC/s | 差 2.2 倍，缺的是指令密度与寄存器 |
| 预填充：FFN/线性形状 | 48.6 TMAC/s（64.4%） | ≥52 TMAC/s | 差 1 档 |
| 吐字：M=4 权重带宽 | 390 GB/s | ≥600 GB/s | 需共享内存分块+双缓冲 |
| 吐字：MTP 接受率 | 未实现（llama.cpp 对照只有 32%） | ≥2.6 token/步 | 需按共享隐藏态的 MTP 链实现 |

按现有实测数字折算：预填充 **~650 t/s**（本轮从 519 提上来）、吐字 ~58 t/s（接受 2.5 token/步）。

## 本轮（2026-09-27 深夜）：开始写「能对话」的全模型运行时（未完成）

用户把方向改成 **先把运行时做到真的能对话、能接 API，性能优化放第二步**。
新建了 `src/`（加载器 + 全模型图 + 算子）与 `tools/ref_layer.py`（单层 torch 参考实现），
详细记录见 **[docs/RUNTIME.md](docs/RUNTIME.md)**。

* **能跑通**：mmap 13.91GB RT4 + 分块 H2D（宿主内存 <1GB）、64 层前向、
  T=8 走 int4 GEMM / T≤4 走解码 GEMV、打印 top-k logits。跑法 `bash scripts/build_rt.sh`。
* **修掉了 6 个真 bug**（详细清单见 docs/RUNTIME.md）：权重尺度是 f16 被当 f32 读、
  `k_gemv_f32` 漏 blockIdx.y、GDN 漏 q 的 1/sqrt(d) 缩放、注意力漏 score 缩放、
  FA 内核头内偏移要用补齐行数、GEMV 尾部越界读激活尺度（0*NaN 污染整行）。
* **还没达标**：logits 太平（top-1 ~7.8，正常 15+），说明还有实现错误。
  单层对照（tools/ref_layer.py，参考端也模拟 int4 激活量化）相对 RMS 约 5~8%，
  注意力路径仍差得多；T≤4 时隐藏态最后一维出现 NaN。下一步定位见 docs/RUNTIME.md。

## 本轮（2026-09-27 晚）：int4 FlashAttention 写完并实测

新增 **`kernels/flash_attn_int4.hip`**（+ `scripts/run_fa.sh`、`docs/FLASH-ATTN.md`）。
这是 `docs/FORMAT.md` 第 7 节点名的预填充关键路径：通用 GEMM 打「K=256 窄 K」的注意力
形状只有 18.66 TMAC/s，所以改成「一个 block 沿 key 方向长流水、head_dim 256 整个进内层」。

1. **结果**：目标形状（24 头 / GQA 6:1 / 1024 行 query / 131072 上下文 / 2064 个 tile）
   **65.20 ms/层 = 25.49 TMAC/s（dot8 上限的 33.8%）**，比通用 GEMM 老路线的
   18.66 快 **1.37 倍**；折算 128k 预填充 **519 → 约 650 t/s**（注意力 1.01s +
   线性 0.53s）。**没到 35~45 的目标档**，但差距的性质已经完全清楚了（见第 3 条）。
2. **正确性**：三重对照（`docs/FLASH-ATTN.md` 第 2 节）。内核 vs「算法模型」（把
   per-tile 行最大、beta 折叠、online 重标定用 double 原样重算）**5.5e-08（短）
   ~ 3.0e-06（128k）**，即内核就是照着设计跑的；vs 精确 double softmax 在
   长上下文也只有 **4.2e-06**（定点 P 的代价）。短上下文 7.7e-03 是 P 量化在有界
   数据上的正常量级。
3. **卡点诊断（消融表在 docs/FLASH-ATTN.md 第 3 节）**：
   * **寄存器压力是头号矛盾**：同一份代码 NT=512 溢出 152B → 14.6 TMAC/s，
     换 NT=256 不溢出（245 VGPR）→ 22.6；O 的 fp32 累加器与 PV 的 int32 累加器
     各占 64 个 VGPR，是 4 wave/CU 上不去的原因。
   * **指令密度**：每线程每 tile 512 条 dot8 + 约 500 条其它（软最大 ~150 为主），
     dot8 只占 ~50% 的指令槽 ⇒ 天花板约 38 TMAC/s，现在离它还差 1.5 倍。
   * **bank 冲突**（已修）：K/V 的 tile 里每个 lane 要连续 8 列、lane 间隔 8 个 dword，
     自然序存放时 LDS.128 退化成 4 路冲突；把每组 8 列拆成两个 4 列半段交错存放后
     **+13%**（22.3 → 25.2）。
   * V 的 cache 布局定为**沿 key 转置打包**（`[n_kv/8][256]`），写入侧转一次、
     读侧（注意力预填充与解码）都受益。

## 本轮（拆分整理）做了什么

1. 把自研部分从早期项目里拆成独立项目（权重用同盘 rename，不占额外空间）：
   `docs/`（4 篇文档）、`tools/`（转换器与验证工具）、`kernels/`（2 个 GEMV）、
   `bench/`（早期算力/GEMM 原型拷贝）、`models/Qwen3.8-27B-NVFP4/`（源权重 + RT4 产物 35GB）。
2. 新增 `scripts/`：`test_tools.sh`（秒级自检）、`run_kernel.sh`（编译+跑算子）、
   `convert_weights.sh`（下载+转换，幂等）、`env.sh`（共享环境）。
3. 编译/运行脚本支持 sudo 免密助手（`$RT_ROOT/.askpass.sh`，权限 700）：里面是
   **明文口令，已被 `.gitignore` 排除，不要入库**；没有这个文件时脚本自动退回普通 `sudo`。
4. 清掉了早期目录下只剩可重建的编译产物（564KB），目录整树移除。

## 下一步（按优先级）

1. **把注意力从 25.5 推到 55 TMAC/s**（1000 t/s 的必要条件，`docs/FLASH-ATTN.md` 第 5 节）：
   * 先压非 dot8 指令：软最大那 ~150 条改用 16-bit 打包归约（`v_pk_max_f16` /
     `v_pk_add_u16`），P 的量化+打包用位运算一次成型 → 预期 dot8 占比 50% → 65%。
   * 再腾寄存器：O 累加器（64 个 VGPR）是最大一块，试验 bf16 对存法或沿 head_dim
     拆两个 block；腾出来之后才能上 TM=4/TN=8（把每 dot8 的 LDS 从 2.5B 降到 1.5B）。
   * split-KV（key 方向再切 2~4 份）改善 3.2 波的尾部损失与并行度。
2. **M=4 GEMV 提速**：权重走共享内存分块 + 双缓冲；同构纯读探针上限是 692GB/s，
   M=1 已经吃到 91%（566/620），M=4 只吃到 56%（390/692），差的是 4 倍 dot8 与激活流量。
3. **KV/SSM**：int4 KV（4.30→2.15GB）、q8/int4 分页解码注意力（直接吃本轮定下的
   V 转置布局）、gated delta net 扫描。
4. **MTP 链**：`mtp.fc` 拼接 [embedding, hidden] + 1 层注意力/MLP，复用主模型算子；
   权重已转好（`rt4/qwen38_27b_mtp.rt4`，0.22GB）。
5. 装载器（mmap 权重 + 分块 H2D）、图执行、采样器、分词器、HTTP 服务层；
   最后与 llama.cpp 同 prompt 对齐 logits（cos > 0.999）再谈指标。

## 复现命令

```bash
cd K100LC-RT4
bash scripts/test_tools.sh          # 转换器自检（合成用例）
bash scripts/run_kernel.sh          # 现役 W4A4 解码 GEMV（M=4）
bash scripts/run_fa.sh              # int4 FlashAttention：正确性 + 128k 吞吐
bash scripts/convert_weights.sh --check   # 权重完整性校验（sha256 + 产物清单）
```
