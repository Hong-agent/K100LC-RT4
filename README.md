# K100LC-RT4 — Qwen3.8-27B 自研推理运行时（海光 K100_LC / gfx926）

在海光 **K100_LC（gfx926）** 上，为 `unsloth/Qwen3.8-27B-NVFP4` 从零写的推理运行时与定制算子。
**不依赖 vLLM / SGLang / llama.cpp 等任何推理框架**，目的是把这张卡的专用指令通路吃满。

名字里的两半：**K100LC** = 海光 K100_LC（`gfx926`）；**RT4** = 本项目自研的权重与运行格式。

## 目标（用户指定，硬指标）

| 项 | 指标 |
|---|---|
| 上下文 | 128k |
| 预填充 | 填充 1000 token，目标 **1000 t/s** |
| 解码 | 开 **MTP3**，输出 100 token，目标 **100 t/s** |

## 现状（2026-09-27，如实列出未达标项）

| 项 | 现在 | 说明 |
|---|---|---|
| 预填充 | 1649 token **419.6 t/s**；1k/12k/32k 上下文 431.5 / 423.9 / 409.5 t/s | 目标是 1000 t/s，**未达标**，瓶颈在注意力指令密度 |
| 解码 | 不开 MTP **25.5 t/s**；MTP3 在长 prompt 上 **48.8 t/s**（接受率 78.9%），但**强依赖 prompt**：交付前实测同一命令在低接受率 prompt 上只有 25.2 t/s（接受 21%），高接受率 filler 上 57 t/s | 目标是 100 t/s，**未达标**；硬地板是每 token 13.2GB 权重读取，MTP 的收益等于草稿接受率（见 [REPRODUCE.md](REPRODUCE.md) §8） |
| 上下文 | 跑通到 32k（服务默认 32768） | 128k 未做验收 |

**从零复现（含权重下载、转换、基准与验收清单）见 [REPRODUCE.md](REPRODUCE.md)。**

## 这张卡的关键事实（本机实测）

* `gfx926`，120 CU @1270MHz，64GB DDR6，**没有矩阵核心**（MFMA/WMMA 汇编器全部拒绝）。
* 算力通路（`bench/bench_isa2.cpp`）：

| 指令 | 实测 | 相对 fp16 |
|---|---|---|
| `v_pk_fma_f16`（打包 fp16） | 33.7 TFLOPS = 16.9 TMAC/s | 1.0× |
| `v_dot4_i32_i8`（int8 点积） | 37.9 TMAC/s | 2.25× |
| **`v_dot8_i32_i4`（int4 点积）** | **75.5 TMAC/s** | **4.5×** |

* 显存可用带宽实测 **804 GB/s**（128-bit 网格跨步读）。
* 结论：要冲 1000 t/s 预填充，**权重和激活都必须是 int4**，走 `v_dot8_i32_i4`。

## 目录

```
K100LC-RT4/
├── docs/
│   ├── FORMAT.md          RT4 运行格式 + 框架设计 + 带宽/算力预算（先看这个）
│   ├── CONVERT-VERIFY.md  权重转换的完整验证报告
│   ├── GEMV.md            解码 GEMV 的实测记录（W4A8 / W4A4 两版 + 消融）
│   ├── FLASH-ATTN.md      预填充 int4 FlashAttention：设计、正确性三重对照、消融表
│   ├── PERF-DECODE.md     解码 12.7 → 26.2 t/s 的完整报告（含计时方法与坑）
│   ├── MTP.md             MTP3 投机解码、状态回滚与三个等价性坑
│   ├── VISION.md          本地 Qwen3.5 视觉塔、权重导出与 embedding 注入协议
│   ├── QUALITY-FIX.md     「模型能不能说人话」那一轮：7 个 bug 的根因与修法
│   ├── RUNTIME.md         运行时框架与本轮之前的对照记录
│   └── DESIGN-INT4.md     最初的路线论证（为什么必须重写算子）
├── tools/
│   ├── convert.c          源 safetensors(NVFP4/FP8) → RT4（mmap 流式，宿主内存 <100MB）
│   ├── fetch_par2.py      多连接断点续传下载器（含 sha256 校验）
│   ├── make_test_st.py    合成用例：造 NVFP4+FP8 小文件
│   ├── check_rt4.py       合成用例的逐元素比对
│   ├── spot_check_rt4.py  真实权重的端到端抽查
│   └── verify_vs_gguf.py  与 BF16 原模型逐张量比 RMS（验证反量化公式）
├── kernels/
│   ├── flash_attn_int4.hip 预填充 int4 FlashAttention 基准（128k，online softmax）
│   │                     └─ 内核本体 flash_attn_core.h，运行时与基准共用
│   ├── gemv_w4a8_core.h   解码 GEMV 现役内核：int4 权重 × int8 激活（dot4）
│   ├── gemv_int4.hip      ↑ 的独立基准与实测（现役，见 docs/GEMV.md）
│   ├── gemv_w4a4_core.h   int4 激活版（dot8），只在 RT_ACT4=1 的性能对照里用
│   └── gemv_w4a4.hip      ↑ 的独立基准
├── bench/                 早期算力/访存/GEMM 原型（int4_gemm2.cpp 是预填充 GEMM 现役版本）
├── scripts/
│   ├── env.sh             共享环境（镜像、路径、askpass）
│   ├── test_tools.sh      工具链自检（秒级）
│   ├── run_kernel.sh      编译并跑一个算子
│   └── convert_weights.sh 下载 + 转换权重（幂等）
│   ├── build_rt.sh        编译全模型运行时 → build/rt
│   ├── prepare_vision.sh  从源权重导出视觉塔 RT4 → qwen38_27b_vision.rt4
│   ├── vision_encoder.py  图片预处理 + 调用运行时 IMG_EMB
│   ├── dsh.sh             在 DTK 容器里跑任意命令（编译/测试都走它）
│   ├── chat.py            命令行对话（分词 → 引擎 → 解码）
│   ├── serve.py/serve.sh  ★ OpenAI 兼容 HTTP 服务
│   ├── dec_bench.py       解码墙钟基准（各条优化的 A/B 用）
│   ├── dec_bench_ctx.py   解码速率 vs 上下文长度
│   ├── attn_decode_check.py  解码注意力的独立复核（内核 vs fp64）
│   ├── make_dist.sh       ★ 生成 dist/K100LC-RT4-offline 离线包（见该目录 README-OFFLINE.md）
│   └── run_fa.sh          int4 FlashAttention 正确性与吞吐
│   └── run_fa_rows.sh     验证批解码注意力自检（逐行 vs 一次算完，逐位比较）
└── models/Qwen3.8-27B-NVFP4/
    ├── model.safetensors           源权重 22.57GB（NVFP4 混合精度，sha256 已校验）
    ├── model_mtp.safetensors       MTP 头
    └── rt4/
        ├── qwen38_27b.rt4          自研运行格式 13.91GB（851 张量，统一 int4/128 组）
        ├── qwen38_27b_mtp.rt4      MTP 头 0.22GB
        ├── qwen38_27b_vision.rt4   视觉塔 RT4 0.93GB（线性 f16，norm/pos f32）
        ├── kv_scales.json          源模型附带的 8bit KV 尺度（本运行时**不用**，KV 为内核自算尺度）
        └── convert.log             逐张量 relerr / src_rms
```

## 快速开始

```bash
cd K100LC-RT4              # 克隆下来的仓库目录；权重按 REPRODUCE.md 第 4、5 步准备

bash scripts/prepare_vision.sh             # 首次：导出视觉塔，启用网页图片理解
bash scripts/serve.sh                      # ★ 起 OpenAI 兼容服务（http://<ip>:8080/v1）
# 浏览器打开 http://<ip>:8080/ 就是自带的简易对话网页（流式、任意模型名，
# 底部显示预填充 t/s、生成 t/s、实时 t/s、首 token 延迟和 token 数；
# “思考”可选 low/medium/xhigh 或“关闭”，关闭时请求带 enable_thinking=false）
bash scripts/dsh.sh 'python3 scripts/chat.py --prompt "你好" --n 128'   # 命令行对话
RT_MTP_N=0 bash scripts/serve.sh           # 只关闭投机（仍加载 MTP 权重）
NO_MTP=1 bash scripts/serve.sh             # 完全不加载 MTP 权重
bash scripts/test_tools.sh                 # 工具链自检（秒级）
bash scripts/dsh.sh 'python3 tools/test_attachments.py'   # 文档提取 + 图片多模态消息自检
bash scripts/run_kernel.sh                 # 跑解码 GEMV 基准（默认 W4A4, M=4；现役是 gemv_int4.hip）
bash scripts/run_fa.sh                     # int4 FlashAttention：正确性 + 128k 吞吐
bash scripts/run_fa_rows.sh                # MTP 验证批注意力：逐行 vs 一次算完自检
bash scripts/dsh.sh 'python3 scripts/bench_rt.py'        # ★ 预填充 + 解码总表
bash scripts/dsh.sh 'python3 scripts/dec_bench.py 48'    # ★ 解码墙钟（默认走 MTP）
bash scripts/dsh.sh 'RT_NO_MTP=1 python3 scripts/dec_bench.py 48'  # 纯逐 token 对照
bash scripts/dsh.sh 'python3 scripts/dec_bench_ctx.py 60,1000,4000'   # 解码 vs 上下文
bash scripts/dsh.sh 'python3 scripts/attn_decode_check.py 3 11'       # 解码注意力复核
FLAGS="-DROWS=2" bash scripts/run_kernel.sh
bash scripts/convert_weights.sh --check    # 校验权重产物是否完整
bash scripts/dsh.sh 'python3 tools/verify_plain.py \
  /rt/models/Qwen3.8-27B-NVFP4/model.safetensors \
  /rt/models/Qwen3.8-27B-NVFP4/rt4/qwen38_27b.rt4 \
  /rt/models/Qwen3.8-27B-NVFP4/rt4/qwen38_27b.rt4.json'    # ★ 权重一致性回归
```

一切编译/运行都在 DTK 容器里（本机没有 DTK 工具链），编译目标固定 `--offload-arch=gfx926`。

服务请求体可加 `"mtp": 0..3` 临时覆盖草稿数（0 即普通逐 token 解码），例如
`{"messages":[...],"max_tokens":64,"mtp":0}`。
`model` 字段不参与路由：填 `qwen38-rt` 或任意名字都会正常调用，响应按请求里的名字回显。
服务端会把 `</think>` 前的思考放到兼容字段 `reasoning_content`/`reasoning`，正式回答放
`content`；`enable_thinking=false` 时全部是 `content`。

网页支持附件（也支持拖拽和粘贴图片）：

* **文档**：可选/拖入 `txt/md/json/csv/html/xml/pdf/docx/pptx/xlsx/rtf/odt`
  及常见代码/日志文本；服务端 `POST /v1/extract` 提取为文本，再随消息送入 RT4。
  `pdf` 在容器装有 `pypdf` 时走完整解析，否则退回内置基础提取器。
* **图片（本地视觉塔，默认）**：网页显示缩略图，视觉塔 27 层也统一成
  RT4 权重并在自研 C++/HIP 运行时里执行；Python 只做图片预处理，把 patch
  交给引擎 `IMG_EMB`，得到的 embedding 再通过 `PREFILL_EMB` 注入文本运行时。
  视觉权重从源 safetensors 导出：

```bash
bash scripts/prepare_vision.sh    # 产物 rt4/qwen38_27b_vision.rt4，约 0.93GB
bash scripts/serve.sh
```

  视觉塔随 `build/rt` 在启动时加载；`/v1/capabilities` 会显示
  `"vision_backend": "local"`。实测本地视觉塔能直接读图中文字（例：上传写着
  `7788` 的图片，模型回答 `7788`），多图、流式、MTP 都通过。

* **图片（外部视觉桥，可选）**：也可以改用一个 OpenAI 兼容的视觉端点，
  服务端会先把图片转成文字描述，再交给 RT4：

```bash
RT_VISION_BASE_URL=https://your-vl-host/v1 \
RT_VISION_MODEL=your-vl-model \
RT_VISION_API_KEY=... \
bash scripts/serve.sh
```

能力查询：`curl http://127.0.0.1:8080/v1/capabilities`。上传上限默认 25MB，
可用 `RT_MAX_UPLOAD_MB` 调整；单次提取文本默认最多 40000 字符，可用
`RT_MAX_EXTRACT_CHARS` 调整。`max_tokens` 是**生成上限**而不是要预留的上下文：
客户端发 `128000`（网页默认）时，服务会按剩余上下文自动收窄（`timings.max_tokens`
返回实际值），只有 **prompt 本身就超出上下文**才返回 413；可用 `--ctx` / `CTX`
放大上下文（默认 131072）。

网页的「停止」按钮现在会真的停住引擎：客户端一断开，服务端发 `STOP` 给引擎
（引擎在 GEN 循环里非阻塞检查，一轮 MTP 之内退出，`END ... stopped`），等引擎收尾
再释放请求锁——所以停止之后立刻发下一条不会等待，也不会串线。

网页也支持简单技能和文件工作区：

* 右上角“技能”开关默认打开。模型可在一次回答里连续调用 `now`、`calc`、
  `write_file`、`read_file`、`list_files`、`delete_file`、`make_csv`。
* **流式**：技能开着也是边生成边发（工具轮先流出草稿，判定为工具调用后前端会
  丢弃草稿、显示技能事件，再流最终回答）。默认开着思考，思考段显示为气泡里的
  灰色小字（`💭`），正式回答在思考结束后继续流出；等待期间气泡里有 `⏳` 计时。
  想直接出结论可把网页的“思考”选成 `off`。
* 每个浏览器会话有独立工作区（`conversation_id` 存在 localStorage）；模型创建的
  文件会出现在输入框上方的文件栏，可直接“打开”预览或“下载”。
* 多轮任务会保留工作区文件，服务端每轮把当前文件列表注入上下文；模型可以继续
  `read_file` / `write_file` 完成后续任务。
* 文件接口：`GET /v1/files?conversation_id=...`、`GET /v1/files/<name>`、
  `DELETE /v1/files/<name>`。默认单文件 2MB、单会话工作区 64MB，可用
  `RT_MAX_FILE_KB` / `RT_MAX_WORKSPACE_MB` 调整；工作区默认在
  `RT_WORKSPACE_DIR`（未设置时是仓库下 `workspaces/`）。

## 现状

**2026-09-27（本轮）：MTP3 投机解码已接入。** 权重同目录存在 `qwen38_27b_mtp.rt4` 时
默认启用，贪心路径一次草拟 3 个 token、主模型一次前向校验并接受前缀。实测（单卡；
预填充行是 12k 上下文，解码行是 1649 token 预填充 + 64 token 生成）：

| 项 | 不开 MTP | MTP3 | 变化 |
|---|---|---|---|
| 预填充 @12k | 423.9 t/s | 417.9 t/s | -1.4%（MTP 自己也要建 KV） |
| 解码 | 25.5 t/s | **48.8 t/s** | **1.91×** |
| MTP 接受 | — | 78.9%（2.368 token/轮） | — |

同 prompt 的贪心输出与不开 MTP **逐 token 完全一致**（32/64 token 两组、短/长提示都验过）；
`ROLLTEST2` 还对保留长度 1..4 分别验了「验证批 + 回滚」与逐 token 前向的 logits 完全一致。
代价是验证批走 M=3~4 的 GEMV；本轮把校验批的每个 warp 改成处理 2 行权重（`ROWS=2`），
让两行共享激活加载，W4A8 M=4 带宽从 260 提到 **317 GB/s**（M=3 从 322 提到 **379**）。
低接受率提示收益小；`"mtp":0` 可以逐请求关掉。MTP 的 KV 是独立的 1 层注意力 KV，预填充时按
「token 错一位 + 主模型最终 norm 隐藏态」建立，草稿时链式复用上一轮 MTP 隐藏态。
预填充不随 12k 上下文明显掉速：不开 MTP 时 1k/12k/32k 分别是 431.5/423.9/409.5 t/s
（32k 才 -5%）；开 MTP3 后 12k 也只从 423.9 降到 417.9 t/s。

**2026-09-28 晚：解码提速 2 倍（MTP 之前的口径）。** 预填充 1649 tok **419.6 t/s**、
解码 **26.2 t/s**
（38.2 ms/token；temp>0 也有 25.2 t/s），解码速率基本不随上下文退化
（60 → 8000 key：36.5 → 37.3 ms/token）。这一轮排掉的主要是「实现级浪费」：
一个每 block 只有 1 个线程的 GEMV、每层一次 `hipMalloc`+阻塞 H2D 拷贝、按 64 行
补齐的 Q 量化、采样器对 248320 个 logit 全排序、GDN 状态布局导致访存完全不合并、
以及解码注意力白算 64 倍。全过程与实测见 **[docs/PERF-DECODE.md](docs/PERF-DECODE.md)**。

**2026-09-28 凌晨：端到端跑通，能对话了。** 预填充 273 t/s、解码 12 t/s（还没优化），
中文/英文生成都通顺，`scripts/serve.py` 提供 OpenAI 兼容接口。
本轮排错全过程见 **[docs/QUALITY-FIX.md](docs/QUALITY-FIX.md)**（含 7 个 bug 的根因与
「为什么以前的单层对照发现不了」——最要命的是转换器把 449 张非量化张量写错位）。

已完成：

* **本地视觉塔**：把 Qwen3.5 `model.visual.*` 转成独立 RT4（线性层 f16，
  norm/bias/pos f32，约 0.93GB），27 层视觉块在自研 C++/HIP 运行时执行，
  经 `PREFILL_EMB` 把图像 embedding 注入文本运行时；单图、多图、流式、MTP
  与网页上传都已跑通。见 [docs/VISION.md](docs/VISION.md)。
* **权重链路**：源 22.57GB（sha256 与 hf-mirror 一致）→ RT4 13.91GB / 851 张量 / 4分37秒；
  与 BF16 原模型逐张量核对（线性层 RMS 比值 0.995~1.004），真实张量抽查相对误差正好等于
  声明的重量化误差（说明偏移/行序/分组无误）。MTP 头与 KV 尺度也一并产出。
* **解码 GEMV 算子**：W4A8 与 W4A4 两版都跑通，与 double 参考最大绝对误差 1.8e-07。
  实测带宽 M=1 **566**、M=4 **390** GB/s（同构纯读探针上限 620/692）。

* **预填充 int4 FlashAttention**（`kernels/flash_attn_int4.hip`，本轮）：一个 block 沿
  key 方向长流水、head_dim 256 整个进内层、per-tile 行最大做 P 的定点尺度。
  目标形状实测 **25.49 TMAC/s（dot8 上限的 33.8%）**，比通用 GEMM 老路线
  （18.66，24.7%）快 1.37 倍 ⇒ 折算 128k 预填充 **519 → 约 650 t/s**。
  正确性：内核 vs 定点算法模型 **5e-08 ~ 3e-06**（128k 深度）。细节与消融见
  [docs/FLASH-ATTN.md](docs/FLASH-ATTN.md)。

还没有（按对目标的杠杆排序）：

0. ~~解码提速~~ **已完成**（12.7 → 26.2 t/s，见 docs/PERF-DECODE.md）；
   ~~MTP 投机~~ **已完成**（再 → 48.8 t/s，见上）。再往上要么把校验批的 M=4 内核
   做快，要么在服务层并发批请求，让多路请求共享一次权重读取。
1. **把注意力推到 55 TMAC/s**：现在 25.5。诊断已明确——dot8 只占指令槽的 ~50%
   （软最大那 ~150 条是大头）、245 VGPR 只够 4 wave/CU。先压非 dot8 指令，
   再腾 O 的累加器寄存器，最后上 TM=4/TN=8 与 split-KV。
2. **M=4 GEMV 再提速**（317 → 500+）：激活共享内存暂存 + 双缓冲。
3. **int4 KV 缓存**（4.30GB → 2.15GB，省约 6ms/步）；V 已定为「沿 key 转置打包」
   布局，解码注意力可直接复用本轮定下的格式。
4. **MTP 再提速**：接受率已经到 2.37 token/轮，校验批 M=4 GEMV 已从 260 提到
   317 GB/s（`NROW=2`）；验证注意力已改成「一次算完 T 行」的 `fa_decode_rows_k`
   （`grid.z` 放行，每行仍走同一个 device 函数，逐位等价；32k 上下文 +3.0%，
   见 [docs/MTP.md](docs/MTP.md) 第 8 节）。下一步：M=4 的激活共享内存暂存。
5. **128k 上下文、分页 KV、int4 KV**：当前服务默认 32768，KV 4.30GB @128k。
6. ~~gated delta net 扫描、图执行、采样器、分词器、HTTP 服务层~~ **都已完成**
   （`src/k_new.hip` 的 GDN/卷积、`src/model.cpp` 的采样器与引擎循环、
   `scripts/serve.py` + `web/index.html` 的服务层）。

对照基线（不在本仓库）：同一个模型换成 llama.cpp + Q4_K_M GGUF 跑，实测 373 t/s 预填充、
23.4 t/s 吐字；本项目的 OpenAI 接口与它保持一致，客户端不用改就能切换过去比数字。

## 许可

本项目源码使用 **Apache License 2.0**（见 [LICENSE](LICENSE)）——选它是因为它附带明确的
专利授权，对含 GPU 内核的工程更稳妥。第三方内容的归属见 [NOTICE](NOTICE)。

模型权重（`unsloth/Qwen3.8-27B-NVFP4`、BF16 GGUF）与海光 DTK 容器镜像属第三方，
各有各的许可，**都不在本仓库内**，请按 [REPRODUCE.md](REPRODUCE.md) 自行获取。
