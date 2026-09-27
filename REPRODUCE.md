# 从零复现（REPRODUCE.md）

这份文档的目标：**在一台干净的海光 DCU 机器上，从 `git clone` 开始，逐步跑到能对话、
能出基准数字、能跑完正确性验收**。所有命令都在这台机器上实跑过，期望值与实测一致
（同一份权重、同一版镜像、同一张卡的前提下）。

> 本仓库只放**源码**：不含权重、不含 Docker 镜像、不含编译产物。它们都要按下面第 3、4 步获取。

---

## 0. 这套运行时是什么

在**海光 K100_LC（`gfx926`）**上，为 `unsloth/Qwen3.8-27B-NVFP4` 从零写的推理运行时：
不依赖 vLLM / SGLang / llama.cpp，权重格式（RT4）、算子（HIP）、图执行、MTP 投机、
OpenAI 兼容服务层全部自研。设计取舍与实测数据都写在 `docs/`。

| 指标 | 目标（立项时） | 当前实测 | 数据来源 |
|---|---|---|---|
| 预填充 | 1000 t/s（1000 token） | 1649 token **419.6 t/s**；1k/12k/32k 上下文 431.5 / 423.9 / 409.5 t/s | RESUME.md、PERF-DECODE.md |
| 解码 | 100 t/s（MTP3） | 不开 MTP 25.5 t/s；**MTP3 48.8 t/s**（接受 78.9%，2.368 token/轮） | RESUME.md、MTP.md |
| 上下文 | 128k | 跑通到 32k（服务默认 32768），128k 未做验收 | FORMAT.md §9 |

**没达标的项如实写在上面**，别按目标数字验收。

---

## 1. 环境要求

### 硬件

| 项 | 要求 | 本机实测 |
|---|---|---|
| DCU | 海光 K100_LC，架构 **`gfx926`** | 120 CU @1270MHz，64GB HBM |
| 设备节点 | `/dev/kfd`、`/dev/dri` 存在 | 有 |
| 驱动运行库 | 宿主 `/opt/hyhal`（`/opt/hyhal/bin/hy-smi` 能出表） | driver 包 `rock-5.7.1 6.2.35`（`dpkg -l \| grep -i hyhal`） |
| 宿主内存 | ≥8GB（转换是 mmap 流式，常驻 <100MB；参考实现要 ~7GB） | 7GB 也能跑完 |
| 磁盘 | 见下方预算 | — |

> 非 `gfx926`（例如没有 `v_dot8_i32_i4` 的卡）**编译能过、跑不通**：核心算子直接内联这条指令。

### 磁盘预算（不含 Docker 自身开销）

| 内容 | 大小 |
|---|---|
| DTK 容器镜像 | 35.3GB |
| 源权重 `model.safetensors` | 22.57GB |
| MTP 头 `model_mtp.safetensors` | 0.85GB |
| 小文件（tokenizer/config/模板） | ≈27MB |
| 转换产物 RT4（主模型 + MTP） | 14.14GB |
| 视觉塔 RT4（独立文件） | 0.93GB |
| **合计** | **≈74GB**（另外每个 `build/rt` 只 0.4MB） |
| 可选：离线包 `dist/`（应用+权重+驱动）/ 镜像 tar | +15.5GB / +35GB |

### 软件

* Docker（当前用户在 `docker` 组，或可用 `sudo`）；本项目所有编译/运行都**在容器里**，宿主机不需要 DTK 工具链。
* DTK 镜像（官方，35.3GB）：
  `harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm0.18.1-ubuntu22.04-dtk26.04-py3.10-20260810-qwen3.8`
  容器内实测：`hipcc` = dcc 25.10.0-0（clang 17）、`torch` 2.10.0、`tokenizers`/`transformers`/`fastapi`/`uvicorn`/`jinja2` 齐备。
* 容器以宿主 uid/gid 运行，且**镜像里没有 uid 1000 的用户**，所以脚本显式传了
  `-e HOME=/tmp -e USER=… -e LOGNAME=…`（否则 `torch`/`getpass.getuser()` 会炸）。
  这些都在 `scripts/env.sh` 里，不用手工处理。

---

## 2. 取源码

```bash
git clone <你的仓库地址> K100LC-RT4
cd K100LC-RT4
```

目录：`src/`（运行时：图执行 + MTP + 引擎协议）、`kernels/`（算子内核与独立基准）、
`bench/`（早期算力/带宽/GEMM 原型）、`tools/`（权重转换与验证）、`scripts/`（环境、编译、
基准、服务）、`docs/`（设计与实测报告）、`web/`（自带网页）。

---

## 3. 拉容器镜像

在线（能访问 harbor）：

```bash
docker login harbor.sourcefind.cn:5443         # 如需账号，见海光开发者平台
docker pull harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm0.18.1-ubuntu22.04-dtk26.04-py3.10-20260810-qwen3.8
```

离线：拿到 `rt-dtk26.04-qwen3.8.tar` 后 `docker load -i rt-dtk26.04-qwen3.8.tar`。

驱动：`bash scripts/make_dist.sh` 会把 DCU 驱动一起打进 `dist/K100LC-RT4-offline/driver/`
（修正版安装包 + 预编译 `/usr/local/hyhal` + udev/modprobe/服务 + 现场快照 `manifest.txt`），
安装步骤与三个坑见包内 `driver/INSTALL.md`：必须用「内核68修正」包、装完把 GRUB 默认项
锁回编译时的内核、以后升级内核要重装。

自检（会自动挂载项目到 `/rt`、source DTK 环境）：

```bash
bash scripts/dsh.sh 'hipcc --version | head -2; python3 -c "import torch; print(torch.__version__)"'
```

> `scripts/env.sh` 里 `sudo_rt()` 优先用 `$RT_ROOT/.askpass.sh`（本机免密），找不到就退回
> 普通 `sudo`；`.askpass.sh` 里有明文口令，**已被 `.gitignore` 排除，不要提交**。

---

## 4. 准备模型权重

一键（幂等，缺什么下什么，然后自动转换）：

```bash
bash scripts/convert_weights.sh            # 下载 + 转换
bash scripts/convert_weights.sh --check    # 只校验已有产物
```

下载源是魔搭镜像仓库 `unsloth/Qwen3.8-27B-NVFP4`；22.57GB 的单文件由
`tools/fetch_par2.py` 用 **16 连接断点续传**（HF 直连会被 302 到被墙的 CDN，实测 200KB/s）。

也可以手工放好（文件名与位置必须一致：`models/Qwen3.8-27B-NVFP4/`）：

| 文件 | 字节数 | sha256 |
|---|---|---|
| `model.safetensors` | 22568192096 | `c473512c70eace07e2256fe9fd76596ac03e3295bee7d54cfb72676416afcc05` |
| `model_mtp.safetensors` | 849400392 | `1d8268aa85ace093a561e3e7b63b9d390dac1cd55a90cd55b5ec509c3c9da9fe` |
| `config.json` | 22564 | `1b3c71868d1299e52df6fc907deb202d5132b1ef0f72aae0ef6d15185dd53a5c` |
| `generation_config.json` | 214 | `d0d0ed2e37cdfafef4a5067d5ea2407b05f4fb50526e47c008a5b235d50240fb` |
| `tokenizer.json` | 19989325 | `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523` |
| `vocab.json` | 6722759 | `ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003` |
| `chat_template.jinja` | 9993 | `12827f24b742ea4e80cdc12dbcf9622227056b9f797252a3149263d4f9aaadce` |
| `model.safetensors.index.json` | 164371 | `429430e1b9e65b2cb98eff8cd10a06e70a09cee89c48487a3914684aeb6df57f` |

运行时真正读取的只有：RT4 权重 + manifest、`tokenizer.json`、`chat_template.jinja`。
`config.json` 只被 `tools/ref_*.py` 参考实现用；`vocab.json` 现在没人读（保留是为了对齐源快照）。

---

## 5. 转换：NVFP4/FP8 safetensors → RT4

```bash
bash scripts/convert_weights.sh
```

它做三件事：校验源 sha256 → `gcc -O2 tools/convert.c -lm` → 分别转换主模型与 MTP 头。

* 为什么必须转：源的 NVFP4(E2M1)+FP8 块尺度在本卡只能走 `v_dot4_i32_i8`（37.9 TMAC/s），
  1000 t/s 预填充需要它的 137%；统一成 **int4 + 每 128 组 f16 尺度** 才能走
  `v_dot8_i32_i4`（75.5 TMAC/s）。
* 反量化公式（`docs/CONVERT-VERIFY.md` 有逐张量核对）：
  `NVFP4: w = e2m1(code) * fp8_e4m3(block_scale) / weight_global_scale`（块尺度存储前乘过 global scale，要除回来）
  `FP8:   w = fp8_e4m3(code) * bf16(channel_scale)`
* 耗时与产物（本机实测，单文件 mmap 流式，宿主常驻 <100MB）：

| 产物 | 字节数 | sha256（参考值） |
|---|---|---|
| `rt4/qwen38_27b.rt4` | 13912487936 | `aa1a266c9e58842b12421214f52b8d6f350f1f52911d683394c3ff6d002e209e` |
| `rt4/qwen38_27b.rt4.json` | 179144 | `cd1fc03c2ff3c2772cd34db5ef858e76cf70e0a85a351059016ded731ea81a1e` |
| `rt4/qwen38_27b_mtp.rt4` | 219056128 | `d9bbea26758a5f5a6e1f7e88bfdbd743ee583fc048180770c111f0d5ff9e769d` |
| `rt4/qwen38_27b_mtp.rt4.json` | 2903 | `2a34e7447ba83e50e800bbc23f6f80e5248b24c429ac8b50cfdf3da7ed840600` |

转换一次约 **4 分 37 秒**，851 张量，逐张量 `relerr`/`src_rms` 写进 `rt4/convert.log`。
视觉塔单独转成 RT4（线性层 f16，norm/bias/pos f32）：

```bash
bash scripts/prepare_vision.sh      # → rt4/qwen38_27b_vision.rt4(.json)
```

> 注意：**不要**用 `kv_scales.json`（源模型附带、但本运行时不用；KV 现在是内核自算尺度的
> int4 打包格式）。

---

## 6. 编译运行时

```bash
bash scripts/build_rt.sh            # 5 个编译单元 → build/rt，实测约 15 秒
FLAGS="-DA8_ROWS=4" bash scripts/build_rt.sh   # 需要覆盖内核调参时
```

产物 `build/rt`（约 0.4MB）同时是 CLI 与引擎（`--engine` 走 stdin/stdout 行协议）。

---

## 7. 跑通

```bash
# 命令行对话（容器里跑）
bash scripts/dsh.sh 'python3 scripts/chat.py --prompt "用一句话介绍你自己" --n 64 --temp 0'

# OpenAI 兼容服务 + 自带网页：http://<本机IP>:8080/
bash scripts/serve.sh
PORT=8080 CTX=32768 MTP_N=3 bash scripts/serve.sh   # 显式指定
NO_MTP=1 bash scripts/serve.sh                      # 完全不加载 MTP 权重
bash scripts/serve.sh --stop
```

图片默认走本地 RT4 视觉塔；`build/rt` 启动时加载 `rt4/qwen38_27b_vision.rt4`。
如果只想纯文本部署，可以不跑 `prepare_vision.sh` 或设置 `RT_VISION_MODE=off`。

首次加载 13.9GB 权重约 **20 秒**。请求体可带 `"mtp": 0..3` 按请求覆盖草稿数。

---

## 8. 基准复现

> 先看一眼这条：**解码速度不是定值**，MTP3 的收益等于草稿接受率，而接受率随 prompt
> 剧烈变化（本机实测 21%~100%）。下表既是期望值也是敏感性说明，报数字时请连接受率
> 一起报（引擎在 stderr 打印 `MTP 统计：轮数…接受…`）。

```bash
bash scripts/dsh.sh 'python3 scripts/bench_rt.py'                        # 预填充 + 解码总表
bash scripts/dsh.sh 'python3 scripts/dec_bench.py 48'                    # 解码墙钟（默认 MTP3）
bash scripts/dsh.sh 'RT_NO_MTP=1 python3 scripts/dec_bench.py 48'        # 纯逐 token 对照
bash scripts/dsh.sh 'python3 scripts/dec_bench_ctx.py 60,1000,4000,8000' # 解码 vs 上下文
bash scripts/run_fa.sh                                                   # int4 FA 正确性 + 128k 吞吐
bash scripts/run_kernel.sh                                               # 单算子（GEMV）基准
```

期望值（同一张卡；±5% 属正常波动，注意机器上是否有别人的任务）。
**下面这些是交付前在这台机器上按左边的命令逐条跑出来的**：

| 命令 | 实测 |
|---|---|
| `scripts/bench_rt.py` 预填充 | 64 / 417 / 1649 token → **191 / 370 / 415 t/s**（越长越接近满速） |
| `scripts/bench_rt.py` 解码 temp=0 | **34.1 t/s**（接受率 42.9%） |
| `scripts/bench_rt.py` 解码 temp=0.7 | 24.1 t/s |
| `scripts/dec_bench.py 48`（中文短 prompt） | MTP3 **25.2 t/s**（接受率仅 21.4%，投机没收益）；`RT_NO_MTP=1` 26.2 t/s |
| `scripts/dec_bench_ctx.py 60,1000,4000` | **57.4 / 57.9 / 56.0 t/s**（每档只生成 16 token，样本小） |
| `run_fa.sh bench 1024`（本次直接跑内核） | 折算 16 层 1024-token 预填充 **978 t/s**（仅注意力部分） |
| `run_fa.sh check 4096` | 内核 vs 精确参考 rel_rms **2.8e-06**、cos **1.000000000** |

**解码速度不是定值：MTP3 的收益等于草稿接受率，而接受率随 prompt 剧烈变化。**
同一台机器、同一份权重、同一天：

| prompt | MTP 接受率 | MTP3 解码 |
|---|---|---|
| 中文短 prompt（`dec_bench.py`） | 21.4% | 25.2 t/s |
| 英文历史文本（`bench_rt.py`） | 42.9% | 34.1 t/s |
| 英文 filler（`dec_bench_ctx.py 60`） | 100% | 57.0 t/s |
| 立项时的长 prompt 测试（RESUME.md 记录） | 78.9% | 48.8 t/s |

所以 README/RESUME 里的 **48.8 t/s 是特定 prompt 下的最好值**，不是任何 prompt 都能复现的承诺；
报数字时请连接受率一起报（引擎会在 stderr 打印 `MTP 统计：轮数…接受…`）。
不接受投机时（`RT_NO_MTP=1` 或 `"mtp":0`）解码稳定在 **25~26 t/s**，与 prompt 无关。

计时口径很重要：`hipEventRecord` 本身每次约 **14µs**，会把小内核放大，
所以墙钟基准都用 Python 侧计时（见 `docs/PERF-DECODE.md` 第 2 节）；`RT_PROF=1`
只用于看**各阶段占比**。

常用环境变量：`RT_PROF=1`（分阶段计时）、`RT_MTP_N`、`RT_NO_MTP`、`RT_ACT4=1`（退回
int4 激活，只用于性能对照）、`RT_GDN_OLD=1` / `RT_ATTN_OLD=1`（旧内核 A/B）、
`RT_VERIFY_SEQ=1`（MTP 验证批退回「逐行算」的慢版本，等价性 A/B 用）、
`RT_NSPLIT=<n>`（强制解码注意力 split 数）、`RT_DUMP_BUF=1` + `--dump`（中间量落盘）。

---

## 9. 正确性与等价性验收

按顺序跑，任一不过就不要相信后面的数字：

| # | 命令 | 判据 | 本机实测 |
|---|---|---|---|
| 1 | `bash scripts/test_tools.sh` | 合成 NVFP4+FP8 用例转换后逐元素一致 | 通过（差异只有整数格平局舍入） |
| 2 | `bash scripts/dsh.sh 'python3 tools/verify_plain.py /rt/models/Qwen3.8-27B-NVFP4/model.safetensors /rt/models/Qwen3.8-27B-NVFP4/rt4/qwen38_27b.rt4 /rt/models/Qwen3.8-27B-NVFP4/rt4/qwen38_27b.rt4.json'` | 所有非量化(f16/f32)张量与源一致 | 449/449 |
| 3 | `bash scripts/dsh.sh 'python3 tools/spot_check_rt4.py /rt/models/Qwen3.8-27B-NVFP4/model.safetensors /rt/models/Qwen3.8-27B-NVFP4/rt4/qwen38_27b.rt4 model.language_model.layers.0.mlp.gate_proj.weight'` | 真实量化张量反量化后与源一致 | 通过 |
| 4 | `bash scripts/dsh.sh 'python3 tools/mtp_rolltest.py'` | MTP 验证批+回滚 vs 逐 token 前向的 logits | keep=1..4 全部 **max diff = 0** |
| 5 | `bash scripts/run_fa.sh check 4096` | int4 FlashAttention vs 定点参考 | 5e-08 ~ 3e-06 |
| 6 | `bash scripts/dsh.sh 'python3 scripts/attn_decode_check.py 3 2100'`（含 dump + 复核） | 解码注意力 vs fp64 单行 softmax | <1e-2（即 int4 量化误差量级） |
| 7 | 同一 prompt、`seed` 固定，跑 `scripts/chat.py` 对比 `--no-mtp` 与 MTP3 的 token 序列 | 贪心输出应逐 token 一致 | 一致（48/64/96 token 都验过） |
| 8 | （可选，需自备 BF16 GGUF，放进 `models/gguf/`）`bash scripts/dsh.sh 'python3 tools/verify_vs_gguf.py --check /rt/models/Qwen3.8-27B-NVFP4/rt4/qwen38_27b.rt4.json /rt/models/gguf/Qwen3.8-27B-BF16-0000{1,2}-of-00002.gguf 25'` | 反量化公式与 BF16 原模型 rms 一致 | 通过 |
| 9 | `bash scripts/dsh.sh 'python3 tools/test_attachments.py'` | 文档提取、多模态消息、HTTP 附件路由 | 通过 |
| 10 | 启动服务后上传写着 `7788` 的图片，问“识别四位数字”；再一次上传 `12`、`34` 两张图 | 本地视觉塔回答 `7788` / `12\n34` | 通过 |
| 11 | 开启 `skills` 让模型创建文件，再在同一 `conversation_id` 下读取并创建后续文件；`GET /v1/files` | 多轮技能调用、文件下载/预览 | 通过 |
| 12 | `bash scripts/run_fa_rows.sh` | 验证批注意力「一次算完 4 行」vs「逐行」逐位相同，且同输入可重复 | 全部 max diff = 0 |
| 13 | `bash scripts/serve.sh` 后按网页的设置发请求（`skills` 开/关 × 思考开/关） | 网页流式：正式回答首字 <7s 且分片递增；技能开时工具轮有 `round_reset` + `skill` 事件，最终回答继续流式；`RT_DEBUG_SKILL=1` 可打印每轮 token/分片数 | 通过（技能开+关思考：67 片；技能开+低思考：思考 100 片 + 回答 48 片；技能关：39 片） |
| 14 | 长提示词（约 1.9 万 token）+ `max_tokens: 128000` 走 `/v1/chat/completions` | `max_tokens` 按剩余上下文收窄、返回 200（不再 413）；只有 prompt 本身超上下文才 413（20 万 token 用例） | 通过 |
| 15 | 发一条会长时间生成的请求，6 秒后断开连接（等于按网页的「停止」），紧接着再发一条 | 引擎在 `END ... stopped` 收尾，服务立刻可用、输出不串线 | 通过（后续请求 0.4~1.4 s 内正常回答） |

`RT_BF16_GGUF_DIR` 指向 BF16 GGUF 所在目录即可跑第 8 项（不设就跳过，不影响前 7 项）。

---

## 10. 已知的坑（都是实测踩出来的，先看能省几小时）

1. **NF4/FP8 的 global scale 是「乘过再存」**：反量化要**除**回来，方向搞反是 1e8 级误差；
   结论靠 `tools/verify_vs_gguf.py` 与 BF16 原模型的 rms 对照定下来。
2. **BF16 源写成 f32 时不能按 4 字节原样搬**：`norm`/`A_log`/`dt_bias`/`conv1d`/`ssm_norm`
   会整体错位（残差被放大 10 倍）。`tools/verify_plain.py` 专门查这一路。
3. **MTP 验证批不能用预填充 FlashAttention**：预填充的 P 是 4bit，单 token 解码是 fp32 P，
   混用会让轨迹分叉。现在 `T≤4` 且 `snap_mode` 时逐行走解码内核。
4. **V 的 tile 尺度不能跨 64-key tile 回滚**，**卷积状态快照差一位**（`t-(K-2)+r`）：
   这两个坑都会让长序列悄悄漂移，靠 `tools/mtp_rolltest.py` 守住。
5. **`gfx926` 每 block 默认线程上限 256**，写 1024 线程的 kernel 会启动失败。
6. **容器里没有 uid 1000 用户** → `torch` 调 `getpass.getuser()` 报错，必须传 `USER/LOGNAME`。
7. **22.57GB 单文件下载必须断点续传**，用 `tools/fetch_par2.py`（多连接 + sha256）。
8. **单 block 的 1MB 归约是延迟受限的**：把 logits argmax 搬设备侧反而更慢
   （0.82 → 2.30 ms/轮，见 `docs/MTP.md` §7），要搬就得多 block 两级归约。

---

## 11. 复现记录模板（发 issue / PR 时请附上）

```bash
/opt/hyhal/bin/hy-smi | head -8                 # 卡与显存
dpkg -l | grep -i hyhal                         # 驱动版本
docker images --digests | grep dtk26.04         # 镜像 digest
bash scripts/dsh.sh 'hipcc --version | head -2' # 工具链
sha256sum models/Qwen3.8-27B-NVFP4/rt4/qwen38_27b.rt4
```

---

## 12. 第三方内容与许可

* 权重（`unsloth/Qwen3.8-27B-NVFP4`）与 BF16 GGUF 属第三方，遵循其原始许可，**不入库**，
  请自行下载；DTK 容器镜像同理（海光官方 harbor）。
* 本仓库的源码许可：**Apache License 2.0**，见 [LICENSE](LICENSE)；第三方归属见 [NOTICE](NOTICE)。
