# 不经过 DTK 容器，在 K100_LC 主机上直接运行模型

## 结论

**运行可以脱离 Docker / DTK 工具链，但不能脱离 DTK 的 HIP 运行库。**

`build/rt` 是预先用 `hipcc --offload-arch=gfx926` 编好的 ELF，链接表里写着：

```text
NEEDED  libgalaxyhip.so.5
RUNPATH /opt/dtk/hip/lib:/opt/dtk/lib
```

主机上没有 `/opt/dtk`，所以直接执行会报 `libgalaxyhip.so.5 => not found`。
把这两个运行库从本地 DTK 镜像抽出来，再用 `LD_LIBRARY_PATH` 指过去，模型就能在
主机上直接跑；HSA 运行时和内核驱动仍然用主机 `/opt/hyhal` 那一套。

实际需要的新增运行库：

| 文件 | 来源 | 大小 | 作用 |
|---|---|---|---|
| `libgalaxyhip.so.5` + `hipkernel.bin.gfx926` | `/opt/dtk/hip/lib` | 约 12MB | HIP 用户态运行时 + gfx926 内置包 |
| `libamd_comgr.so.2` | `/opt/dtk/dcc/comgr/lib` | 约 152MB | HIP 初始化 / code object 处理；缺了会报 `invalid device ordinal` |
| Python 依赖 | `/usr/local/lib/python3.10/dist-packages` | 约 27MB | `tokenizers` / `jinja2` / `fastapi` / `uvicorn` 等 |

## 获取运行库

两种方式，任选一种：

**A. 目标机没有 DTK 镜像：从 GitHub Release 下载**

```bash
bash scripts/fetch_host_runtime.sh
bash scripts/make_host_runtime.sh --check
```

Release：
<https://github.com/Hong-agent/K100LC-RT4/releases/tag/host-runtime-2026-09-27-r2>

压缩包 `K100LC-RT4-host-runtime-2026-09-27-r2.tar.zst` 约 53MB：

```text
内容：build/rt + runtime/dtk-libs + runtime/py
sha256 618a750de04fabfdb360f2a3c6560495508c02c27387f4a3a6e69ba587505824
```

`fetch_host_runtime.sh` 会下载压缩包和 `.sha256`、校验后解压出 `build/rt` 和
`runtime/`；不需要目标机先编译。

**B. 本机已有 DTK 镜像：现场抽取**

`scripts/make_host_runtime.sh` 直接从本机 DTK 镜像里 `tar` 抽取，不需要公网；
Python 依赖也从同一镜像抽，避免在主机装 pip / venv / 系统包。

## 快速使用

```bash
cd K100LC-RT4
bash scripts/fetch_host_runtime.sh       # 目标机：从 Release 下载预提取运行库
# 或：bash scripts/make_host_runtime.sh  # 本机已有 DTK 镜像时现场抽取

bash scripts/chat_host.sh --prompt "你好" --n 32 --temp 0
bash scripts/serve_host.sh               # 默认监听 80，局域网只输 IP 即可
bash scripts/host_dsh.sh 'python3 scripts/bench_rt.py'
```

网页和接口默认显示为 `http://<本机IP>/`、`http://<本机IP>/v1`。如果 80 端口被别的
程序占用，可以用 `PORT=8080 bash scripts/serve_host.sh` 改到 8080。

服务停止：

```bash
bash scripts/serve_host.sh --stop
```

`runtime/` 已加入 `.gitignore`，不会进仓库；它只包含从镜像里抽出的第三方运行库。

## 已验证的结果

在本机（Ubuntu 22.04 / 内核 6.8 / DCU 驱动已加载）实测：

* 一次性前向：3 token，加载 18.0s，前向 0.07s，输出 top-5 logits 正常。
* 引擎协议：`READY 18.0`，`PREFILL 3 tokens` 约 50ms；
  不开 MTP 连续生成 12 token，`END 12 440.4`。
* MTP3：生成 16 token，`END 16 393.7`，接受率 58.8%（短提示）。
* 主机端 `chat.py`：`你好` 预填充 53 token / 164 t/s，正常出中文。
* 主机端 `serve.py`：`/v1/capabilities` 正常，`/v1/chat/completions`
  15 token 约 26.7 t/s（短提示）。

## 边界

这不是“完全没有任何 DTK 代码”。

1. **编译仍然需要 DTK。** `hipcc`、HIP 头文件、gfx926 编译后端都在 DTK 里；
   主机 `/opt/hyhal` 只有驱动、HSA 运行库和 `hy-smi` 等工具。
   `/opt/hyhal/bin/hycc` 也需要 `HIP_PATH`/完整 ROCm 目录，单独拿 hyhal 不能编译。
2. **运行依赖 DTK 的 `libgalaxyhip` 与 `libamd_comgr`。** 现有 `build/rt` 是
   对着这套 ABI 编的；要做到连这两个 `.so` 也不用，需要把算子改成直接对
   `/opt/hyhal` 的 HSA 接口重新编译/装载，那是另一条较大的改造路线。
3. **本地视觉塔默认在 CPU 上跑。** 主机服务设置 `RT_VISION_DEVICE=cpu`，只依赖
   `numpy` / `PIL` 和视觉 RT4 权重，不需要 torch/transformers，也不会把视觉塔
   加载到 DCU；典型 91-token 图片编码约 4 秒。想改回 GPU/HIP 可设
   `RT_VISION_DEVICE=gpu`，此时需要容器里的 torch/transformers 预处理环境。
   离线一键包还会带独立 Python 3.10、numpy 和 Pillow（`runtime/python` +
   `runtime/py`），目标机不需要预装 Python。
4. **PDF 抽取是基础版。** 容器里有 `pypdf` 时走完整解析；主机抽取包没带它，
   `serve.py` 会自动退回内置提取器。

## 出问题时先查这三件事

```bash
/opt/hyhal/bin/hy-smi                    # 能不能看到 DCU 0
ls -l /dev/kfd /dev/dri/renderD*         # 设备节点在不在
bash scripts/make_host_runtime.sh --check
```

`--check` 会检查 `build/rt` 的动态依赖能不能解析、Python import 能不能通过。
