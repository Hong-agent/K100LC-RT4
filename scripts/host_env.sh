#!/bin/bash
# 主机直跑环境：不经过 Docker / DTK 工具链，只用本机 DCU 驱动 + 从 DTK 镜像抽出的
# 最小 HIP/COMGR 运行库 + 一份 Python 依赖。
#
# 用法：
#   source scripts/host_env.sh
#   ./build/rt --model ... --json ... --ids 1,2,3
#
# 先运行一次 `bash scripts/make_host_runtime.sh` 准备 runtime/。
# 这个文件不设置 set -e/-u，避免影响调用它的交互式 shell。

export RT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"

# 本机有项目自带的 sudo askpass 时，主机脚本也能非交互地设置低端口权限。
if [ -z "${SUDO_ASKPASS:-}" ] && [ -x "$RT_ROOT/.askpass.sh" ]; then
  export SUDO_ASKPASS="$RT_ROOT/.askpass.sh"
fi

# 主机直跑运行库目录（默认 gitignore，不随源码入库）
export RT_HOST_RUNTIME="${RT_HOST_RUNTIME:-$RT_ROOT/runtime}"
export RT_HOST_LIBS="${RT_HOST_LIBS:-$RT_HOST_RUNTIME/dtk-libs}"
export RT_HOST_PY="${RT_HOST_PY:-$RT_HOST_RUNTIME/py}"

# 与 scripts/env.sh 对齐的权重与产物路径
export RT_MODEL_DIR="${RT_MODEL_DIR:-$RT_ROOT/models/Qwen3.8-27B-NVFP4}"
export RT_RT4="${RT_RT4:-$RT_MODEL_DIR/rt4/qwen38_27b.rt4}"
export RT_RT4_JSON="${RT_RT4_JSON:-$RT_MODEL_DIR/rt4/qwen38_27b.rt4.json}"
export RT_MTP="${RT_MTP:-$RT_MODEL_DIR/rt4/qwen38_27b_mtp.rt4}"
export RT_VISION_RT4="${RT_VISION_RT4:-$RT_MODEL_DIR/rt4/qwen38_27b_vision.rt4}"

# `build/rt` 的 RUNPATH 写死为 /opt/dtk/{hip,lib}；主机没有 /opt/dtk，所以必须用
# LD_LIBRARY_PATH 把抽出的运行库放在前面。HSA 运行时来自 /opt/hyhal。
if [ -d "$RT_HOST_LIBS/hip" ] && [ -d "$RT_HOST_LIBS/comgr" ]; then
  export LD_LIBRARY_PATH="$RT_HOST_LIBS/hip:$RT_HOST_LIBS/comgr:/opt/hyhal/lib:/opt/hyhal/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
else
  echo "[host_env] 缺少主机运行库，先运行: bash scripts/make_host_runtime.sh" >&2
fi

# 只用主机自己的 Python；这层只补 tokenizers / jinja2 / fastapi / uvicorn 等
# 容器里已有的依赖，不碰系统 site-packages。
if [ -d "$RT_HOST_PY" ]; then
  export PYTHONPATH="$RT_HOST_PY${PYTHONPATH:+:$PYTHONPATH}"
fi

export USER="${USER:-$(id -un)}"
export LOGNAME="${LOGNAME:-$USER}"
export RT_ARCH="${RT_ARCH:-gfx926}"

if [ ! -x "$RT_ROOT/build/rt" ]; then
  echo "[host_env] 找不到可执行文件 $RT_ROOT/build/rt；编译仍需要 DTK/hipcc" >&2
fi
if [ ! -x /opt/hyhal/bin/hy-smi ]; then
  echo "[host_env] 找不到 /opt/hyhal/bin/hy-smi；主机的 DCU 驱动未安装或不完整" >&2
fi
