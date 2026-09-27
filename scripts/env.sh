#!/bin/bash
# 自研运行时的共享环境变量。其它脚本都 source 这个文件。
# 用 `source scripts/env.sh` 后在当前 shell 也能直接用。

export RT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"

# DTK 容器镜像（海光官方 DTK 26.04 + 编译链；本机没有 DTK 工具链，一切编译/运行都在容器里）
export RT_IMG="${RT_IMG:-harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm0.18.1-ubuntu22.04-dtk26.04-py3.10-20260810-qwen3.8}"

# sudo 非交互密码助手（本机 su 密码，权限 700）。
# 离线包里不带这个文件（里面有明文密码），找不到就退回普通 sudo——
# 目标机上要么已经 NOPASSWD，要么会正常提示输密码。
if [ -z "${SUDO_ASKPASS:-}" ] && [ -x "$RT_ROOT/.askpass.sh" ]; then
  export SUDO_ASKPASS="$RT_ROOT/.askpass.sh"
fi

# 权重与产物
export RT_MODEL_DIR="${RT_MODEL_DIR:-$RT_ROOT/models/Qwen3.8-27B-NVFP4}"
export RT_RT4="${RT_RT4:-$RT_MODEL_DIR/rt4/qwen38_27b.rt4}"
export RT_RT4_JSON="${RT_RT4_JSON:-$RT_MODEL_DIR/rt4/qwen38_27b.rt4.json}"
export RT_MTP="${RT_MTP:-$RT_MODEL_DIR/rt4/qwen38_27b_mtp.rt4}"

# 可选：BF16 原模型（GGUF）所在目录，只用于 tools/verify_vs_gguf.py 这类
# 「反量化公式对不对」的交叉验证。不放就跳过那一步，不影响编译与运行。
export RT_BF16_GGUF_DIR="${RT_BF16_GGUF_DIR:-$RT_ROOT/models/gguf}"

# 编译目标固定 gfx926
export RT_ARCH="${RT_ARCH:-gfx926}"

# 容器公共参数
# 注意：容器里没有 uid 1000 的用户，torch/transformers 会调 getpass.getuser() 而炸，
# 所以显式给 USER/LOGNAME（getpass 优先读这两个环境变量）。
export RT_DOCKER_ARGS=(--rm --user "$(id -u):$(id -g)" -e HOME=/tmp
  -e USER="${USER:-t}" -e LOGNAME="${LOGNAME:-t}" -e TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor
  --device /dev/kfd --device /dev/dri -v /opt/hyhal:/opt/hyhal:ro)

sudo_rt() {
  if [ -n "${SUDO_ASKPASS:-}" ]; then sudo -A "$@"; else sudo "$@"; fi
}
