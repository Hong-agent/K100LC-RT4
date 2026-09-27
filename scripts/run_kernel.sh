#!/bin/bash
# 在 DTK 容器里编译并跑一个自研算子（默认 W4A4 解码 GEMV）。
#
# 用法:
#   bash scripts/run_kernel.sh [kernel.hip] [tensor] [M] [reps]
#   FLAGS="-DROWS=2" bash scripts/run_kernel.sh gemv_w4a4.hip
#
# 例:
#   bash scripts/run_kernel.sh                                   # W4A4, gate_proj, M=4
#   bash scripts/run_kernel.sh gemv_int4.hip 'lm_head.weight' 1  # W4A8, lm_head, M=1
set -euo pipefail
source "$(dirname "$0")/env.sh"

KERNEL="${1:-gemv_w4a4.hip}"
TENSOR="${2:-model.language_model.layers.0.mlp.gate_proj.weight}"
M="${3:-4}"
REPS="${4:-30}"
FLAGS="${FLAGS:-}"
NAME="${KERNEL%.hip}"

sudo_rt docker run "${RT_DOCKER_ARGS[@]}" \
  -v "$RT_ROOT/kernels":/kernels -v "$RT_MODEL_DIR/rt4":/rt:ro -w /kernels "$RT_IMG" \
  bash -c "source /opt/dtk/env.sh >/dev/null 2>&1
    hipcc -O3 --offload-arch=$RT_ARCH $FLAGS -o /tmp/$NAME $KERNEL || exit 1
    /tmp/$NAME '$TENSOR' $M /rt/qwen38_27b.rt4 /rt/qwen38_27b.rt4.json $REPS"
