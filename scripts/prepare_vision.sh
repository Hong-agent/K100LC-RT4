#!/bin/bash
# 从源 safetensors 导出视觉塔为独立 RT4 文件（线性层 f16，norm/bias/pos f32）。
# 产物：models/Qwen3.8-27B-NVFP4/rt4/qwen38_27b_vision.rt4(.json)
#
#   bash scripts/prepare_vision.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"

MODEL_IN_CONTAINER=/models/Qwen3.8-27B-NVFP4
SRC="$RT_MODEL_DIR/model.safetensors"
CFG="$RT_MODEL_DIR/config.json"
OUT_DIR="$RT_MODEL_DIR/rt4"
OUT="$OUT_DIR/qwen38_27b_vision.rt4"

[ -s "$SRC" ] || { echo "缺少源权重：$SRC" >&2; exit 1; }
[ -s "$CFG" ] || { echo "缺少 config.json：$CFG" >&2; exit 1; }

sudo_rt docker run "${RT_DOCKER_ARGS[@]}" \
  -v "$RT_ROOT":/rt \
  -v "$RT_MODEL_DIR":$MODEL_IN_CONTAINER \
  -w /rt "$RT_IMG" \
  bash -c "source /opt/dtk/env.sh >/dev/null 2>&1
    python3 tools/convert_vision_rt4.py $MODEL_IN_CONTAINER/model.safetensors \
      $MODEL_IN_CONTAINER/rt4/qwen38_27b_vision.rt4"

echo "视觉塔 RT4 产物：$OUT"
ls -lh "$OUT" "$OUT.json"
