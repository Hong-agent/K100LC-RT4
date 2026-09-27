#!/bin/bash
# 验证批解码注意力（fa_decode_rows_k）的算子级自检：
# 「一次算 4 行」必须与「逐行调用」逐位相同，而且内核本身必须可重复。
# 这是 MTP 等价性的地基（见 docs/MTP.md 第 4 节）：合成随机输入，不依赖权重。
#
#   bash scripts/run_fa_rows.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"

sudo_rt docker run "${RT_DOCKER_ARGS[@]}" \
  -v "$RT_ROOT":/rt -w /rt "$RT_IMG" \
  bash -c "source /opt/dtk/env.sh >/dev/null 2>&1
    hipcc -O3 --offload-arch=$RT_ARCH -o /tmp/fa_rows_check kernels/fa_rows_check.hip || exit 1
    /tmp/fa_rows_check"
