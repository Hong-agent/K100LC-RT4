#!/bin/bash
# 在 DTK 容器里跑一条命令（自动 source DTK 环境、挂载项目到 /rt）。用法：
#   bash scripts/dsh.sh '命令...'
#   bash scripts/dsh.sh -i          # 交互式
set -euo pipefail
source "$(dirname "$0")/env.sh"
if [ "${1:-}" = "-i" ]; then
  exec sudo_rt docker run -it "${RT_DOCKER_ARGS[@]}" -v "$RT_ROOT":/rt -w /rt "$RT_IMG" \
    bash -c 'source /opt/dtk/env.sh >/dev/null 2>&1; exec bash'
fi
sudo_rt docker run "${RT_DOCKER_ARGS[@]}" -v "$RT_ROOT":/rt -w /rt "$RT_IMG" \
  bash -c 'source /opt/dtk/env.sh >/dev/null 2>&1; '"$*"
