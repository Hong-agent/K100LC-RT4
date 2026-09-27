#!/bin/bash
# 在“主机直跑”环境里执行一条命令（对应容器版 scripts/dsh.sh）。
#
#   bash scripts/host_dsh.sh 'python3 scripts/bench_rt.py'
#   bash scripts/host_dsh.sh -i
set -euo pipefail
source "$(dirname "$0")/host_env.sh"
cd "$RT_ROOT"

if [ "${1:-}" = "-i" ]; then
  exec bash
fi
if [ "$#" -eq 0 ]; then
  echo "用法: bash scripts/host_dsh.sh '命令...' 或 bash scripts/host_dsh.sh -i" >&2
  exit 2
fi
exec bash -c "$*"
