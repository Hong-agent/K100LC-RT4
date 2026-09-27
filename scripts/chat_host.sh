#!/bin/bash
# 主机直跑的命令行对话：与 scripts/chat.py 相同参数，但不经过 Docker。
#
#   bash scripts/chat_host.sh --prompt "你好" --n 32 --temp 0
#   bash scripts/chat_host.sh --prompt "你好" --n 32 --mtp-n 3
set -euo pipefail
source "$(dirname "$0")/host_env.sh"
cd "$RT_ROOT"
exec python3 scripts/chat.py "$@"
