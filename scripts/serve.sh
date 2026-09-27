#!/bin/bash
# 启动自研运行时的 OpenAI 兼容服务（局域网可直连）。接口与常见客户端一致。
#
#   bash scripts/serve.sh              # 8080 端口
#   PORT=8080 CTX=32768 bash scripts/serve.sh
#   bash scripts/serve.sh --stop
set -euo pipefail
source "$(dirname "$0")/env.sh"
PORT="${PORT:-8080}"
# 上下文默认 128k（131072 = 128k + prompt 余量），这样 max_tokens=128000 才装得下
CTX="${CTX:-131072}"
DEFAULT_MAX_TOKENS="${RT_DEFAULT_MAX_TOKENS:-128000}"
# 保留 MTP_N，也兼容文档里写过的 RT_MTP_N。
MTP_N="${MTP_N:-${RT_MTP_N:-3}}"
NAME="${NAME:-rt-serve}"
NO_MTP_FLAG=""
if [ "${NO_MTP:-0}" != "0" ]; then NO_MTP_FLAG="--no-mtp"; fi

sudo_rt docker rm -f "$NAME" >/dev/null 2>&1 || true
if [ "${1:-}" = "--stop" ]; then echo "已停止 $NAME"; exit 0; fi

sudo_rt docker run -d --name "$NAME" --restart unless-stopped \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e USER=t -e LOGNAME=t \
  -e RT_MODEL_DIR=/rt/models/Qwen3.8-27B-NVFP4 \
  -e RT_VISION_BASE_URL="${RT_VISION_BASE_URL:-}" \
  -e RT_VISION_MODEL="${RT_VISION_MODEL:-}" \
  -e RT_VISION_API_KEY="${RT_VISION_API_KEY:-}" \
  -e RT_VISION_MODE="${RT_VISION_MODE:-auto}" \
  -e RT_VISION_RT4="${RT_VISION_RT4:-}" \
  -e RT_VISION_DTYPE="${RT_VISION_DTYPE:-float16}" \
  -e RT_VISION_MAX_PATCHES="${RT_VISION_MAX_PATCHES:-4096}" \
  -e RT_VISION_MIN_PIXELS="${RT_VISION_MIN_PIXELS:-3136}" \
  -e RT_VISION_MAX_PIXELS="${RT_VISION_MAX_PIXELS:-1003520}" \
  -e RT_VISION_TIMEOUT="${RT_VISION_TIMEOUT:-120}" \
  -e RT_VISION_MAX_TOKENS="${RT_VISION_MAX_TOKENS:-768}" \
  -e RT_VISION_PROMPT="${RT_VISION_PROMPT:-}" \
  -e RT_MAX_UPLOAD_MB="${RT_MAX_UPLOAD_MB:-25}" \
  -e RT_WORKSPACE_DIR="${RT_WORKSPACE_DIR:-}" \
  -e RT_MAX_FILE_KB="${RT_MAX_FILE_KB:-2048}" \
  -e RT_MAX_WORKSPACE_MB="${RT_MAX_WORKSPACE_MB:-64}" \
  -e RT_MAX_SKILL_ROUNDS="${RT_MAX_SKILL_ROUNDS:-6}" \
  -e RT_DEBUG_SKILL="${RT_DEBUG_SKILL:-}" \
  --device /dev/kfd --device /dev/dri -v /opt/hyhal:/opt/hyhal:ro \
  -v "$RT_ROOT":/rt -w /rt --network host "$RT_IMG" \
  bash -c "source /opt/dtk/env.sh >/dev/null 2>&1; exec python3 scripts/serve.py --port $PORT --ctx $CTX --default-max-tokens $DEFAULT_MAX_TOKENS --mtp-n $MTP_N $NO_MTP_FLAG"

LAN_IP="$(ip -4 -o addr show scope global 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | grep -v '^172\.17\.' | head -1)"
echo "容器 $NAME 已启动（加载 13.9GB 权重约 20 秒）"
echo "  网页 : http://${LAN_IP:-<本机IP>}:$PORT/        (简易对话界面)"
echo "  接口 : http://${LAN_IP:-<本机IP>}:$PORT/v1   (OpenAI 兼容)"
echo "  日志 : sudo docker logs -f $NAME"
echo "  停止 : bash scripts/serve.sh --stop"
