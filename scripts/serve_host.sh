#!/bin/bash
# 主机直跑的 OpenAI 兼容服务（对应容器版 scripts/serve.sh）。
#
#   bash scripts/serve_host.sh               # 默认 80 端口，局域网只输 IP 即可
#   PORT=18080 CTX=4096 MTP_N=0 bash scripts/serve_host.sh
#   bash scripts/serve_host.sh --stop
#
# 主机默认把视觉塔放在 CPU 上跑；配了外部视觉桥会自动用 external。
set -euo pipefail
source "$(dirname "$0")/host_env.sh"
cd "$RT_ROOT"

NAME=rt-serve-host
PIDFILE="$RT_HOST_RUNTIME/$NAME.pid"
LOGFILE="$RT_HOST_RUNTIME/$NAME.log"
PORT="${PORT:-80}"
CTX="${CTX:-131072}"
DEFAULT_MAX_TOKENS="${RT_DEFAULT_MAX_TOKENS:-128000}"
MTP_N="${MTP_N:-${RT_MTP_N:-3}}"
export RT_VISION_DEVICE="${RT_VISION_DEVICE:-cpu}"

NO_MTP_FLAG=""
if [ "${NO_MTP:-0}" != "0" ]; then NO_MTP_FLAG="--no-mtp"; fi

if [ -z "${RT_VISION_MODE:-}" ]; then
  if [ -n "${RT_VISION_BASE_URL:-}" ] && [ -n "${RT_VISION_MODEL:-}" ]; then
    export RT_VISION_MODE=external
  else
    export RT_VISION_MODE=auto
  fi
fi

if [ "${1:-}" = "--stop" ]; then
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    kill "$(cat "$PIDFILE")"
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      kill -0 "$(cat "$PIDFILE")" 2>/dev/null || break
      sleep 0.5
    done
    rm -f "$PIDFILE"
    echo "已停止 $NAME"
  else
    echo "$NAME 未运行"
  fi
  exit 0
fi

ensure_port_bindable() {
  [ "$PORT" -ge 1024 ] && return 0
  local current
  current="$(sysctl -n net.ipv4.ip_unprivileged_port_start 2>/dev/null || echo 1024)"
  [ "$current" -le "$PORT" ] && return 0

  echo "[serve_host] 允许非 root 绑定端口 $PORT（net.ipv4.ip_unprivileged_port_start=$PORT）"
  if [ "$(id -u)" = 0 ]; then
    sysctl -w "net.ipv4.ip_unprivileged_port_start=$PORT" >/dev/null
  elif [ -n "${SUDO_ASKPASS:-}" ]; then
    sudo -A sysctl -w "net.ipv4.ip_unprivileged_port_start=$PORT" >/dev/null
  else
    sudo sysctl -w "net.ipv4.ip_unprivileged_port_start=$PORT" >/dev/null
  fi
}
ensure_port_bindable

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "$NAME 已在运行，PID $(cat "$PIDFILE")（日志 $LOGFILE）" >&2
  exit 0
fi

mkdir -p "$RT_HOST_RUNTIME"
if command -v setsid >/dev/null 2>&1; then
  # setsid 让它脱离当前进程组；PID 在子 shell 里写，避免 setsid fork 后 $! 失真。
  setsid bash -c 'echo $$ >"$1"; shift; exec python3 "$@"' _ "$PIDFILE" \
    scripts/serve.py --port "$PORT" --ctx "$CTX" \
    --default-max-tokens "$DEFAULT_MAX_TOKENS" --mtp-n "$MTP_N" $NO_MTP_FLAG \
    >"$LOGFILE" 2>&1 </dev/null &
else
  nohup python3 scripts/serve.py --port "$PORT" --ctx "$CTX" \
    --default-max-tokens "$DEFAULT_MAX_TOKENS" --mtp-n "$MTP_N" $NO_MTP_FLAG \
    >"$LOGFILE" 2>&1 </dev/null &
  echo "$!" >"$PIDFILE"
fi
sleep 0.2
PID="$(cat "$PIDFILE" 2>/dev/null || true)"

for _ in $(seq 1 240); do
  if grep -q '\[serve\] 模型就绪' "$LOGFILE" 2>/dev/null; then
    break
  fi
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "服务启动失败，日志尾部：" >&2
    tail -40 "$LOGFILE" >&2 || true
    rm -f "$PIDFILE"
    exit 1
  fi
  sleep 0.5
done

LAN_IP="$(ip -4 -o addr show scope global 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | grep -v '^172\.17\.' | head -1)"
if [ "$PORT" = "80" ]; then HTTP_PORT=""; else HTTP_PORT=":$PORT"; fi
echo "$NAME 已启动（PID $PID，加载权重约 20 秒）"
echo "  网页 : http://${LAN_IP:-<本机IP>}${HTTP_PORT}/"
echo "  接口 : http://${LAN_IP:-<本机IP>}${HTTP_PORT}/v1"
echo "  日志 : tail -f $LOGFILE"
echo "  停止 : bash scripts/serve_host.sh --stop"
