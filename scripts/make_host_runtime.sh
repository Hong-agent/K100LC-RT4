#!/bin/bash
# 准备“主机直跑”所需的最小运行库，全部从本机已有的 DTK 镜像里抽取，不需要公网。
#
#   bash scripts/make_host_runtime.sh            # 抽取运行库 + Python 依赖
#   bash scripts/make_host_runtime.sh --check    # 只校验现有 runtime/
#   bash scripts/make_host_runtime.sh --native-only
#   bash scripts/make_host_runtime.sh --python-only
#
# 产物（默认 gitignore）：
#   runtime/dtk-libs/hip/    libgalaxyhip.so.5 + gfx926 内置包（约 12MB）
#   runtime/dtk-libs/comgr/  libamd_comgr.so.2（约 152MB）
#   runtime/py/              主机 Python 用的 tokenizers/fastapi/uvicorn 等（约 27MB）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/scripts/env.sh"

RUNTIME="${RT_HOST_RUNTIME:-$ROOT/runtime}"
LIBS="${RT_HOST_LIBS:-$RUNTIME/dtk-libs}"
PY="${RT_HOST_PY:-$RUNTIME/py}"
PYBIN="${RT_PYTHON:-python3}"
if [ -z "${RT_PYTHON:-}" ] && [ -x "$ROOT/runtime/python/bin/python3.10" ]; then
  export PYTHONHOME="$ROOT/runtime/python"
  export LD_LIBRARY_PATH="$ROOT/runtime/python/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  PYBIN="$ROOT/runtime/python/bin/python3.10"
fi

MODE=all
case "${1:-}" in
  --check) MODE=check ;;
  --native-only) MODE=native ;;
  --python-only) MODE=python ;;
  -h|--help|-help)
    sed -n '2,16p' "$0"
    exit 0
    ;;
  "") ;;
  *)
    echo "未知参数: $1" >&2
    exit 2
    ;;
esac

check_runtime() {
  local need_native="${1:-1}" need_python="${2:-1}"
  local ok=1
  if [ "$need_native" = 1 ]; then
    echo "[check] 动态库"
    if [ ! -x "$ROOT/build/rt" ]; then
      echo "  找不到 $ROOT/build/rt" >&2
      ok=0
    elif [ -d "$LIBS/hip" ] && [ -d "$LIBS/comgr" ]; then
      local missing
      missing="$(LD_LIBRARY_PATH="$LIBS/hip:$LIBS/comgr:/opt/hyhal/lib:/opt/hyhal/lib64" \
        ldd "$ROOT/build/rt" 2>/dev/null | awk '/not found/ {print $1}' || true)"
      if [ -n "$missing" ]; then
        echo "  缺少: $missing" >&2
        ok=0
      else
        echo "  build/rt 的动态依赖可解析"
      fi
    else
      echo "  尚未抽取 $LIBS" >&2
      ok=0
    fi
  fi

  if [ "$need_python" = 1 ]; then
    echo "[check] Python 依赖"
    if PYTHONPATH="$PY" "$PYBIN" -c 'import fastapi, h11, jinja2, multipart, numpy, PIL, pydantic, starlette, tokenizers, uvicorn' 2>/dev/null; then
      echo "  import 通过"
    else
      echo "  尚未抽取 $PY（或版本不匹配）" >&2
      ok=0
    fi
  fi

  if [ "$ok" = 1 ]; then
    echo "[check] OK"
    return 0
  fi
  return 1
}

if [ "$MODE" = check ]; then
  check_runtime 1 1
  exit $?
fi

if ! sudo_rt docker image inspect "$RT_IMG" >/dev/null 2>&1; then
  echo "本地没有 DTK 镜像: $RT_IMG" >&2
  echo "可以先拉取它，或指定已经有运行库的镜像: RT_IMG=... bash scripts/make_host_runtime.sh" >&2
  exit 1
fi

mkdir -p "$RUNTIME" "$LIBS/hip" "$LIBS/comgr" "$PY"

if [ "$MODE" = all ] || [ "$MODE" = native ]; then
  if [ -e "$LIBS/hip/libgalaxyhip.so.5" ] && [ -e "$LIBS/comgr/libamd_comgr.so.2" ]; then
    echo "[native] 已有，跳过"
  else
    echo "[native] 从镜像抽取 libgalaxyhip / hipkernel.gfx926 / libamd_comgr"
    sudo_rt docker run --rm --user "$(id -u):$(id -g)" "$RT_IMG" bash -lc '
      set -e
      shopt -s nullglob
      cd /opt/dtk/hip/lib
      files=()
      for p in libgalaxyhip.so* hipkernel.bin.gfx926 .hipInfo; do
        [ -e "$p" ] && files+=("$p")
      done
      tar -cf - "${files[@]}"
    ' | tar -C "$LIBS/hip" -xf -

    sudo_rt docker run --rm --user "$(id -u):$(id -g)" "$RT_IMG" bash -lc '
      set -e
      shopt -s nullglob
      cd /opt/dtk/dcc/comgr/lib
      files=(libamd_comgr.so*)
      tar -cf - "${files[@]}"
    ' | tar -C "$LIBS/comgr" -xf -
  fi
fi

if [ "$MODE" = all ] || [ "$MODE" = python ]; then
  if [ -e "$PY/tokenizers/__init__.py" ] && PYTHONPATH="$PY" \
      "$PYBIN" -c 'import fastapi, jinja2, numpy, PIL, tokenizers, uvicorn' >/dev/null 2>&1; then
    echo "[python] 已有，跳过"
  else
    echo "[python] 从镜像抽取容器里已验证过的版本"
    sudo_rt docker run --rm --user "$(id -u):$(id -g)" "$RT_IMG" bash -lc '
      set -e
      shopt -s nullglob
      cd /usr/local/lib/python3.10/dist-packages
      files=()
      for p in \
        annotated_types annotated_types-* anyio anyio-* click click-* \
        exceptiongroup exceptiongroup-* fastapi fastapi-[0-9]* h11 h11-* idna idna-* \
        jinja2 jinja2-* markupsafe markupsafe-* multipart \
        pydantic pydantic-[0-9]* pydantic_core pydantic_core-[0-9]* \
        python_multipart python_multipart-* sniffio sniffio-* \
        numpy numpy.libs numpy-* PIL pillow.libs pillow-* \
        starlette starlette-* tokenizers tokenizers-* \
        typing_extensions.py typing_extensions-* \
        typing_inspection typing_inspection-* uvicorn uvicorn-*; do
        [ -e "$p" ] && files+=("$p")
      done
      tar -cf - "${files[@]}"
    ' | tar -C "$PY" -xf -
  fi
fi

echo
du -sh "$LIBS/hip" "$LIBS/comgr" "$PY" 2>/dev/null || true
echo
case "$MODE" in
  native) check_runtime 1 0 ;;
  python) check_runtime 0 1 ;;
  *)      check_runtime 1 1 ;;
esac

cat <<EOF

主机直跑环境已就绪：
  bash scripts/chat_host.sh --prompt "你好" --n 32 --temp 0
  bash scripts/serve_host.sh
  bash scripts/host_dsh.sh 'python3 scripts/bench_rt.py'

注意：这里只替换“运行”工具链。要重新编译 build/rt 或 .hip 算子，仍然需要
DTK 的 hipcc；/opt/hyhal 只有驱动和 HSA 运行库，不包含 HIP 头文件/编译器。
EOF
