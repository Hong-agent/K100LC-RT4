#!/bin/bash
# 在主目录 runtime/ 里准备一份可随离线包携带的 CPython 3.10。
#
# 产物：
#   runtime/python/               Python 解释器 + stdlib + 少量系统共享库
#   runtime/py/numpy, PIL/        自带的 numpy / Pillow wheel
#
# 目标机因此不需要预装 python3 / python3-numpy / python3-pil。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYDIR="$ROOT/runtime/python"
PYVER=3.10

[ -x "/usr/bin/python$PYVER" ] || {
  echo "缺少 /usr/bin/python$PYVER，需要 Ubuntu 22.04 的 CPython 3.10" >&2
  exit 1
}

mkdir -p "$PYDIR/bin" "$PYDIR/lib"
cp -a "/usr/bin/python$PYVER" "$PYDIR/bin/"
ln -sf "python$PYVER" "$PYDIR/bin/python3"
cp -a "/usr/lib/python$PYVER" "$PYDIR/lib/"

LIBPY="/usr/lib/x86_64-linux-gnu/libpython$PYVER.so.1.0"
if [ -e "$LIBPY" ]; then
  cp -a "$LIBPY" "$PYDIR/lib/"
  ln -sf "libpython$PYVER.so.1.0" "$PYDIR/lib/libpython$PYVER.so.1"
  ln -sf "libpython$PYVER.so.1.0" "$PYDIR/lib/libpython$PYVER.so"
fi

TMPLD="$(mktemp /tmp/k100lc-ldconfig-XXXXXX)"
ldconfig -p > "$TMPLD"
for lib in libexpat.so.1 libz.so.1 libssl.so.3 libcrypto.so.3 \
           libbz2.so.1.0 liblzma.so.5 libsqlite3.so.0 libffi.so.8 \
           libncursesw.so.6 libtinfo.so.6 libreadline.so.8; do
  src="$(awk -v n="$lib" '$1==n {print $NF; exit}' "$TMPLD")"
  [ -n "$src" ] && cp -L "$src" "$PYDIR/lib/"
done

if [ ! -d "$ROOT/runtime/py/numpy" ] || [ ! -d "$ROOT/runtime/py/PIL" ]; then
  echo "runtime/py 缺少 numpy/Pillow，从本地 DTK 镜像补齐..."
  bash "$ROOT/scripts/make_host_runtime.sh" --python-only
fi

echo "校验自带 Python..."
PYTHONHOME="$PYDIR" LD_LIBRARY_PATH="$PYDIR/lib" \
PYTHONPATH="$ROOT/runtime/py" "$PYDIR/bin/python$PYVER" - <<'PY'
import sys, numpy, PIL, fastapi, uvicorn, tokenizers, jinja2
print('  prefix :', sys.prefix)
print('  python :', sys.version.split()[0])
print('  numpy  :', numpy.__version__)
print('  Pillow :', PIL.__version__)
print('  web    :', fastapi.__version__, uvicorn.__version__)
PY

du -sh "$PYDIR" "$ROOT/runtime/py"
echo "自带 Python 运行时：$PYDIR"
