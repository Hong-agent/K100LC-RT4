#!/bin/bash
# 把“主机直跑”运行库打包成可上传 GitHub Release 的压缩包。
#
#   bash scripts/make_host_runtime_archive.sh
#   TAG=host-runtime-2026-09-27 bash scripts/make_host_runtime_archive.sh
#
# 产物（默认在 gitignore 的 dist/ 下）：
#   dist/K100LC-RT4-host-runtime-YYYY-MM-DD.tar.zst
#   dist/K100LC-RT4-host-runtime-YYYY-MM-DD.tar.zst.sha256
#
# 包内含 build/rt 和 runtime/，解压到仓库根目录即可：
#   tar --zstd -xf <archive> -C K100LC-RT4
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TAG="${TAG:-host-runtime-$(date +%F)}"
ASSET="${ASSET:-K100LC-RT4-${TAG}.tar.zst}"
OUT_DIR="${OUT_DIR:-$ROOT/dist}"
OUT="$OUT_DIR/$ASSET"

for f in \
  "$ROOT/build/rt" \
  "$ROOT/runtime/dtk-libs/hip/libgalaxyhip.so.5" \
  "$ROOT/runtime/dtk-libs/comgr/libamd_comgr.so.2" \
  "$ROOT/runtime/py/tokenizers/__init__.py"; do
  if [ ! -e "$f" ]; then
    echo "缺少 $f" >&2
    echo "先运行: bash scripts/make_host_runtime.sh" >&2
    exit 1
  fi
done

mkdir -p "$OUT_DIR"
STAGE="$(mktemp -d /tmp/k100lc-host-runtime-XXXXXX)"
cleanup() { rm -rf "$STAGE"; }
trap cleanup EXIT

mkdir -p "$STAGE/runtime"
cp -a "$ROOT/runtime/dtk-libs" "$STAGE/runtime/"
cp -a "$ROOT/runtime/py" "$STAGE/runtime/"
mkdir -p "$STAGE/build"
cp -a "$ROOT/build/rt" "$STAGE/build/rt"

# 去掉运行中产生的缓存，保持包干净、可重复。
find "$STAGE/runtime" -type d -name __pycache__ -prune -exec rm -rf {} +
find "$STAGE/runtime" -type f -name '*.pyc' -delete

cat > "$STAGE/runtime/HOST-RUNTIME-README.txt" <<EOF
K100LC-RT4 主机直跑运行库
=========================

内容：
  build/rt         预编译的自研运行时（gfx926）
  dtk-libs/hip/    libgalaxyhip.so.5 + hipkernel.bin.gfx926
  dtk-libs/comgr/  libamd_comgr.so.2
  py/              tokenizers / jinja2 / fastapi / uvicorn 等

用法：
  解压到 K100LC-RT4 仓库根目录：
    tar --zstd -xf $(basename "$OUT") -C K100LC-RT4

  校验：
    bash scripts/make_host_runtime.sh --check

  运行：
    bash scripts/chat_host.sh --prompt "你好" --n 32 --temp 0
    bash scripts/serve_host.sh

来源：
  从 DTK 26.04 镜像中抽取。运行仍需要主机 DCU 驱动 /opt/hyhal 和 /dev/kfd。
  编译新的 .hip 算子仍然需要 DTK 的 hipcc。

构建信息：
  tag: $(basename "$TAG")
  built_at_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)
  kernel: $(uname -r)
EOF

(
  cd "$STAGE/runtime"
  find dtk-libs py \( -type f -o -type l \) -print0 \
    | sort -z | xargs -0 sha256sum > SHA256SUMS
)

echo "压缩中：$OUT"
tar -C "$STAGE" -cf - build runtime | zstd -T0 -12 -q -f -o "$OUT"
(
  cd "$OUT_DIR"
  sha256sum "$ASSET" > "$ASSET.sha256"
)

echo
du -h "$OUT"
cat "$OUT.sha256"
echo
echo "上传到 GitHub Release 时，tag 用：$TAG"
echo "资产文件：$ASSET 和 $ASSET.sha256"
