#!/bin/bash
# 从 GitHub Release 下载、校验并解压“主机直跑”运行库，不依赖本机 DTK 镜像。
#
#   bash scripts/fetch_host_runtime.sh
#   TAG=host-runtime-2026-09-27 bash scripts/fetch_host_runtime.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="${REPO:-Hong-agent/K100LC-RT4}"
TAG="${TAG:-host-runtime-2026-09-27-r2}"
ASSET="${ASSET:-K100LC-RT4-${TAG}.tar.zst}"
OUT_DIR="${OUT_DIR:-$ROOT/dist}"
ARCHIVE="$OUT_DIR/$ASSET"
SUM="$ARCHIVE.sha256"
URL="https://github.com/$REPO/releases/download/$TAG"

mkdir -p "$OUT_DIR"
if [ ! -f "$ARCHIVE" ]; then
  echo "下载 $URL/$ASSET"
  curl -L --fail --retry 5 --retry-all-errors -o "$ARCHIVE" "$URL/$ASSET"
fi
if [ ! -f "$SUM" ]; then
  echo "下载 $URL/$ASSET.sha256"
  curl -L --fail --retry 5 --retry-all-errors -o "$SUM" "$URL/$ASSET.sha256"
fi

(
  cd "$OUT_DIR"
  sha256sum -c "$(basename "$SUM")"
)

echo "解压到 $ROOT/runtime/"
tar --zstd -xf "$ARCHIVE" -C "$ROOT"
bash "$ROOT/scripts/make_host_runtime.sh" --check
