#!/bin/bash
# 生成/刷新离线包 dist/K100LC-RT4-offline/app。
#
#   bash scripts/make_dist.sh              # 源码 + 脚本 + 预编译 rt + 权重（真实文件，约 15GB）
#   bash scripts/make_dist.sh --app-only   # 只同步源码/脚本/二进制，不动权重
#   WITH_IMAGE=1 bash scripts/make_dist.sh # 额外 docker save DTK 镜像（约 35GB，很慢）
#
# 说明：权重一律用「真实文件」拷进包里（先写 .tmp 再 mv，避免和仓库里的
# models/ 共用 inode）。这样离线包可以单独拷走/打包，不会因为主仓库删改而变。
set -euo pipefail
source "$(dirname "$0")/env.sh"

APP_ONLY=0
[ "${1:-}" = "--app-only" ] && APP_ONLY=1
DIST="$RT_ROOT/dist/K100LC-RT4-offline"
APP="$DIST/app"
SRC="$RT_MODEL_DIR"

mkdir -p "$APP/build" "$APP/models/Qwen3.8-27B-NVFP4/rt4"

# 真实文件拷贝（先写临时名再 mv，断掉硬链接）
copy_real() {
  local s="$1" d="$2"
  [ -f "$s" ] || { echo "缺少文件：$s" >&2; exit 1; }
  cp -f "$s" "$d.tmp"
  mv -f "$d.tmp" "$d"
}

echo "== [1/4] 同步源码 / 内核 / 文档 / 脚本 =="
for f in README.md RESUME.md; do cp -p "$RT_ROOT/$f" "$APP/$f"; done
for d in src kernels docs scripts tools web bench; do
  rm -rf "$APP/$d"
  cp -a "$RT_ROOT/$d" "$APP/$d"
done
# 包内不需要的东西：调试残留、本脚本自身、容器里的 __pycache__
rm -rf "$APP"/scripts/__pycache__ "$APP"/tools/__pycache__ "$APP"/build/scratch
rm -f "$APP"/scripts/make_dist.sh

echo "== [2/4] 放预编译运行时 build/rt =="
[ -x "$RT_ROOT/build/rt" ] || { echo "先跑 bash scripts/build_rt.sh" >&2; exit 1; }
cp -p "$RT_ROOT/build/rt" "$APP/build/rt"

if [ "$APP_ONLY" = "1" ]; then
  echo "== 跳过权重（--app-only） =="
else
  echo "== [3/4] 拷贝权重（真实文件，13.9+0.22+0.92GB） =="
  need=$((16 * 1024 * 1024))            # 16GB，单位 KB
  free=$(df -Pk "$APP" | awk 'NR==2 {print $4}')
  if [ "$free" -lt "$need" ]; then
    echo "空间不足：需要约 16GB，$APP 所在分区只剩 $((free / 1024 / 1024))GB" >&2
    exit 1
  fi
  for f in config.json generation_config.json tokenizer.json vocab.json chat_template.jinja; do
    copy_real "$SRC/$f" "$APP/models/Qwen3.8-27B-NVFP4/$f"
  done
  for f in qwen38_27b.rt4 qwen38_27b.rt4.json qwen38_27b_mtp.rt4 \
           qwen38_27b_mtp.rt4.json qwen38_27b_vision.rt4 \
           qwen38_27b_vision.rt4.json kv_scales.json; do
    [ -e "$SRC/rt4/$f" ] || continue
    printf '  %-28s ' "$f"
    copy_real "$SRC/rt4/$f" "$APP/models/Qwen3.8-27B-NVFP4/rt4/$f"
    du -h "$APP/models/Qwen3.8-27B-NVFP4/rt4/$f" | cut -f1
  done
fi

if [ "${WITH_IMAGE:-0}" != "0" ]; then
  echo "== [4/4] docker save 离线镜像（约 35GB，几分钟） =="
  mkdir -p "$DIST/image"
  sudo_rt docker save "$RT_IMG" -o "$DIST/image/rt-dtk26.04-qwen3.8.tar"
  ls -lh "$DIST/image/rt-dtk26.04-qwen3.8.tar"
else
  echo "== [4/4] 未生成镜像 tar（需要时：WITH_IMAGE=1 bash scripts/make_dist.sh --app-only）=="
fi

echo
echo "离线包就绪：$DIST"
echo "  体积（含硬链接去重后的真实占用）：$(du -sh "$DIST" | cut -f1)"
echo "  权重校验（links 应为 1，表示是独立文件）："
stat -c '    %h links  %s bytes  %n' "$APP/models/Qwen3.8-27B-NVFP4/rt4/qwen38_27b.rt4"
echo "  自检：bash $DIST/status.sh（目标机上）/ 或直接 bash $DIST/start.sh"
