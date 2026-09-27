#!/bin/bash
# 从源头把 unsloth/Qwen3.8-27B-NVFP4 搬下来并转成 RT4 运行格式。
#
#   bash scripts/convert_weights.sh            # 缺什么补什么（幂等）
#   bash scripts/convert_weights.sh --check    # 只校验已有产物，不下载/不转换
#
# 镜像源用魔搭（hf-mirror 单连接只有 200KB/s 且会 302 到被墙的 CDN）。
set -euo pipefail
source "$(dirname "$0")/env.sh"

# 源权重指纹（字节数 + sha256），改源快照时同步更新 REPRODUCE.md 的表
SHA256=c473512c70eace07e2256fe9fd76596ac03e3295bee7d54cfb72676416afcc05
SIZE=22568192096
SHA256_MTP=1d8268aa85ace093a561e3e7b63b9d390dac1cd55a90cd55b5ec509c3c9da9fe
SIZE_MTP=849400392
MS_BASE="https://modelscope.cn/api/v1/models/unsloth/Qwen3.8-27B-NVFP4/repo?Revision=master&FilePath="
SRC="$RT_MODEL_DIR/model.safetensors"
MTP_SRC="$RT_MODEL_DIR/model_mtp.safetensors"
CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

mkdir -p "$RT_MODEL_DIR/rt4"
cd "$RT_MODEL_DIR"

files=(config.json model.safetensors.index.json tokenizer.json vocab.json chat_template.jinja
       model_mtp.safetensors)
if [ "$CHECK_ONLY" = 0 ]; then
  for f in "${files[@]}"; do
    [ -s "$f" ] || { echo "下载 $f"; curl -L -C - --retry 20 --retry-delay 4 --retry-all-errors -sS -o "$f" "$MS_BASE$f"; }
  done
  if [ ! -s model.safetensors ] || [ "$(stat -c%s model.safetensors)" != "$SIZE" ]; then
    echo "并行下载 model.safetensors (22.57GB, 16 连接)"
    python3 "$RT_ROOT/tools/fetch_par2.py" "$MS_BASEmodel.safetensors" model.safetensors \
            "$SIZE" 16 128 "$SHA256"
  fi
fi

echo "== 校验源权重 sha256 / 字节数 =="
check_one() {                       # $1=文件 $2=sha256 $3=字节数
  local f="$1" want="$2" size="$3" got sz
  [ -s "$f" ] || { echo "  FAIL 缺少 $f"; exit 1; }
  sz=$(stat -c%s "$f")
  got=$(sha256sum "$f" | cut -d' ' -f1)
  if [ "$got" = "$want" ] && [ "$sz" = "$size" ]; then
    echo "  OK   $f  $sz bytes  $got"
  else
    echo "  FAIL $f  实得 $sz bytes / $got" >&2
    echo "       期望 $size bytes / $want" >&2
    exit 1
  fi
}
check_one "$SRC" "$SHA256" "$SIZE"
check_one "$MTP_SRC" "$SHA256_MTP" "$SIZE_MTP"

if [ "$CHECK_ONLY" = 0 ]; then
  echo "== 转换（约 4.5 分钟）=="
  gcc -O2 -o "$RT_MODEL_DIR/rt4/convert" "$RT_ROOT/tools/convert.c" -lm
  "$RT_MODEL_DIR/rt4/convert" model.safetensors rt4/qwen38_27b.rt4
  "$RT_MODEL_DIR/rt4/convert" model_mtp.safetensors rt4/qwen38_27b_mtp.rt4
fi

echo "== 产物 =="
ls -lh rt4/ | grep -E 'rt4|json'
