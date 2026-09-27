#!/bin/bash
# 生成“主机直跑”离线一键包：复用 dist/K100LC-RT4-offline 里的权重，
# 刷新 app/ 源码和 runtime/ 运行库，写出 start.sh / status.sh，最后压到桌面。
#
#   bash scripts/make_host_offline.sh
#   OUT_DIR=/home/t/桌面 NAME=K100LC-RT4-offline-host-2026-09-27 bash scripts/make_host_offline.sh
#   NO_ARCHIVE=1 bash scripts/make_host_offline.sh   # 只刷新目录，不压缩
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIST="${DIST:-$ROOT/dist/K100LC-RT4-offline}"
APP="$DIST/app"
DRIVER="$DIST/driver"
OUT_DIR="${OUT_DIR:-/home/t/桌面}"
NAME="${NAME:-K100LC-RT4-offline-host-$(date +%F)}"
ARCHIVE="$OUT_DIR/$NAME.tar.zst"

MODEL_DIR="$APP/models/Qwen3.8-27B-NVFP4"
RT4="$MODEL_DIR/rt4"

[ -x "$ROOT/build/rt" ] || { echo "缺少 build/rt" >&2; exit 1; }
if [ ! -x "$ROOT/runtime/python/bin/python3.10" ] || \
   [ ! -d "$ROOT/runtime/py/numpy" ] || [ ! -d "$ROOT/runtime/py/PIL" ]; then
  echo "== 准备自带 Python / numpy / Pillow =="
  bash "$ROOT/scripts/bundle_python_runtime.sh"
fi
for f in qwen38_27b.rt4 qwen38_27b.rt4.json qwen38_27b_mtp.rt4 \
         qwen38_27b_vision.rt4 qwen38_27b_vision.rt4.json; do
  [ -f "$RT4/$f" ] || { echo "离线包缺少权重：$RT4/$f（先用 make_dist.sh 生成）" >&2; exit 1; }
done
[ -d "$DRIVER" ] || { echo "离线包缺少 driver/（先用 make_dist.sh 生成）" >&2; exit 1; }

mkdir -p "$APP" "$APP/build" "$APP/runtime" "$MODEL_DIR"

echo "== 同步 app 源码 / 脚本 / 文档 =="
for f in README.md RESUME.md LICENSE NOTICE .gitignore; do
  [ -f "$ROOT/$f" ] && cp -p "$ROOT/$f" "$APP/$f"
done
for d in src kernels docs scripts tools web bench; do
  mkdir -p "$APP/$d"
  cp -a "$ROOT/$d/." "$APP/$d/"
done
cp -p "$ROOT/build/rt" "$APP/build/rt"

echo "== 同步主机直跑运行库 =="
mkdir -p "$APP/runtime/dtk-libs" "$APP/runtime/py" "$APP/runtime/python"
cp -a "$ROOT/runtime/dtk-libs/." "$APP/runtime/dtk-libs/"
cp -a "$ROOT/runtime/py/." "$APP/runtime/py/"
cp -a "$ROOT/runtime/python/." "$APP/runtime/python/"

echo "== 同步模型元数据（权重已在离线包内，不重复拷 22GB 源权重） =="
for f in config.json generation_config.json tokenizer.json vocab.json chat_template.jinja; do
  [ -f "$ROOT/models/Qwen3.8-27B-NVFP4/$f" ] && \
    cp -p "$ROOT/models/Qwen3.8-27B-NVFP4/$f" "$MODEL_DIR/$f"
done
[ -f "$RT4/kv_scales.json" ] || cp -p "$ROOT/models/Qwen3.8-27B-NVFP4/rt4/kv_scales.json" "$RT4/" 2>/dev/null || true

cat > "$DIST/start.sh" <<'EOF'
#!/bin/bash
# K100LC-RT4 离线一键启动（主机直跑，不需要 Docker / DTK 镜像）。
#   bash start.sh                 # 默认 80 端口，局域网只输 IP
#   PORT=8080 bash start.sh       # 改端口
#   CTX=131072 bash start.sh      # 改上下文
#   bash start.sh --stop          # 停止
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
APP="$HERE/app"

if [ ! -e /dev/kfd ] || [ ! -x /opt/hyhal/bin/hy-smi ]; then
  echo "DCU 驱动未就绪：/dev/kfd 或 /opt/hyhal/bin/hy-smi 缺失。" >&2
  echo "先安装驱动并重启：sudo bash $HERE/driver/installer/rock-*.aio.run" >&2
  echo "细节见 $HERE/driver/INSTALL.md" >&2
  exit 1
fi
if [ ! -f "$APP/models/Qwen3.8-27B-NVFP4/rt4/qwen38_27b.rt4" ]; then
  echo "离线包缺少主权重 qwen38_27b.rt4" >&2
  exit 1
fi
if [ ! -e "$APP/runtime/py/tokenizers/__init__.py" ] || \
   [ ! -e "$APP/runtime/dtk-libs/hip/libgalaxyhip.so.5" ]; then
  echo "主机运行库不完整，尝试从 GitHub Release 补齐..." >&2
  bash "$APP/scripts/fetch_host_runtime.sh"
fi

export RT_VISION_DEVICE="${RT_VISION_DEVICE:-cpu}"
exec bash "$APP/scripts/serve_host.sh" "$@"
EOF
chmod +x "$DIST/start.sh"

cat > "$DIST/status.sh" <<'EOF'
#!/bin/bash
# K100LC-RT4 离线包自检：驱动 / 设备 / 权重 / 主机运行库。
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
APP="$HERE/app"
ok()   { printf '  \033[32mOK\033[0m   %s\n' "$*"; }
bad()  { printf '  \033[31mNG\033[0m   %s\n' "$*"; }
warn() { printf '  \033[33m--\033[0m   %s\n' "$*"; }

echo "== 系统 / 驱动 =="
uname -sr | sed 's/^/  /'
[ -e /dev/kfd ] && ok "/dev/kfd 存在" || bad "/dev/kfd 缺失（装驱动并重启）"
ls /dev/dri/renderD* >/dev/null 2>&1 && ok "/dev/dri/renderD* 存在" || bad "/dev/dri/renderD* 缺失"
[ -x /opt/hyhal/bin/hy-smi ] && ok "/opt/hyhal/bin/hy-smi 可执行" || bad "/opt/hyhal 缺失"
[ -x /opt/hyhal/bin/hy-smi ] && /opt/hyhal/bin/hy-smi 2>&1 | sed -n '4,6p' | sed 's/^/  /'

echo "== 权重 =="
RT4="$APP/models/Qwen3.8-27B-NVFP4/rt4"
[ -x "$APP/build/rt" ] && ok "build/rt 可执行" || bad "build/rt 缺失"
for f in qwen38_27b.rt4 qwen38_27b_mtp.rt4 qwen38_27b_vision.rt4; do
  if [ -f "$RT4/$f" ]; then
    ok "$(printf '%-28s %s' "$f" "$(du -h "$RT4/$f" | cut -f1)")"
  else
    warn "$f 不在包里"
  fi
done

echo "== 主机运行库 =="
if bash "$APP/scripts/make_host_runtime.sh" --check >/dev/null 2>&1; then
  ok "libgalaxyhip / COMGR / Python 依赖可用"
else
  bad "主机运行库不可用，运行: bash $APP/scripts/fetch_host_runtime.sh"
fi
echo
echo "一键启动：bash $HERE/start.sh"
EOF
chmod +x "$DIST/status.sh"

cat > "$DIST/README-OFFLINE.md" <<'EOF'
# K100LC-RT4 离线一键包（主机直跑）

一台装好 DCU 驱动的 K100_LC 机器 + 这个目录，即可不联网运行：

```bash
bash status.sh     # 自检：驱动 / 权重 / 运行库
bash start.sh      # 默认 80 端口；局域网直接访问 http://<本机IP>/
```

启动后：

- 网页：`http://<本机IP>/`
- OpenAI 接口：`http://<本机IP>/v1`
- 停止：`bash start.sh --stop`
- 改端口：`PORT=8080 bash start.sh`

包内自带独立 Python 3.10、numpy、Pillow 和 Web 依赖；目标机不需要预装
Python。视觉塔默认在 CPU 上运行（`RT_VISION_DEVICE=cpu`），不占用 DCU，
也不需要 torch/transformers；文本模型仍在 DCU 上跑。

```
K100LC-RT4-offline/
├── app/        源码 + build/rt + 自带 Python/运行库 + RT4 权重
├── driver/     DCU 驱动安装包 / 预编译 hyhal / udev / 服务 / 快照
├── start.sh    主机直跑一键启动
└── status.sh   自检
```

首次上卡先装驱动（装完重启）：

```bash
sudo bash driver/installer/rock-*.aio.run
sudo reboot
/opt/hyhal/bin/hy-smi
```

许可：`app/` 内代码为 Apache-2.0；`driver/` 中的海光驱动与预编译 hyhal
按你有权使用的范围分发。
EOF

echo "== 离线目录已就绪 =="
du -sh "$DIST" "$APP" "$DRIVER"

if [ "${NO_ARCHIVE:-0}" = "1" ]; then
  echo "NO_ARCHIVE=1，跳过压缩。目录：$DIST"
  exit 0
fi

mkdir -p "$OUT_DIR"
echo "== 压缩到 $ARCHIVE =="
TMP_ARCHIVE="$ARCHIVE.tmp"
tar -C "$(dirname "$DIST")" -cf - "$(basename "$DIST")" | \
  zstd -T0 -3 -q -f -o "$TMP_ARCHIVE"
mv -f "$TMP_ARCHIVE" "$ARCHIVE"
(
  cd "$OUT_DIR"
  sha256sum "$(basename "$ARCHIVE")" > "$(basename "$ARCHIVE").sha256"
)
echo
du -h "$ARCHIVE"
cat "$ARCHIVE.sha256"
