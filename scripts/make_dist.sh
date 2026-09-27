#!/bin/bash
# 生成/刷新离线包 dist/K100LC-RT4-offline：应用 + 权重 + **DCU 驱动**（+ 可选镜像）。
#
#   bash scripts/make_dist.sh               # 源码 + 预编译 rt + 权重 + 驱动（约 15.5GB）
#   bash scripts/make_dist.sh --src-only    # 源码 + 驱动，**不带 RT4 权重**（约 160MB，
#                                           #   目标机联网自己拉源模型并本地量化）
#   bash scripts/make_dist.sh --app-only    # 只同步源码/脚本/二进制，不动权重
#   NO_DRIVER=1 bash scripts/make_dist.sh   # 不带驱动
#   WITH_IMAGE=1 bash scripts/make_dist.sh  # 额外 docker save DTK 镜像（约 35GB，很慢）
#
# 说明：权重一律用「真实文件」拷进包里（先写 .tmp 再 mv，避免和仓库里的
# models/ 共用 inode）。这样离线包可以单独拷走/打包，不会因为主仓库删改而变。
#
# 驱动部分（driver/）包含三样东西：
#   1) 修正过的安装包 rock-*.aio.run + 编译修正说明（内核 6.8 上原包会编译失败）
#   2) 本机装好的用户态运行时 /usr/local/hyhal 打包（含已编译的 7 个 .ko、hy-smi、
#      hymgr、vbios 固件），以及 udev 规则 / modprobe 配置 / hymgr.service
#   3) manifest.txt：内核版本、包版本、lsmod、hy-smi、GRUB 默认项等现场快照
#   安装步骤见 driver/INSTALL.md（关键：驱动只对「安装时正在运行的内核」编译，
#   装完必须把 GRUB 默认项锁回那个内核，升级内核后要重装一次）。
set -euo pipefail
source "$(dirname "$0")/env.sh"

APP_ONLY=0
[ "${1:-}" = "--app-only" ] && APP_ONLY=1
SRC_ONLY=0
[ "${1:-}" = "--src-only" ] && SRC_ONLY=1
if [ "$SRC_ONLY" = "1" ]; then
  DIST="$RT_ROOT/dist/K100LC-RT4-src"
else
  DIST="$RT_ROOT/dist/K100LC-RT4-offline"
fi
APP="$DIST/app"
SRC="$RT_MODEL_DIR"
# 驱动安装包/说明所在目录（默认找桌面上的 k100lc资料，可用 RT_DRIVER_SRC 覆盖）
DRIVER_SRC="${RT_DRIVER_SRC:-/home/t/桌面/k100lc资料}"

mkdir -p "$APP/build" "$APP/models/Qwen3.8-27B-NVFP4/rt4"

# 真实文件拷贝（先写临时名再 mv，断掉硬链接）
copy_real() {
  local s="$1" d="$2"
  [ -f "$s" ] || { echo "缺少文件：$s" >&2; exit 1; }
  cp -f "$s" "$d.tmp"
  mv -f "$d.tmp" "$d"
}

echo "== [1/5] 同步源码 / 内核 / 文档 / 脚本 =="
for f in README.md RESUME.md; do cp -p "$RT_ROOT/$f" "$APP/$f"; done
for d in src kernels docs scripts tools web bench; do
  rm -rf "$APP/$d"
  cp -a "$RT_ROOT/$d" "$APP/$d"
done
# 包内不需要的东西：调试残留、本脚本自身、容器里的 __pycache__
rm -rf "$APP"/scripts/__pycache__ "$APP"/tools/__pycache__ "$APP"/build/scratch
rm -f "$APP"/scripts/make_dist.sh

echo "== [2/5] 放预编译运行时 build/rt =="
[ -x "$RT_ROOT/build/rt" ] || { echo "先跑 bash scripts/build_rt.sh" >&2; exit 1; }
cp -p "$RT_ROOT/build/rt" "$APP/build/rt"

if [ "$APP_ONLY" = "1" ] || [ "$SRC_ONLY" = "1" ]; then
  if [ "$SRC_ONLY" = "1" ]; then
    echo "== 跳过权重（--src-only：目标机用 bootstrap.sh 联网下载 + 本地量化） =="
    # 顺手清掉上一轮留下的 RT4 产物，保证这个包真的「不带模型」
    rm -f "$APP/models/Qwen3.8-27B-NVFP4/rt4/"*.rt4 \
          "$APP/models/Qwen3.8-27B-NVFP4/rt4/"*.rt4.json
    mkdir -p "$APP/models/Qwen3.8-27B-NVFP4/rt4"
  else
    echo "== 跳过权重（--app-only） =="
  fi
else
  echo "== [3/5] 拷贝权重（真实文件，13.9+0.22+0.92GB） =="
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

if [ "${NO_DRIVER:-0}" != "0" ]; then
  echo "== [4/5] 跳过驱动（NO_DRIVER=1） =="
else
  echo "== [4/5] 打包 DCU 驱动（安装包 + 已装好的 /usr/local/hyhal + 系统配置） =="
  DRIVER="$DIST/driver"
  rm -rf "$DRIVER"
  mkdir -p "$DRIVER/installer" "$DRIVER/system" "$DRIVER/hyhal"
  # 1) 修正版安装包（内核 6.8 上原包编译失败，见 说明.md）+ 校验值
  shopt -s nullglob
  inst=("$DRIVER_SRC"/rock-*.aio.run)
  shopt -u nullglob
  if [ ${#inst[@]} -eq 0 ]; then
    echo "  警告：$DRIVER_SRC 下没找到 rock-*.aio.run，驱动安装包缺失" >&2
  else
    for f in "${inst[@]}"; do
      cp -p "$f" "$DRIVER/installer/"
      printf '  %-52s %s\n' "$(basename "$f")" "$(du -h "$f" | cut -f1)"
    done
  fi
  for f in "$DRIVER_SRC"/*内核68修正*.md "$DRIVER_SRC"/*驱动*.md; do
    [ -f "$f" ] && cp -p "$f" "$DRIVER/installer/"
  done
  ( cd "$DRIVER/installer" && md5sum ./*.aio.run > MD5SUMS.txt 2>/dev/null ) || true
  # 2) 已装好的用户态运行时（含 dkms/*.ko、hy-smi、hymgr、vbios 固件）
  KREL="$(uname -r)"
  sudo_rt tar -C /usr/local -czf "$DRIVER/hyhal/hyhal-prebuilt-${KREL}.tar.gz" hyhal
  printf '  %-52s %s\n' "hyhal-prebuilt-${KREL}.tar.gz" \
         "$(du -h "$DRIVER/hyhal/hyhal-prebuilt-${KREL}.tar.gz" | cut -f1)"
  # 3) 系统侧配置（没有安装包时也能手工还原）
  for f in /etc/udev/rules.d/16-dcu.rules /etc/modprobe.d/hydcu.conf \
           /etc/modprobe.d/blacklist-hydcu.conf /lib/systemd/system/hymgr.service; do
    [ -f "$f" ] && sudo_rt cp -p "$f" "$DRIVER/system/$(basename "$f")"
  done
  # 4) 现场快照：目标机器上对照用
  {
    echo "# DCU 驱动现场快照（$(date '+%F %T')）"
    echo
    echo "## 内核 / 系统"
    uname -a
    echo "GRUB_DEFAULT: $(grep -E '^GRUB_DEFAULT' /etc/default/grub 2>/dev/null)"
    echo "已装内核模块目录: $(ls /lib/modules | tr '\n' ' ')"
    echo
    echo "## 驱动包"
    dpkg -l | grep -E "rock-5|hydcu|hyhal" || echo "(未装 dpkg 包)"
    echo
    echo "## 已加载模块"
    lsmod | grep -E "^(hydcu|hyttm|hykcl|hydcu_sched|hydrm|hy_extra)" || true
    echo
    echo "## 设备节点"
    ls -l /dev/kfd 2>/dev/null; ls -l /dev/dri 2>/dev/null | head -5
    echo
    echo "## hy-smi"
    /opt/hyhal/bin/hy-smi 2>&1 | head -12
    echo
    echo "## 用户态驱动 /usr/local/hyhal"
    du -sh /usr/local/hyhal 2>/dev/null
    ls /opt/hyhal/dkms 2>/dev/null
  } > "$DRIVER/manifest.txt" 2>&1
  cat > "$DRIVER/INSTALL.md" <<'EOF'
# DCU 驱动安装（K100 标准版 / rock-5.7.1，内核 6.8）

本目录是「能把这台机器跑起来」的驱动现场快照。目标机器按下面顺序做，装完再跑
`bash ../start.sh` 起服务。

## 1. 装驱动（必做）

```bash
sudo bash installer/rock-5.7.1-6.2.35-V1.6.7-内核68修正.aio.run
sudo reboot
/opt/hyhal/bin/hy-smi          # 看得到 DCU 0 就说明驱动 OK
```

三个坑（都是实测踩过的）：

* **必须用「内核68修正」那个包**。原厂包在 Ubuntu 22.04 / 内核 6.8 上会因为
  `enum drm_debug_category` 探测误判（`-Werror=missing-prototypes`）而编译失败；
  修正内容与验证见同目录的《内核68修正-说明.md》，安装包 MD5 见 `installer/MD5SUMS.txt`。
* 驱动**只针对安装时正在运行的内核**编译。本机装的时候是 `6.8.0-40-generic`，安装
  脚本会把 GRUB 默认项改回它（`GRUB_DEFAULT="Advanced options for Ubuntu>Ubuntu,
  with Linux 6.8.0-40-generic"`）；目标机器上如果默认启动的不是装驱动时的内核，
  重启后会没有 `/dev/kfd`，必须把 GRUB 默认项锁到那一个。
* 以后**每次升级内核都要重装一次驱动**。安装器需要 `linux-headers-$(uname -r)`
  和 `gcc`/`make`。

## 2. 用户态运行时（一般不用手工装）

安装器会自己铺 `/usr/local/hyhal`（`/opt/hyhal` 是指向它的软链）。要还原本机这一份：

```bash
sudo tar -C /usr/local -xzf hyhal/hyhal-prebuilt-<内核版本>.tar.gz
```

里面有 `bin/hy-smi`、`bin/hymgr`、`lib/*.so`（HIP/HSA）、`hsa/`、`vbios/` 固件，以及
`dkms/*.ko`（本机编译好的 7 个内核模块）。容器靠 `-v /opt/hyhal:/opt/hyhal:ro` 挂它，
所以这个目录**必须存在**，否则 `start.sh` 起不来。

## 3. 系统侧配置

`system/` 里是这几份文件（装驱动时会自动写好，手工核对/恢复用）：

```bash
sudo cp system/16-dcu.rules /etc/udev/rules.d/
sudo cp system/hydcu.conf system/blacklist-hydcu.conf /etc/modprobe.d/
sudo cp system/hymgr.service /lib/systemd/system/
sudo udevadm control --reload-rules && sudo udevadm trigger
sudo systemctl enable --now hymgr.service
```

## 4. 验证

```bash
ls -l /dev/kfd /dev/dri/renderD*         # 设备节点在
lsmod | grep -E "^(hydcu|hyttm|hykcl)"   # 模块已加载
/opt/hyhal/bin/hy-smi                    # 看得到卡
bash ../status.sh                        # 离线包整体自检
```

`manifest.txt` 是打包时的现场快照（内核、包版本、lsmod、hy-smi、GRUB 默认项），
目标机器对不上时先看它。
EOF
  echo "  manifest.txt / INSTALL.md 已写入"
fi

if [ "${WITH_IMAGE:-0}" != "0" ]; then
  echo "== [5/5] docker save 离线镜像（约 35GB，几分钟） =="
  mkdir -p "$DIST/image"
  sudo_rt docker save "$RT_IMG" -o "$DIST/image/rt-dtk26.04-qwen3.8.tar"
  ls -lh "$DIST/image/rt-dtk26.04-qwen3.8.tar"
else
  echo "== [5/5] 未生成镜像 tar（需要时：WITH_IMAGE=1 bash scripts/make_dist.sh --app-only）=="
fi

# ---------------------------- 包内的入口文件 --------------------------------
# start.sh / status.sh / README-OFFLINE.md（每次刷新都重写，保证与当前包一致）
cat > "$DIST/start.sh" <<'EOF'
#!/bin/bash
# 启动离线包里的服务（默认 8080，OpenAI 兼容 + 自带网页）。参数原样透传：
#   bash start.sh --stop          # 停服务
#   PORT=8080 CTX=131072 bash start.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/app/scripts/serve.sh" "$@"
EOF
chmod +x "$DIST/start.sh"

if [ "$SRC_ONLY" = "1" ]; then
cat > "$DIST/bootstrap.sh" <<'EOF'
#!/bin/bash
# 首次引导：联网拉 DTK 镜像 + 下载 NVFP4 源模型 + 本地量化成 RT4。
#
#   bash bootstrap.sh               # 全流程（体检 → 镜像 → 模型+量化 → 视觉塔 → 自检）
#   bash bootstrap.sh --no-vision   # 不导出视觉塔（网页就不能看图）
#   bash bootstrap.sh --check       # 只体检，不下载不转换
#
# 需要联网：harbor.sourcefind.cn:5443（DTK 镜像，约 35GB）、modelscope.cn
# （unsloth/Qwen3.8-27B-NVFP4，22.57GB + MTP 0.85GB）。
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
APP="$HERE/app"
cd "$APP"
source scripts/env.sh

VISION=1; CHECK=0
for a in "$@"; do
  case "$a" in
    --no-vision) VISION=0 ;;
    --check) CHECK=1 ;;
    *) echo "未知参数：$a" >&2; exit 2 ;;
  esac
done

say()  { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
need() { command -v "$1" >/dev/null 2>&1 || { echo "缺少 $1，请先安装" >&2; exit 1; }; }

say "1/5 体检（驱动 / 依赖）"
need docker; need python3; need gcc; need curl
if [ -e /dev/kfd ]; then echo "  OK   /dev/kfd"; else
  echo "  NG   没有 /dev/kfd → 驱动没装好，见 $HERE/driver/INSTALL.md"; fi
if [ -x /opt/hyhal/bin/hy-smi ]; then
  /opt/hyhal/bin/hy-smi 2>&1 | sed -n '4,6p' | sed 's/^/  /'
else
  echo "  NG   /opt/hyhal 缺失 → 见 $HERE/driver/INSTALL.md"
fi
if [ "$CHECK" = "1" ]; then echo; echo "（--check：到此为止）"; exit 0; fi

say "2/5 DTK 容器镜像（约 35GB）"
IMGS="$(docker images 2>/dev/null || true)"
[ -z "$IMGS" ] && IMGS="$(sudo_rt docker images 2>/dev/null || true)"
if printf '%s' "$IMGS" | awk -v p="${RT_IMG##*/}" 'index($0,p){f=1} END{exit !f}'; then
  echo "  OK   已有 ${RT_IMG##*/}"
else
  shopt -s nullglob
  tars=("$HERE"/image/*.tar)
  shopt -u nullglob
  if [ ${#tars[@]} -gt 0 ]; then
    for t in "${tars[@]}"; do echo "  从 $t 载入"; sudo_rt docker load -i "$t"; done
  else
    echo "  docker pull $RT_IMG"
    sudo_rt docker pull "$RT_IMG"
  fi
fi

say "3/5 下载 NVFP4 源模型 + 本地量化（22.57GB + 0.85GB；转换约 4.5 分钟）"
bash scripts/convert_weights.sh

if [ "$VISION" = "1" ]; then
  say "4/5 导出视觉塔 RT4（约 0.93GB，1~2 分钟）"
  bash scripts/prepare_vision.sh
else
  say "4/5 跳过视觉塔（--no-vision）"
fi

say "5/5 自检"
bash scripts/test_tools.sh | tail -3
bash "$HERE/status.sh"
cat <<'EOT'

下一步：
  bash start.sh                 # 起服务（网页 http://<本机IP>:8080/）
  bash start.sh --stop          # 停服务
EOT
EOF
chmod +x "$DIST/bootstrap.sh"

cat > "$DIST/README-FIRST-RUN.md" <<'EOF'
# K100LC-RT4（源码 + 驱动，不带 RT4 权重）

这个包**不含量化好的 RT4 权重**：目标机器联网自己拉源模型并本地量化。适合「带宽够、
想把权重在自己机器上生成一遍」的场景；要开箱即用（含 15GB 权重）请用另一个离线包。

```
K100LC-RT4-src/
├── app/               源码 + 脚本 + 工具 + 文档 + 预编译 build/rt（无 RT4 权重）
├── driver/            DCU 驱动：修正版安装包 + 预编译 /usr/local/hyhal + udev/服务 + 现场快照
├── bootstrap.sh       首次引导：体检 → 拉镜像 → 下载源模型 → 本地量化 → 视觉塔 → 自检
├── start.sh           起服务（OpenAI 兼容 + 网页，默认 8080）
├── status.sh          自检
└── README-OFFLINE.md  包结构说明
```

## 目标机器上的顺序

```bash
# 0) 驱动（必做，用「内核68修正」包；装完锁内核并重启，细节见 driver/INSTALL.md）
sudo bash driver/installer/rock-*.aio.run && sudo reboot
/opt/hyhal/bin/hy-smi

# 1) 首次引导（要联网；约 25GB 下载 + 35GB 镜像）
bash bootstrap.sh

# 2) 起服务
bash start.sh
```

## 会下载/生成什么，需要多少磁盘与时间

| 步骤 | 内容 | 体积 | 备注 |
|---|---|---|---|
| DTK 容器镜像 | `harbor.sourcefind.cn:5443/...` | ≈35GB | 编译与运行都靠它；也可放 `image/*.tar` 让脚本 `docker load` |
| 源模型 | `unsloth/Qwen3.8-27B-NVFP4`（safetensors + MTP） | 22.57GB + 0.85GB | 走魔搭，多连接下载 + sha256 校验（`tools/fetch_par2.py`） |
| 量化产物 | `rt4/qwen38_27b.rt4` + MTP | 13.91GB + 0.22GB | 本地 `tools/convert.c` 量化，约 **4.5 分钟** |
| 视觉塔 | `rt4/qwen38_27b_vision.rt4` | 0.93GB | `--no-vision` 可跳过（跳过就不能在网页看图） |

**磁盘**：源模型 23.4GB + RT4 15GB + 镜像 35GB ≈ **75GB**（`--no-vision` 省 1GB；不需要
源模型时常驻的话可以删掉 `app/models/Qwen3.8-27B-NVFP4/model*.safetensors`）。
**依赖**：docker、python3、gcc、curl（量化在宿主机跑，不需要 DTK 工具链）。

## 网络

* 镜像：`harbor.sourcefind.cn:5443`（海光官方 harbor；如需账号见海光开发者平台）
* 模型：`modelscope.cn`（脚本默认走魔搭；`hf-mirror` 单连接只有 200KB/s 且会 302 到被墙 CDN）

完全离线的机器请用另一个包（`K100LC-RT4-offline-*.tar.zst`，含权重）。

## 许可

`app/` 里的代码是本项目的（Apache-2.0，见 `app/LICENSE`）；`driver/` 与镜像属于海光/DCU
第三方内容，源模型与 DTK 镜像遵循各自许可，请自行下载、按许可使用。
EOF
fi

cat > "$DIST/status.sh" <<'EOF'
#!/bin/bash
# 离线包自检：驱动 / 设备 / 权重 / 镜像 / 容器，能一眼看出缺哪一块。
#   bash status.sh           # 快检（只查大小）
#   bash status.sh --full    # 额外校验 RT4 权重的 sha256（13.9GB，约半分钟）
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
APP="$HERE/app"
FULL=0
[ "${1:-}" = "--full" ] && FULL=1
ok()   { printf '  \033[32mOK\033[0m   %s\n' "$*"; }
bad()  { printf '  \033[31mNG\033[0m   %s\n' "$*"; }
warn() { printf '  \033[33m--\033[0m   %s\n' "$*"; }
# 注意：不要用 `xxx | grep -q`——grep -q 提前退出会把上游弄成 SIGPIPE(141)，
# 配上 set -o pipefail 就会误判成失败（这个坑在 status.sh 第一版里踩到了）。
has_line() { awk -v pat="$1" '$0 ~ pat { f = 1 } END { exit !f }'; }

echo "== 系统 / 内核 =="
echo "  $(uname -sr)"
grep -E '^GRUB_DEFAULT' /etc/default/grub 2>/dev/null | sed 's/^/  /' || warn "读不到 /etc/default/grub"

echo "== DCU 驱动 =="
lsmod | has_line '^hydcu' && ok "内核模块 hydcu 已加载" || bad "hydcu 未加载（装驱动并重启，见 driver/INSTALL.md）"
[ -e /dev/kfd ] && ok "/dev/kfd 存在" || bad "/dev/kfd 不存在"
ls /dev/dri/renderD* >/dev/null 2>&1 && ok "/dev/dri/renderD* 存在" || bad "/dev/dri/renderD* 不存在"
[ -x /opt/hyhal/bin/hy-smi ] && ok "/opt/hyhal/bin/hy-smi 可执行" || bad "/opt/hyhal 缺失（容器要挂它）"
if [ -x /opt/hyhal/bin/hy-smi ]; then
  /opt/hyhal/bin/hy-smi 2>&1 | sed -n '4,6p' | sed 's/^/  /'
fi

echo "== 应用 / 权重 =="
[ -x "$APP/build/rt" ] && ok "build/rt 可执行" || bad "build/rt 缺失（重新解包）"
RT4="$APP/models/Qwen3.8-27B-NVFP4/rt4"
if [ ! -f "$RT4/qwen38_27b.rt4" ] && [ -f "$HERE/bootstrap.sh" ]; then
  warn "还没量化出 RT4 —— 这是「源码+驱动」包，先跑：bash $HERE/bootstrap.sh"
  warn "（它会联网拉 DTK 镜像、下载 NVFP4 源模型 22.57GB 并本地量化，约 25GB 下载量）"
else
  for f in qwen38_27b.rt4 qwen38_27b_mtp.rt4 qwen38_27b_vision.rt4; do
    if [ -f "$RT4/$f" ]; then
      ok "$(printf '%-28s %s' "$f" "$(du -h "$RT4/$f" | cut -f1)")"
    else
      warn "$f 不在包里（MTP/视觉可选，主权重必须在）"
    fi
  done
  [ -f "$RT4/qwen38_27b.rt4.json" ] && ok "RT4 manifest 在" || bad "qwen38_27b.rt4.json 缺失"
fi

if [ "$FULL" = "1" ] && [ -f "$RT4/qwen38_27b.rt4" ]; then
  echo "== 权重 sha256（--full） =="
  ( cd "$RT4" && sha256sum qwen38_27b.rt4 ) | sed 's/^/  /'
fi

echo "== 容器镜像 =="
IMG_PAT='dtk26.04|vllm0.18.1'
IMGS="$(docker images 2>/dev/null || true)"
[ -z "$IMGS" ] && IMGS="$(sudo -n docker images 2>/dev/null || true)"
[ -z "$IMGS" ] && [ -n "${SUDO_ASKPASS:-}" ] && \
  IMGS="$(sudo -A docker images 2>/dev/null || true)"
if printf '%s' "$IMGS" | has_line "$IMG_PAT"; then
  ok "本机已有 DTK 镜像"
elif [ -f "$HERE/image/rt-dtk26.04-qwen3.8.tar" ]; then
  warn "镜像 tar 在包里但还没 load：sudo docker load -i image/rt-dtk26.04-qwen3.8.tar"
else
  warn "无法确认镜像（docker 可能需要 sudo）：编译/运行都要它，见 README-OFFLINE.md"
fi
echo
echo "起服务：bash $HERE/start.sh"
EOF
chmod +x "$DIST/status.sh"

cat > "$DIST/README-OFFLINE.md" <<'EOF'
# K100LC-RT4 离线包

目标：**一台装好 DCU 驱动的机器 + 这个目录**，就能不联网跑起 Qwen3.8-27B 的自研运行时。

```
K100LC-RT4-offline/
├── app/       源码 + 脚本 + 预编译 build/rt + RT4 权重（约 15GB）
├── driver/    DCU 驱动：安装包 + 预编译 /usr/local/hyhal + udev/modprobe/服务 + 现场快照
├── image/     DTK 容器镜像 tar（可选，约 35GB；没有它就得本机已有该镜像）
├── start.sh   起服务（OpenAI 兼容 + 自带网页，默认 8080）
└── status.sh  自检：驱动 / 设备 / 权重 / 镜像
```

## 新机器上的顺序

```bash
# 1) 驱动（必做，装完要重启；用「内核68修正」包，细节见 driver/INSTALL.md）
sudo bash driver/installer/rock-*.aio.run && sudo reboot
/opt/hyhal/bin/hy-smi                        # 能看到卡

# 2) 容器镜像（包里带 image/ 时才需要）
sudo docker load -i image/rt-dtk26.04-qwen3.8.tar

# 3) 自检 + 起服务
bash status.sh
bash start.sh                                # http://<本机IP>:8080/
```

工作区文件、上传的图片/文档都落在 `app/workspaces/` 与系统临时目录，不污染系统。

## 常见问题

* `hy-smi` 报找不到设备 → 驱动没装好，或 GRUB 默认内核不是装驱动时的那个
  （驱动只对安装时运行的内核编译；升级内核要重装）。
* `start.sh` 起不来 → 先 `bash status.sh`；日志在 `sudo docker logs -f rt-serve`。
* 如果 `sudo` 需要输密码而当前不是终端（脚本里跑、远程 CI 等），给 `SUDO_ASKPASS`
  指向一个能打印密码的脚本即可（`app/scripts/env.sh` 支持，包里故意**不带**密码文件）：
  `SUDO_ASKPASS=/path/to/askpass.sh bash start.sh`。
* 报 413 → 提示词本身超过了上下文（默认 131072），可用 `CTX=262144 bash start.sh`
  放大（KV 会按比例吃显存）。

## 许可提醒

`app/` 里的代码是本项目的（Apache-2.0，见 `app/LICENSE`）；`driver/` 与 `image/`
是海光/DCU 的第三方驱动与容器镜像，仅在你有权使用的范围内复制与分发。
EOF

echo
echo "入口文件：start.sh / status.sh / README-OFFLINE.md"
echo "离线包就绪：$DIST"
echo "  体积（含硬链接去重后的真实占用）：$(du -sh "$DIST" | cut -f1)"
if [ -f "$APP/models/Qwen3.8-27B-NVFP4/rt4/qwen38_27b.rt4" ]; then
  echo "  权重校验（links 应为 1，表示是独立文件）："
  stat -c '    %h links  %s bytes  %n' "$APP/models/Qwen3.8-27B-NVFP4/rt4/qwen38_27b.rt4"
else
  echo "  本包不含 RT4 权重（--src-only）：目标机执行 bash bootstrap.sh 联网下载 + 本地量化"
fi
echo "  自检：bash $DIST/status.sh（目标机上）/ 起服务：bash $DIST/start.sh"
