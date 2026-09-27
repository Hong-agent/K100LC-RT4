#!/bin/bash
# 采集「跑通本项目」的完整环境版本，输出 markdown（仓库里的 docs/ENV.md 就是它的产物）。
# 目标机器上排查问题时也跑它，和 docs/ENV.md 对一遍就知道差在哪。
#
#   bash scripts/collect_env.sh > docs/ENV.md
set -uo pipefail
source "$(dirname "$0")/env.sh"

sh_() { bash -c "$*" 2>&1 | sed '/^$/d'; }
have() { command -v "$1" >/dev/null 2>&1; }

echo "# 环境版本清单（实测跑通的那台机器）"
echo
echo "> 由 \`scripts/collect_env.sh\` 自动生成，采集时间 $(date '+%F %T %Z')。"
echo "> 目标机器上先跑一遍这个脚本，和本文对不上的地方（尤其内核与驱动）优先排查。"
echo
echo "## 操作系统与内核"
echo
echo '| 项 | 值 |'
echo '|---|---|'
echo "| 发行版 | $(. /etc/os-release; echo "$PRETTY_NAME") |"
echo "| 内核 | \`$(uname -r)\` |"
echo "| 架构 | \`$(uname -m)\` |"
echo "| 已装内核 | \`$(ls /lib/modules | tr '\n' ' ')\` |"
echo "| GRUB 默认启动项 | \`$(grep -E '^GRUB_DEFAULT' /etc/default/grub 2>/dev/null | cut -d= -f2-)\` |"
echo "| CPU | $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//') |"
echo "| CPU 核数 | $(nproc) |"
echo "| 内存 | $(awk '/MemTotal/{printf "%.1f GB", $2/1024/1024}' /proc/meminfo)（跑 27B 模型也够：权重走 mmap + 分块 H2D，宿主常驻 <1GB） |"
echo
echo "## 加速卡与驱动"
echo
echo '| 项 | 值 |'
echo '|---|---|'
echo "| 卡 | Hygon K100_LC（\`gfx926\`，120 CU，64GB HBM，无矩阵核心） |"
echo "| 驱动包 | \`$(dpkg -l 2>/dev/null | awk '/rock-5\.7/{print $2" "$3}' | head -1)\` |"
echo "| 内核模块 | \`hydcu $(cat /sys/module/hydcu/version 2>/dev/null || echo '?')\`，已加载：$(lsmod | awk '$1 ~ /^(hy|hydrm)/ {printf "`%s` ", $1}') |"
echo "| 用户态运行时 | \`/usr/local/hyhal\`（\`/opt/hyhal\` 软链，$(sudo_rt du -sh /usr/local/hyhal 2>/dev/null | cut -f1)） |"
echo "| 监控进程 | $(systemctl is-active hymgr.service 2>/dev/null)（\`hymgr.service\`） |"
echo "| 设备节点 | $( [ -e /dev/kfd ] && echo -n '`/dev/kfd`' || echo -n '缺 /dev/kfd' ) $(ls /dev/dri/renderD* 2>/dev/null | tr '\n' ' ')  |"
echo
echo '```'
sudo_rt /opt/hyhal/bin/hy-smi 2>/dev/null | head -8
echo '```'
echo
echo "## 容器与工具链"
echo
echo '| 项 | 值 |'
echo '|---|---|'
echo "| docker | \`$(sudo_rt docker --version 2>/dev/null | cut -d, -f1)\` |"
echo "| DTK 镜像 | \`$RT_IMG\` |"
echo "| 镜像 ID | \`$(sudo_rt docker inspect --format '{{.Id}}' "$RT_IMG" 2>/dev/null | cut -c8-19)\`（构建于 $(sudo_rt docker inspect --format '{{.Created}}' "$RT_IMG" 2>/dev/null | cut -c1-10)） |"
echo "| 容器内 hipcc | \`$(bash "$RT_ROOT/scripts/dsh.sh" 'hipcc --version 2>/dev/null | head -1' 2>/dev/null)\` |"
echo "| 容器内 python | \`$(bash "$RT_ROOT/scripts/dsh.sh" 'python3 --version' 2>/dev/null)\` |"
echo "| 容器内 gcc | \`$(bash "$RT_ROOT/scripts/dsh.sh" 'gcc --version 2>/dev/null | head -1' 2>/dev/null)\` |"
echo "| 容器内 torch / transformers | \`$(bash "$RT_ROOT/scripts/dsh.sh" 'python3 -c "import torch,transformers;print(torch.__version__, transformers.__version__)"' 2>/dev/null)\` |"
echo "| 宿主机 gcc（量化用） | \`$(gcc --version 2>/dev/null | head -1)\` |"
echo "| 宿主机 python | \`$(python3 --version)\` |"
echo "| zstd | \`$(zstd --version 2>/dev/null | grep -o 'v[0-9][0-9.]*' | head -1)\` |"
echo "| zip / unzip | $(dpkg -l zip unzip 2>/dev/null | awk '/^ii/ {printf "`%s %s` ", $2, $3}') |"
echo
echo "## 本仓库"
echo
echo '| 项 | 值 |'
echo '|---|---|'
echo "| 提交 | \`$(git -C "$RT_ROOT" log -1 --format='%h %ad' --date=short 2>/dev/null)\` |"
echo "| 编译目标 | \`$RT_ARCH\` |"
echo "| 预编译运行时 | \`build/rt\`（$(stat -c%s "$RT_ROOT/build/rt" 2>/dev/null | awk '{printf "%.0f KB", $1/1024}')） |"
echo
echo "## 说明"
echo
echo "* **驱动只对「安装时正在运行的内核」编译**：本机装驱动时是 \`6.8.0-40-generic\`，"
echo "  GRUB 默认项也被锁到它（另一个已装内核 \`6.8.0-138-generic\` 下没有驱动模块与固件）。"
echo "  升级内核后必须重装驱动，否则重启会没有 \`/dev/kfd\`。"
echo "* 装驱动用的安装包是**修正版** \`rock-5.7.1-6.2.35-V1.6.7-内核68修正.aio.run\`"
echo "  （原厂包在内核 6.8 上编译失败，原因与 diff 见 \`driver/INSTALL.md\`）。"
echo "* 版本对不上时先看 \`REPRODUCE.md\` 的复现记录模板，把上面这些值一起贴出来。"
