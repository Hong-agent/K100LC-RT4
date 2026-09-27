#!/bin/bash
# 编译并跑 int4 FlashAttention（正确性 + 目标形状吞吐）。
#
# 用法:
#   bash scripts/run_fa.sh                 # 默认配置：正确性（3 个深度）+ 128k 吞吐
#   bash scripts/run_fa.sh check 4096      # 指定上下文深度的正确性
#   bash scripts/run_fa.sh bench 1024      # 指定 q 行数的吞吐
#   bash scripts/run_fa.sh abl             # 重跑 docs/FLASH-ATTN.md 里的消融表
#   FLAGS="-DBN=128 -DNT=512" bash scripts/run_fa.sh bench 1024   # 换分块参数
set -euo pipefail
source "$(dirname "$0")/env.sh"

MODE="${1:-all}"
N="${2:-1024}"
FLAGS="${FLAGS:-}"

sudo_rt docker run "${RT_DOCKER_ARGS[@]}" \
  -v "$RT_ROOT/kernels":/kernels -w /kernels "$RT_IMG" \
  bash -c "source /opt/dtk/env.sh >/dev/null 2>&1
    hipcc -O3 --offload-arch=$RT_ARCH $FLAGS -o /tmp/fa flash_attn_int4.hip || exit 1
    case '$MODE' in
      check) /tmp/fa check $N ;;
      bench) /tmp/fa bench $N ;;
      abl)   for f in '' '-DABL_QK_ONLY' '-DABL_PV_ONLY' '-DABL_QK_ONLY -DABL_NOLDS_QK' '-DBN=128 -DNT=512'; do
               hipcc -O3 --offload-arch=$RT_ARCH \$f -o /tmp/faa flash_attn_int4.hip 2>/dev/null
               printf '%-32s ' \"[\$f]\"; /tmp/faa bench $N | grep -E 'VGPR|吞吐' | tr '\n' ' '; echo
             done ;;
      *)     for c in 64 4096 130944; do /tmp/fa check \$c | grep -E 'n_kv=|内核 vs'; done
             /tmp/fa bench $N ;;
    esac"
