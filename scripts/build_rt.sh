#!/bin/bash
# 编译自研运行时（src/ 全部 .hip + model.cpp）→ build/rt
# FLAGS="-DA8_ROWS=4" 可以给所有编译单元加宏（调内核分块参数用）
set -euo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p "$RT_ROOT/build"
FLAGS="${FLAGS:-}"

sudo_rt docker run "${RT_DOCKER_ARGS[@]}" \
  -v "$RT_ROOT":/rt -w /rt "$RT_IMG" \
  bash -c "source /opt/dtk/env.sh >/dev/null 2>&1
    set -e
    for f in src/k_new.hip src/k_fa.hip src/k_gemv.hip src/k_gemm.hip src/k_vision.hip; do
      echo \"  CC \$(basename \$f)\"
      hipcc -O3 -std=c++17 --offload-arch=$RT_ARCH $FLAGS -c \$f -I src -o build/\$(basename \$f).o
    done
    echo '  CC model.cpp'
    hipcc -O3 -std=c++17 --offload-arch=$RT_ARCH $FLAGS -c src/model.cpp -I src -o build/model.o
    hipcc --offload-arch=$RT_ARCH -o build/rt build/*.o
    echo '  链接完成: build/rt'"
