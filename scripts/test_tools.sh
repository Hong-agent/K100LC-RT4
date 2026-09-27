#!/bin/bash
# 工具链自检：合成用例（造 NVFP4+FP8 小文件 → 转换 → 逐元素比对）。
# 秒级完成，用来确认转换器没被改坏。
set -euo pipefail
source "$(dirname "$0")/env.sh"

cd "$RT_ROOT/tools"
python3 make_test_st.py >/dev/null
gcc -O2 -o /tmp/rt_convert_test convert.c -lm
/tmp/rt_convert_test /tmp/rt_test/in.safetensors /tmp/rt_test/out.rt4 2>&1 | tail -4
python3 check_rt4.py /tmp/rt_test/out.rt4 /tmp/rt_test/expect.json
echo
echo "注：转换器内部用 roundf（半数远离零），Python 期望值用 banker's rounding，"
echo "    因此在正好 .5 的平局点上会差 1 个量化格——这是两条都合法的舍入，不是 bug。"
