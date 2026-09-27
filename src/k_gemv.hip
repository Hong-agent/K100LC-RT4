// 解码 GEMV 封装：复用现役 W4A4 GEMV（kernels/gemv_w4a4_core.h）
#include "kernels.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include "../kernels/gemv_w4a4_core.h"
#include "../kernels/gemv_w4a8_core.h"

// 量化缓冲（激活按 32 个一组）：随 rows 变化，按需扩容
static uint8_t* s_aq = nullptr;
static float*   s_as = nullptr;
static long long s_cap = 0;

void k_gemv_w4a4(float* y, const u32* wq, const float* wsc, const float* x, int rows, int N, int K) {
    if (rows > 4) { printf("k_gemv_w4a4: rows=%d > 4（预填充请走 GEMM）\n", rows); exit(1); }
    const long long need = (long long)rows * K;
    if (need > s_cap) {
        if (s_aq) hipFree(s_aq);
        if (s_as) hipFree(s_as);
        hipMalloc(&s_aq, need / 2 + 64);
        hipMalloc(&s_as, (size_t)rows * (K / SG) * 4 + 64);
        s_cap = need;
    }
    dim3 qgrid(K / SG, rows);
    quant_act4<<<qgrid, 32>>>(x, s_aq, s_as, rows, K);
    if (getenv("GEMV_DEBUG")) {
        float h[8]; uint8_t hq[8];
        hipMemcpy(h, s_as, 16, hipMemcpyDeviceToHost);
        hipMemcpy(hq, s_aq, 8, hipMemcpyDeviceToHost);
        printf("    [gemv] N=%d K=%d rows=%d  as[0:4]=%.5f %.5f %.5f %.5f  aq[0:4]=%02x %02x %02x %02x  x[0:2]=%.5f %.5f\n",
               N, K, rows, h[0], h[1], h[2], h[3], hq[0], hq[1], hq[2], hq[3], x[0], x[1]);
    }
    const int nwarp = NTHREADS / 64;
    const dim3 grid((N + nwarp * ROWS - 1) / (nwarp * ROWS));
    switch (rows) {
        case 1: gemv_w4a4<1><<<grid, NTHREADS>>>(wq, (const uint16_t*)wsc, (const u32*)s_aq, s_as, y, N, K); break;
        case 2: gemv_w4a4<2><<<grid, NTHREADS>>>(wq, (const uint16_t*)wsc, (const u32*)s_aq, s_as, y, N, K); break;
        case 3: gemv_w4a4<3><<<grid, NTHREADS>>>(wq, (const uint16_t*)wsc, (const u32*)s_aq, s_as, y, N, K); break;
        case 4: gemv_w4a4<4><<<grid, NTHREADS>>>(wq, (const uint16_t*)wsc, (const u32*)s_aq, s_as, y, N, K); break;
    }
    if (hipGetLastError() != hipSuccess) { printf("gemv launch fail\n"); exit(1); }
}

// ---------------------------------------------------------------- W4A8 ----
// 解码路径现役内核：int4 权重 × int8 激活（dot4），权重只读一遍。
// 激活缓冲按需扩容；ae/ao 是「偶数下标 / 奇数下标」两半的 int8。
static int8_t* s_ae = nullptr;
static int8_t* s_ao = nullptr;
static float*  s_asc = nullptr;
static int*    s_asu = nullptr;
static long long s_cap8 = 0;

void k_gemv_w4a8(float* y, const u32* wq, const float* wsc, const float* x, int rows, int N, int K) {
    if (rows > 4) { printf("k_gemv_w4a8: rows=%d > 4（预填充请走 GEMM）\n", rows); exit(1); }
    if (K % GRP) { printf("k_gemv_w4a8: K=%d 不是 %d 的整数倍\n", K, GRP); exit(1); }
    const long long need = (long long)rows * K;
    if (need > s_cap8) {
        if (s_ae) hipFree(s_ae);
        if (s_ao) hipFree(s_ao);
        if (s_asc) hipFree(s_asc);
        if (s_asu) hipFree(s_asu);
        hipMalloc(&s_ae, need / 2 + 64);
        hipMalloc(&s_ao, need / 2 + 64);
        hipMalloc(&s_asc, (size_t)rows * (K / GRP) * 4 + 64);
        hipMalloc(&s_asu, (size_t)rows * (K / GRP) * 4 + 64);
        s_cap8 = need;
    }
    // RT_NOQ=1 只用于计时对照：跳过量化内核（结果是错的，只回答「这些启动值多少时间」）
    static const int noq = getenv("RT_NOQ") ? atoi(getenv("RT_NOQ")) : 0;
    if (!noq) quant_act<<<dim3(K / GRP, rows), 32>>>(x, s_ae, s_ao, s_asc, s_asu, rows, K);
    const int nwarp = NTHREADS / 64;
    // M=2~4 是 MTP 校验批：一个 warp 处理 2 行权重、共享激活加载，实测带宽
    // M=2 +7%、M=3 +18%、M=4 +22%。M=1 继续用 1 行/warp（ROWS=2 在单 token 解码反而慢）。
    const int rpp = rows >= 2 ? 2 : 1;
    const dim3 grid((N + nwarp * rpp - 1) / (nwarp * rpp));
    // M==1（解码）走共享内存版本：激活先协作搬进 LDS，全局侧只剩权重。
    // 共享内存 = 2*(K/8)*4（激活）+ 2*(K/GRP)*4（尺度）字节。
    // 实测：把激活搬进 LDS 反而更慢（41.7 vs 38.2 ms/token）——4 个 warp 重复读的
    // 那份激活本来就命中 L1/L2，而协作装载多了一次块内 barrier 和共享内存占用。
    // 代码留着，默认关闭（RT_SMEM=1 才启用）以便以后换形状时再试。
    static const int nosmem = getenv("RT_SMEM") ? 0 : 1;
    const size_t smem_bytes = (size_t)(2 * (K / 8) + 2 * (K / GRP)) * 4;
    if (rows == 1 && !nosmem && smem_bytes <= 48 * 1024) {
        static bool done_attr = false;
        if (!done_attr) {
            hipFuncSetAttribute((const void*)gemv_w4a8<1, true, 1>,
                                hipFuncAttributeMaxDynamicSharedMemorySize, 48 * 1024);
            done_attr = true;
        }
        gemv_w4a8<1, true, 1><<<grid, NTHREADS, smem_bytes>>>(
            wq, (const uint16_t*)wsc, (const u32*)s_ae, (const u32*)s_ao, s_asc, s_asu, y, N, K);
        if (hipGetLastError() != hipSuccess) { printf("gemv_w4a8 smem launch fail\n"); exit(1); }
        return;
    }
    switch (rows) {
        case 1: gemv_w4a8<1, false, 1><<<grid, NTHREADS>>>(wq, (const uint16_t*)wsc, (const u32*)s_ae,
                                                           (const u32*)s_ao, s_asc, s_asu, y, N, K); break;
        case 2: gemv_w4a8<2, false, 2><<<grid, NTHREADS>>>(wq, (const uint16_t*)wsc, (const u32*)s_ae,
                                                           (const u32*)s_ao, s_asc, s_asu, y, N, K); break;
        case 3: gemv_w4a8<3, false, 2><<<grid, NTHREADS>>>(wq, (const uint16_t*)wsc, (const u32*)s_ae,
                                                           (const u32*)s_ao, s_asc, s_asu, y, N, K); break;
        case 4: gemv_w4a8<4, false, 2><<<grid, NTHREADS>>>(wq, (const uint16_t*)wsc, (const u32*)s_ae,
                                                           (const u32*)s_ao, s_asc, s_asu, y, N, K); break;
    }
    if (hipGetLastError() != hipSuccess) { printf("gemv_w4a8 launch fail\n"); exit(1); }
}
