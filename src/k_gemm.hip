// 预填充 int4 GEMM 封装：复用 int4_gemm2 的内核（kernels/gemm_core.h）
#include "kernels.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include "../kernels/gemm_core.h"

void k_gemm_i4(float* c, const u32* wq, const float* wsc_gm, const u32* aq, const float* asc_gm,
               int M, int N, int K) {
    if (M % BM || N % BN || K % G) {
        printf("k_gemm_i4: 形状必须是 BM=%d/BN=%d/G=%d 的整数倍（M=%d N=%d K=%d）\n",
               BM, BN, G, M, N, K);
        exit(1);
    }
    gemm_w4a4<<<dim3(N / BN, M / BM), NTHREADS>>>(aq, wq, asc_gm, wsc_gm, c, M, N, K);
    if (hipGetLastError() != hipSuccess) { printf("gemm launch fail\n"); exit(1); }
}

// W4A8：激活是 8bit。把它拆成 q8 = 16*h + l（h、l 都是 [-8,7] 的 int4 数字），
// 于是 Σ w·q8 = 16·Σ w·h + Σ w·l —— 同一个 int4×int4 GEMM 内核跑两遍再相加，
// 精度等价 int8 激活，代价是 2 倍的 dot8 与 2 倍的权重读取（预填充实测见 docs/）。
void k_gemm_i4_a8(float* c, float* c2, const u32* wq, const float* wsc_gm, const u32* ah,
                  const u32* al, const float* sc, const float* sc16, int M, int N, int K) {
    if (M % BM || N % BN || K % G) {
        printf("k_gemm_i4_a8: 形状必须是 BM=%d/BN=%d/G=%d 的整数倍（M=%d N=%d K=%d）\n",
               BM, BN, G, M, N, K);
        exit(1);
    }
    const dim3 grid(N / BN, M / BM);
    gemm_w4a4<<<grid, NTHREADS>>>(ah, wq, sc16, wsc_gm, c, M, N, K);
    gemm_w4a4<<<grid, NTHREADS>>>(al, wq, sc, wsc_gm, c2, M, N, K);
    if (hipGetLastError() != hipSuccess) { printf("gemm a8 launch fail\n"); exit(1); }
    k_add_inplace(c, c2, (long long)M * N);
}
