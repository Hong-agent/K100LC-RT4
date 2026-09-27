// INT4 GEMM, second generation.  Supersedes bench/int4_gemm.cpp (kept as the
// reference implementation and as the "what the naive version measured" record).
//
// Measured on K100_LC / gfx926, M=2048 N=17408 K=5120, ceiling 75.5 TMAC/s:
//
//   int4_gemm.cpp (v1)                     37.9 TMAC/s   50.2%
//   this file, QG=128                      45.3          60.0%   accuracy-neutral
//   this file, QG=512                      50.7          67.2%
//   this file, QG=1024                     52.1          68.9%
//
// The three changes that matter, in the order they were found (see x_int4.cpp,
// x2_int4.cpp, x3_int4.cpp for the A/B measurements):
//
//  1. stage the next K-tile with the STS *after* the dot8 loop, not before it.
//     Otherwise the s_waitcnt for the global load lands in front of the compute
//     and the latency is never hidden.                                    +8.8%
//  2. pipeline the shared-memory fragments one kd ahead in registers, so the
//     LDS latency is covered by the previous kd's 32 dot8s.               +6%
//  3. carry the group scales through shared memory inside the same pipeline
//     instead of loading them from global at the point of use.            +4%
//
// And one optional lever that costs accuracy:
//
//  4. QG, the quantization group size, decoupled from the staging depth.
//     int32 never overflows over the whole K (64 * 5120 = 3.3e5 << 2^31), so
//     the only thing that limits QG is the quantizer.  qerr.cpp measures the
//     cost: relative Frobenius error of the GEMM output goes 0.33 -> 0.50 ->
//     0.60 for QG = 128 / 512 / 1024 on heavy-tailed synthetic data.
//     -> keep QG=128 until real-weight perplexity says otherwise.
//
// build: hipcc -O3 --offload-arch=gfx926 -DQG=128 -o int4_gemm2 int4_gemm2.cpp
// run:   ./int4_gemm2 [M] [N] [K]
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    printf("HIP ERR %s @%d: %s\n", #x, __LINE__, hipGetErrorString(e_)); exit(1);} } while (0)

typedef unsigned int u32;

// ---------------------------------------------------------------- config ---
#include "../kernels/gemm_core.h"

// -------------------------------------------------------------------- host --
int main(int argc, char** argv) {
    int M = 2048, N = 17408, K = 5120;
    if (argc > 3) { M = atoi(argv[1]); N = atoi(argv[2]); K = atoi(argv[3]); }
    printf("GEMM  M=%d N=%d K=%d   QG=%d   (%.2f GMAC)\n",
           M, N, K, QG, (double)M * N * K / 1e9);

    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> hA((size_t)M * K), hB((size_t)N * K);
    for (auto& v : hA) v = nd(rng) * 0.5f;
    for (auto& v : hB) v = nd(rng) * 0.1f;

    const int KD = K / 8, NG = K / QG;
    std::vector<u32>   hAp((size_t)M * KD), hBp((size_t)N * KD);
    std::vector<float> hsa((size_t)M * NG), hsb((size_t)N * NG);
    std::vector<float> dA((size_t)M * K), dB((size_t)N * K);

    std::vector<int8_t> q(QG);
    for (int r = 0; r < M; r++)
        for (int g = 0; g < NG; g++) {
            float s = quant_group(&hA[(size_t)r * K + g * QG], QG, q.data());
            hsa[(size_t)g * M + r] = s;              // group-major sa[ng][M]
            for (int u = 0; u < QG; u += 8) {
                hAp[(size_t)r * KD + g * (QG / 8) + u / 8] = pack8(&q[u]);
                for (int t = 0; t < 8; t++)
                    dA[(size_t)r * K + g * QG + u + t] = q[u + t] * s;
            }
        }
    for (int r = 0; r < N; r++)
        for (int g = 0; g < NG; g++) {
            float s = quant_group(&hB[(size_t)r * K + g * QG], QG, q.data());
            hsb[(size_t)g * N + r] = s;              // group-major sb[ng][N]
            for (int u = 0; u < QG; u += 8) {
                hBp[(size_t)r * KD + g * (QG / 8) + u / 8] = pack8(&q[u]);
                for (int t = 0; t < 8; t++)
                    dB[(size_t)r * K + g * QG + u + t] = q[u + t] * s;
            }
        }

    u32 *dAp, *dBp; float *dsa, *dsb, *dC, *hC;
    CK(hipMalloc(&dAp, hAp.size() * 4)); CK(hipMalloc(&dBp, hBp.size() * 4));
    CK(hipMalloc(&dsa, hsa.size() * 4)); CK(hipMalloc(&dsb, hsb.size() * 4));
    CK(hipMalloc(&dC, (size_t)M * N * 4));
    hC = (float*)malloc((size_t)M * N * 4);
    CK(hipMemcpy(dAp, hAp.data(), hAp.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(dBp, hBp.data(), hBp.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(dsa, hsa.data(), hsa.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(dsb, hsb.data(), hsb.size() * 4, hipMemcpyHostToDevice));

    dim3 grid(N / BN, M / BM);
    gemm_w4a4<<<grid, NTHREADS>>>(dAp, dBp, dsa, dsb, dC, M, N, K);
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(hC, dC, (size_t)M * N * 4, hipMemcpyDeviceToHost));

    // correctness vs f64 over the dequantized operands, reported against rms(|c|)
    {
        double worst_abs = 0, ref_rms = 0;
        for (int t = 0; t < 400; t++) {
            int i = (int)(rng() % M), j = (int)(rng() % N);
            double ref = 0;
            for (int k = 0; k < K; k += 8) {
                double s = 0;
                for (int u = 0; u < 8; u++)
                    s += (double)dA[(size_t)i * K + k + u] * dB[(size_t)j * K + k + u];
                ref += s;
            }
            worst_abs = fmax(worst_abs, fabs((double)hC[(size_t)i * N + j] - ref));
            ref_rms += ref * ref;
        }
        ref_rms = sqrt(ref_rms / 400);
        printf("correctness: worst |abs err| = %.3e   vs rms|c| = %.3e   ratio = %.2e%s\n",
               worst_abs, ref_rms, worst_abs / ref_rms,
               worst_abs / ref_rms > 1e-5 ? "   *** CHECK ***" : "   ok");
    }

    {
        hipEvent_t s, e; CK(hipEventCreate(&s)); CK(hipEventCreate(&e));
        const int reps = 20;
        gemm_w4a4<<<grid, NTHREADS>>>(dAp, dBp, dsa, dsb, dC, M, N, K);
        CK(hipDeviceSynchronize());
        CK(hipEventRecord(s));
        for (int r = 0; r < reps; r++)
            gemm_w4a4<<<grid, NTHREADS>>>(dAp, dBp, dsa, dsb, dC, M, N, K);
        CK(hipEventRecord(e)); CK(hipEventSynchronize(e));
        float ms = 0; CK(hipEventElapsedTime(&ms, s, e)); ms /= reps;
        double macs = (double)M * N * K;
        printf("throughput : %.3f ms  -> %.2f TMAC/s  (%.1f%% of the 75.5 TMAC/s dot8 ceiling)\n",
               ms, macs / (ms * 1e-3) / 1e12, 100.0 * (macs / (ms * 1e-3) / 1e12) / 75.5);
    }
    return 0;
}
