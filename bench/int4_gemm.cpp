// Decisive prototype: can a hand-written INT4 GEMM on K100_LC (gfx926) reach
// the throughput the 1000 tok/s prefill target needs?
//
// The chip has no matrix cores (no MFMA/WMMA of any flavour - the assembler
// rejects every variant for gfx926). Its fastest arithmetic primitive is
//   v_dot8_i32_i4: 8 signed int4 MACs accumulated into one int32
// measured at 75.5 TMAC/s versus 37.9 for int8 dot4 and 16.9 for packed fp16.
//
// So the whole engine is designed around W4A4 (4-bit weights, 4-bit
// activations, int32 accumulate) with per-group scales applied in f32.
//
// This file: quantize + pack + tiled GEMM + correctness check vs dequantized
// reference + throughput measurement.
//
// build: hipcc -O3 --offload-arch=gfx926 -o int4_gemm int4_gemm.cpp
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

// ---------------------------------------------------------------- tiling ---
#define BM 128         // tokens per block
#define BN 64          // output channels per block
#ifndef GD
#define GD 16          // dwords of K processed per pipeline stage
#endif
#define G  (GD * 8)    // elements per quantization group (= one stage)
#define TM 8           // output rows per thread
#define TN 4           // output cols per thread
#define NTHREADS 256   // (BM/TM) * (BN/TN)

// ------------------------------------------------------------- reference ---
// Quantize one group of G floats to signed int4 (-8..7) with a single scale.
static inline float quant_group(const float* x, int8_t* q) {
    float amax = 0.f;
    for (int i = 0; i < G; i++) amax = fmaxf(amax, fabsf(x[i]));
    float scale = amax > 0.f ? amax / 7.0f : 1.f;
    for (int i = 0; i < G; i++) {
        int v = (int)lrintf(x[i] / scale);
        if (v > 7) v = 7;
        if (v < -8) v = -8;
        q[i] = (int8_t)v;
    }
    return scale;
}

// pack 8 signed int4 (low nibble first) into a dword
static inline u32 pack8(const int8_t* q) {
    u32 w = 0;
    for (int i = 0; i < 8; i++) w |= (u32)(q[i] & 0xF) << (4 * i);
    return w;
}

// ---------------------------------------------------------------- kernel ---
// C[M,N] = A[M,K] @ B[N,K]^T  with A,B stored as int4 dwords + group scales.
//
// Layout choices that matter here:
//   * shared tiles are K-major (AsT[kd][row]) so a thread's TM A-dwords sit
//     contiguously and come back with 128-bit LDS. 2 LDS.128 + 1 LDS.128 feed
//     TM*TN = 32 dot8, i.e. 35 issue slots per 256 MACs.
//   * the group scales live in group-major arrays sa[ng][M] / sb[ng][N], so a
//     warp reads 16 consecutive floats per row-group instead of striding by
//     640 bytes. That single change is what took the first version from 12%.
//   * two shared stages, so the next K-tile is in flight while the current one
//     is being multiplied.
__global__ void __launch_bounds__(NTHREADS)
gemm_w4a4(const u32* __restrict__ Ap, const u32* __restrict__ Bp,
          const float* __restrict__ sa, const float* __restrict__ sb,
          float* __restrict__ C, int M, int N, int K) {
    __shared__ u32 AsT[2][GD][BM + 4];
    __shared__ u32 BsT[2][GD][BN + 4];

    const int KD = K >> 3;            // dwords per row
    const int NG = K / G;             // groups per row
    const int tid = threadIdx.x;
    const int ty = tid >> 4;          // 0..15  (16 thread rows)
    const int tx = tid & 15;          // 0..15
    const int bm = blockIdx.y * BM;
    const int bn = blockIdx.x * BN;

    int   iacc[TM][TN];
    float facc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
        for (int j = 0; j < TN; j++) { iacc[i][j] = 0; facc[i][j] = 0.f; }

    // ---- tile staging: A is BM x GD dwords, B is BN x GD dwords ----
    // 2 threads cooperate per A row, 4 per B row.
    const int AD = GD / 2, BD = GD / 4;      // dwords staged per thread
    const int arow = tid >> 1, aseg = (tid & 1) * AD;
    const int brow = tid >> 2, bseg = (tid & 3) * BD;

    const int nsteps = K / G;

    // prologue: stage K-tile 0 into buffer 0
    {
        const u32* ap = Ap + (size_t)(bm + arow) * KD + aseg;
        const u32* bp = Bp + (size_t)(bn + brow) * KD + bseg;
        u32 av[AD], bv[BD];
#pragma unroll
        for (int m = 0; m < AD; m += 4) *(uint4*)&av[m] = *(const uint4*)(ap + m);
#pragma unroll
        for (int m = 0; m < BD; m += 4) *(uint4*)&bv[m] = *(const uint4*)(bp + m);
#pragma unroll
        for (int m = 0; m < AD; m++) AsT[0][aseg + m][arow] = av[m];
#pragma unroll
        for (int m = 0; m < BD; m++) BsT[0][bseg + m][brow] = bv[m];
    }
    __syncthreads();

    for (int s = 0; s < nsteps; s++) {
        const int cur = s & 1, nxt = cur ^ 1;

        // ---- prefetch the next K-tile while the current one is multiplied ----
        if (s + 1 < nsteps) {
            const int k0 = (s + 1) * G;
            const u32* ap = Ap + (size_t)(bm + arow) * KD + (k0 >> 3) + aseg;
            const u32* bp = Bp + (size_t)(bn + brow) * KD + (k0 >> 3) + bseg;
            u32 av[AD], bv[BD];
#pragma unroll
            for (int m = 0; m < AD; m += 4) *(uint4*)&av[m] = *(const uint4*)(ap + m);
#pragma unroll
            for (int m = 0; m < BD; m += 4) *(uint4*)&bv[m] = *(const uint4*)(bp + m);
#pragma unroll
            for (int m = 0; m < AD; m++) AsT[nxt][aseg + m][arow] = av[m];
#pragma unroll
            for (int m = 0; m < BD; m++) BsT[nxt][bseg + m][brow] = bv[m];
        }

        // ---- this thread's TM row scales / TN col scales (group-major) ----
        float rsa[TM], rsb[TN];
#pragma unroll
        for (int h = 0; h < TM / 4; h++)
            *(float4*)&rsa[h * 4] = *(const float4*)&sa[(size_t)s * M + bm + ty * TM + h * 4];
#pragma unroll
        for (int j = 0; j < TN; j++) rsb[j] = sb[(size_t)s * N + bn + tx * TN + j];

        // ---- int32 accumulation: GD dwords x TM x TN dot8 ----
#pragma unroll
        for (int i = 0; i < TM; i++)
#pragma unroll
            for (int j = 0; j < TN; j++) iacc[i][j] = 0;

        const u32* ap = &AsT[cur][0][ty * TM];
        const u32* bp = &BsT[cur][0][tx * TN];
#pragma unroll
        for (int kd = 0; kd < GD; kd++) {
            u32 a8[TM], b4[TN];
            *(uint4*)&a8[0] = *(const uint4*)(ap + kd * (BM + 4));
            *(uint4*)&a8[4] = *(const uint4*)(ap + kd * (BM + 4) + 4);
            *(uint4*)b4 = *(const uint4*)(bp + kd * (BN + 4));
#pragma unroll
            for (int i = 0; i < TM; i++)
#pragma unroll
                for (int j = 0; j < TN; j++)
                    asm volatile("v_dot8_i32_i4 %0, %1, %2, %0\n"
                                 : "+v"(iacc[i][j]) : "v"(a8[i]), "v"(b4[j]));
        }

        // ---- fold the int32 partials back into f32 with the group scales ----
#pragma unroll
        for (int i = 0; i < TM; i++)
#pragma unroll
            for (int j = 0; j < TN; j++)
                facc[i][j] = fmaf((float)iacc[i][j], rsa[i] * rsb[j], facc[i][j]);

        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
        for (int j = 0; j < TN; j++)
            C[(size_t)(bm + ty * TM + i) * N + (bn + tx * TN + j)] = facc[i][j];
}

int main(int argc, char** argv) {
    int M = 512, N = 4096, K = 5120;
    if (argc > 3) { M = atoi(argv[1]); N = atoi(argv[2]); K = atoi(argv[3]); }
    printf("GEMM  M=%d N=%d K=%d   (%.2f GMAC)\n", M, N, K, (double)M * N * K / 1e9);

    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> hA((size_t)M * K), hB((size_t)N * K);
    for (auto& v : hA) v = nd(rng) * 0.5f;
    for (auto& v : hB) v = nd(rng) * 0.1f;

    // ---- quantize + pack ----
    const int KD = K / 8, NG = K / G;
    std::vector<u32>   hAp((size_t)M * KD), hBp((size_t)N * KD);
    std::vector<float> hsa((size_t)M * NG), hsb((size_t)N * NG);
    std::vector<float> dA((size_t)M * K), dB((size_t)N * K);   // dequantized

    std::vector<int8_t> q(G);
    for (int r = 0; r < M; r++)
        for (int g = 0; g < NG; g++) {
            float s = quant_group(&hA[(size_t)r * K + g * G], q.data());
            hsa[(size_t)g * M + r] = s;          // group-major: sa[ng][M]
            for (int u = 0; u < G; u += 8) {
                hAp[(size_t)r * KD + g * (G / 8) + u / 8] = pack8(&q[u]);
                for (int t = 0; t < 8; t++) dA[(size_t)r * K + g * G + u + t] = q[u + t] * s;
            }
        }
    for (int r = 0; r < N; r++)
        for (int g = 0; g < NG; g++) {
            float s = quant_group(&hB[(size_t)r * K + g * G], q.data());
            hsb[(size_t)g * N + r] = s;          // group-major: sb[ng][N]
            for (int u = 0; u < G; u += 8) {
                hBp[(size_t)r * KD + g * (G / 8) + u / 8] = pack8(&q[u]);
                for (int t = 0; t < 8; t++) dB[(size_t)r * K + g * G + u + t] = q[u + t] * s;
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

    // ---- correctness: compare a sample of outputs against f64 reference ----
    {
        double worst_abs = 0, worst_rel = 0;
        int n_bad = 0;
        for (int t = 0; t < 400; t++) {
            int i = (int)(rng() % M), j = (int)(rng() % N);
            double ref = 0;
            for (int k = 0; k < K; k += 8) {          // 8-way unrolled f64 dot
                double s = 0;
                for (int u = 0; u < 8; u++)
                    s += (double)dA[(size_t)i * K + k + u] * dB[(size_t)j * K + k + u];
                ref += s;
            }
            double got = hC[(size_t)i * N + j];
            double abs_e = fabs(got - ref);
            double rel_e = abs_e / (fabs(ref) + 1e-9);
            if (rel_e > 1e-4) n_bad++;
            worst_abs = fmax(worst_abs, abs_e);
            worst_rel = fmax(worst_rel, rel_e);
        }
        printf("correctness: worst |abs|=%.3e  worst rel=%.3e  (%d/400 over 1e-4)\n",
               worst_abs, worst_rel, n_bad);
        if (n_bad > 0) printf("  *** KERNEL MISMATCH ***\n");
    }

    // ---- throughput ----
    {
        hipEvent_t s, e; CK(hipEventCreate(&s)); CK(hipEventCreate(&e));
        int reps = 20;
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
