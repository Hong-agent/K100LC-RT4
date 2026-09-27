// INT4 GEMM 内核部分（单一来源）。独立基准（含 QG 消融与 f64 对照）在
// bench/int4_gemm2.cpp，运行时 src/kernels.hip 也 include 本文件。
// 注意：尺度布局是 **组优先** sb[g*N + n]（int4_gemm2 的实测结论），
// 运行时会为 GEMM 路径单独准备这份尺度（GEMV 路径用 RT4 原生的行优先）。
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

#define BM 128          // tokens per block
#define BN 64           // output channels per block
#define GD 16           // staged dwords of K per pipeline step (8 elements each)
#define G  (GD * 8)     // elements staged per step
#ifndef QG
#define QG 128          // quantization group (elements): 128 / 256 / 512 / 1024
#endif
#define SPG (QG / G)    // staging steps per quantization group
#define TM 8            // output rows per thread
#define TN 4            // output cols per thread
#define NTHREADS 256    // (BM/TM) * (BN/TN)

// ---------------------------------------------------------------- quantize --
static inline float quant_group(const float* x, int n, int8_t* q) {
    float amax = 0.f;
    for (int i = 0; i < n; i++) amax = fmaxf(amax, fabsf(x[i]));
    float scale = amax > 0.f ? amax / 7.0f : 1.f;
    for (int i = 0; i < n; i++) {
        int v = (int)lrintf(x[i] / scale);
        if (v > 7) v = 7;
        if (v < -8) v = -8;
        q[i] = (int8_t)v;
    }
    return scale;
}

static inline u32 pack8(const int8_t* q) {
    u32 w = 0;
    for (int i = 0; i < 8; i++) w |= (u32)(q[i] & 0xF) << (4 * i);
    return w;
}

template<int N>
static inline __device__ void ldv(const u32* __restrict__ p, u32* v) {
#pragma unroll
    for (int m = 0; m < N; m += 4) *(uint4*)&v[m] = *(const uint4*)(p + m);
}

// ---------------------------------------------------------------- kernel ---
// C[M,N] = A[M,K] @ B[N,K]^T, A/B int4-packed with one scale per QG elements.
//
// Shared-memory layout notes:
//   * tiles are K-major (AsT[kd][row]) so a thread's TM A-dwords are contiguous
//     and come back in 2 LDS.128; with TN=4 the B-dwords need one more.
//   * the scales are staged as whole tiles (one float4 per participating
//     thread) and land in a buffer keyed by quantization group, so they ride
//     the same AHEAD=1 pipeline as the int4 tiles.
//   * BUF=2 is the maximum that still allows 2 blocks/CU, which is what the
//     104-VGPR footprint permits anyway (measured: forcing 3 blocks/CU via
//     __launch_bounds__ spills and drops the kernel to 8.5 TMAC/s).
__global__ void __launch_bounds__(NTHREADS)
gemm_w4a4(const u32* __restrict__ Ap, const u32* __restrict__ Bp,
          const float* __restrict__ sa, const float* __restrict__ sb,
          float* __restrict__ C, int M, int N, int K) {
    constexpr int NTC = BN / TN;           // thread columns
    constexpr int AD  = BM * GD / NTHREADS;  // A dwords staged per thread
    constexpr int BD  = BN * GD / NTHREADS;  // B dwords staged per thread
    constexpr int TPA = GD / AD;             // threads per A row
    constexpr int TPB = GD / BD;             // threads per B row
    constexpr int BUF = 2;                   // shared stages

    __shared__ u32 AsT[BUF][GD][BM + 4];
    __shared__ u32 BsT[BUF][GD][BN + 4];
    __shared__ float SAs[BUF][BM];
    __shared__ float SBs[BUF][BN];

    const int KD = K >> 3;                 // packed dwords per row
    const int tid = threadIdx.x;
    const int ty  = tid / NTC;             // output row group
    const int tx  = tid % NTC;             // output col group
    const int bm  = blockIdx.y * BM;
    const int bn  = blockIdx.x * BN;

    const int arow = tid / TPA, aseg = (tid % TPA) * AD;
    const int brow = tid / TPB, bseg = (tid % TPB) * BD;
    const int nsteps = K / G;

    const u32* ap0 = Ap + (size_t)(bm + arow) * KD + aseg;
    const u32* bp0 = Bp + (size_t)(bn + brow) * KD + bseg;

    int   iacc[TM][TN];
    float facc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
        for (int j = 0; j < TN; j++) { iacc[i][j] = 0; facc[i][j] = 0.f; }

    u32 av[AD], bv[BD];
    float4 sv = make_float4(0.f, 0.f, 0.f, 0.f);

    // scale staging: threads [0, BM/4) cover A's rows, the next BN/4 cover B's
    constexpr int WA = BM / 4;
    const int sc_idx = tid - WA;
    auto sc_ld = [&](int s) {                    // group s/SPG, global -> regs
        if (s % SPG == 0) {
            const int g = s / SPG;
            if (tid < WA) sv = *(const float4*)&sa[(size_t)g * M + bm + tid * 4];
            else if (sc_idx < BN / 4)
                sv = *(const float4*)&sb[(size_t)g * N + bn + sc_idx * 4];
        }
    };
    auto sc_st = [&](int s) {                    // registers -> shared
        if (s % SPG == 0) {
            const int buf = (s / SPG) % BUF;
            if (tid < WA) *(float4*)&SAs[buf][tid * 4] = sv;
            else if (sc_idx < BN / 4) *(float4*)&SBs[buf][sc_idx * 4] = sv;
        }
    };

    // ---- prologue: stage K-tile 0 (and its scales) ----
    {
        ldv<AD>(ap0, av);
        ldv<BD>(bp0, bv);
        sc_ld(0); sc_st(0);
#pragma unroll
        for (int m = 0; m < AD; m++) AsT[0][aseg + m][arow] = av[m];
#pragma unroll
        for (int m = 0; m < BD; m++) BsT[0][bseg + m][brow] = bv[m];
    }
    __syncthreads();

    for (int s = 0; s < nsteps; s++) {
        const int cur = s & 1;

        // ---- next K-tile: global -> registers only; the STS comes after the
        //      dot8 loop so the load latency hides behind it ----
        if (s + 1 < nsteps) {
            ldv<AD>(ap0 + (size_t)(s + 1) * (G / 8), av);
            ldv<BD>(bp0 + (size_t)(s + 1) * (G / 8), bv);
            sc_ld(s + 1);
        }

        const u32* ap = &AsT[cur][0][ty * TM];
        const u32* bp = &BsT[cur][0][tx * TN];

        // ---- int32 accumulation, fragments pipelined one kd ahead ----
        {
            u32 a8[TM], b4[TN], a8n[TM], b4n[TN];
#pragma unroll
            for (int i = 0; i < TM; i += 4)
                *(uint4*)&a8[i] = *(const uint4*)(ap + i);
#pragma unroll
            for (int j = 0; j < TN; j += 4)
                *(uint4*)&b4[j] = *(const uint4*)(bp + j);
#pragma unroll
            for (int kd = 0; kd < GD; kd++) {
                if (kd + 1 < GD) {
#pragma unroll
                    for (int i = 0; i < TM; i += 4)
                        *(uint4*)&a8n[i] = *(const uint4*)(ap + (kd + 1) * (BM + 4) + i);
#pragma unroll
                    for (int j = 0; j < TN; j += 4)
                        *(uint4*)&b4n[j] = *(const uint4*)(bp + (kd + 1) * (BN + 4) + j);
                }
#pragma unroll
                for (int i = 0; i < TM; i++)
#pragma unroll
                    for (int j = 0; j < TN; j++)
                        asm volatile("v_dot8_i32_i4 %0, %1, %2, %0\n"
                                     : "+v"(iacc[i][j]) : "v"(a8[i]), "v"(b4[j]));
#pragma unroll
                for (int i = 0; i < TM; i++) a8[i] = a8n[i];
#pragma unroll
                for (int j = 0; j < TN; j++) b4[j] = b4n[j];
            }
        }

        // ---- fold f32 *= (int) at the end of every quantization group ----
        if ((s % SPG) == SPG - 1) {
            const int sbuf = (s / SPG) % BUF;
#pragma unroll
            for (int i = 0; i < TM; i++)
#pragma unroll
                for (int j = 0; j < TN; j++) {
                    facc[i][j] = fmaf((float)iacc[i][j],
                                      SAs[sbuf][ty * TM + i] * SBs[sbuf][tx * TN + j],
                                      facc[i][j]);
                    iacc[i][j] = 0;
                }
        }

        // ---- land the prefetched tile ----
        if (s + 1 < nsteps) {
            sc_st(s + 1);
#pragma unroll
            for (int m = 0; m < AD; m++) AsT[cur ^ 1][aseg + m][arow] = av[m];
#pragma unroll
            for (int m = 0; m < BD; m++) BsT[cur ^ 1][bseg + m][brow] = bv[m];
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
        for (int j = 0; j < TN; j++)
            C[(size_t)(bm + ty * TM + i) * N + (bn + tx * TN + j)] = facc[i][j];
}
