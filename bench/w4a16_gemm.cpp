// W4A16 GEMM for K100_LC (gfx926): int4 weights x fp16 activations, fp32
// accumulate, built on v_dot2_f32_f16 (2 MACs per instruction).
//
// Why this exists
// ---------------
// The deployed model is Qwen3.8-27B-UD-Q4_K_M, i.e. W4A16: 4-bit weights with
// 16-bit activations and no activation quantization.  llama.cpp has no int4
// GEMM, so every prompt-processing matmul on this card is
//
//     dequantize(Q4_K -> f16 into a global scratch buffer)   ~15% of prefill
//     hipblasGemmEx(f16 x f16 -> f16, COMPUTE_16F)           ~72% of prefill
//
// w4a16_blas.cpp measured the second step on this model's real shapes and found
// rocBLAS/Tensile is only well tuned for a few of them:
//
//   ffn_gate/up   5120->17408  27.3 TFLOPS  (81% of the fp16 peak)   x65
//   ffn_down     17408->5120   13.8          41%                     x65
//   attn_kv       5120->1024   12.7          38%                     x34
//   attn_q        5120->12288   9.5          28%                     x17
//   attn_qkv      5120->10240   9.4          28%                     x48
//   ssm_out       6144->5120    6.6          20%                     x48
//   attn_gate     5120->6144    5.1          15%                     x48
//
// which averages to ~14 TFLOPS over the whole pass.  This kernel replaces both
// steps with one fixed, shape-independent schedule and keeps the int4 weights
// packed all the way into shared memory.
//
// Arithmetic ceiling (measured, bench_isa2.cpp): v_dot2_f32_f16 sustains
// 9.45e12 instructions/s on this part, i.e. 16.9 TMAC/s = 33.8 TFLOPS.  That
// is 4.5x below the v_dot8_i32_i4 path the W4A4 prototype used, and it is the
// reason W4A16 cannot reach a 1000 t/s prefill on this chip no matter how the
// kernel is written: 25.46 GMAC/token / 16.9 TMAC/s = 664 t/s hard ceiling.
//
// Layout
// ------
//   A [M,K]  fp16 row-major (activations, already fp16 in llama.cpp)
//   B [N,K]  int4 packed 8/word, K-major per row  + fp16 scale per 32 elements
//   C [M,N]  fp32
// Shared memory keeps both operands as *dword = half2* tiles in K-major order
// (AsT[kp][row], BsT[kp][col]) so one LDS.128 feeds four dot2 pairs.
//
// build: hipcc -O3 --offload-arch=gfx926 -o w4a16_gemm w4a16_gemm.cpp
// run:   ./w4a16_gemm [M] [N] [K]
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    printf("HIP ERR %s @%d: %s\n", #x, __LINE__, hipGetErrorString(e_)); exit(1);} } while (0)

typedef unsigned int u32;
typedef __half hf;

// ---------------------------------------------------------------- config ---
#define BM 128           // tokens per block
#define BN 64            // output channels per block
#define TM 8             // rows per thread
#define TN 4             // cols per thread
#define NTHREADS 256     // (BM/TM) * (BN/TN)
#define G  32            // K elements staged per step == weight group size
#define KP (G / 2)       // half2 dwords per staged tile
#define BUF 2
#define APAD (BM + 1)    // +1 dword: kills the STS bank conflicts in staging
#define BPAD (BN + 4)

// --------------------------------------------------------------- dequant ---
// One packed dword holds 8 signed int4.  Expand to 4 dwords of half2, already
// scaled: (v0,v1), (v2,v3), (v4,v5), (v6,v7).
//
// Note: the popular byte trick `(x ^ 0x08080808) - 0x08080808` is wrong here.
// The intermediate bytes are 0x00..0x0F, so the subtraction borrows into the
// next byte whenever a nibble is >= 8 and corrupts its neighbour (measured: the
// first version of this kernel failed the f64 check at 0.45 rms).  Shifting each
// nibble into the sign position and using an arithmetic shift keeps every
// nibble independent.
static inline __device__ void deq8(u32 x, hf sc, u32 out[4]) {
    const __half2 sc2 = __half2half2(sc);
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const int v0 = ((int)(x << (28 - 8 * j))) >> 28;
        const int v1 = ((int)(x << (24 - 8 * j))) >> 28;
        __half2 h = __halves2half2(__float2half_rn((float)v0),
                                   __float2half_rn((float)v1));
        h = __hmul2(h, sc2);
        out[j] = *(u32*)&h;
    }
}

// ---------------------------------------------------------------- kernel ---
__global__ void __launch_bounds__(NTHREADS)
gemm_w4a16(const hf* __restrict__ A, const u32* __restrict__ Bp,
           const hf* __restrict__ sb, float* __restrict__ C,
           int M, int N, int K) {
    __shared__ u32 AsT[BUF][KP][APAD];
    __shared__ u32 BsT[BUF][KP][BPAD];

    const int KD = K >> 3;                    // packed int4 dwords per B row
    const int NG = K / G;                     // weight groups per row
    const int tid = threadIdx.x;
    const int ty  = tid / (BN / TN);          // output row group
    const int tx  = tid % (BN / TN);          // output col group
    const int bm  = blockIdx.y * BM;
    const int bn  = blockIdx.x * BN;

    // ---- staging assignment ---------------------------------------------
    // A: 2 threads per row, 16 halves (8 dwords) each.
    const int a_row = tid >> 1;
    const int a_kp  = (tid & 1) * (G / 4);    // 8 dwords == 4 kp... (see below)
    // B: col = tid % BN, one packed dword (8 elements) per thread per stage.
    const int b_col = tid % BN;
    const int b_d   = tid / BN;               // 0..3 -> kp = b_d*4 .. +3

    float facc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
        for (int j = 0; j < TN; j++) facc[i][j] = 0.f;

    const hf*  a0 = A + (size_t)(bm + a_row) * K;
    const u32* b0 = Bp + (size_t)(bn + b_col) * KD;
    const hf*  s0 = sb + (size_t)(bn + b_col) * NG;

    // prefetch buffers (registers): A = 8 dwords of half2, B = 4 dwords
    u32 av[8], bv[4];

    auto load_stage = [&](int s) {
        const int k0 = s * G;
#ifndef SKIP_GLOBAL
        // a_kp is an index in half2 dwords, i.e. 2 fp16 per dword
        const uint4* pa = (const uint4*)(a0 + k0 + a_kp * 2);
        *(uint4*)&av[0] = pa[0];
        *(uint4*)&av[4] = pa[1];
        deq8(b0[(k0 >> 3) + b_d], s0[s], bv);
#else
        (void)k0;
        av[0] = av[1] = av[2] = av[3] = 0x3c003c00u;
        av[4] = av[5] = av[6] = av[7] = 0x3c003c00u;
        bv[0] = bv[1] = bv[2] = bv[3] = 0x3c003c00u;
#endif
    };

    auto store_stage = [&](int buf) {
#pragma unroll
        for (int i = 0; i < 8; i++) AsT[buf][a_kp + i][a_row] = av[i];
#pragma unroll
        for (int i = 0; i < 4; i++) BsT[buf][b_d * 4 + i][b_col] = bv[i];
    };

    // ---- prologue ----
    load_stage(0);
    store_stage(0);
    __syncthreads();

    const int nsteps = NG;
    for (int s = 0; s < nsteps; s++) {
        const int cur = s & 1;

        // global load for the next stage issued before the compute, so the
        // s_waitcnt lands after it (this alone was +8.8% on the W4A4 kernel)
        if (s + 1 < nsteps) load_stage(s + 1);

        // ---- compute: 16 kp x (3 LDS.128 + 32 dot2) ----
        const u32* ap = &AsT[cur][0][ty * TM];
        const u32* bp = &BsT[cur][0][tx * TN];
        // Fragments for *two* kp at a time.  With one kp the compiler emits
        //   6x ds_read_b128 / s_waitcnt lgkmcnt(0) / 32x dot2
        // and pays the full shared-memory latency every 32 dot2s; doubling the
        // work per wait is what the SKIP_LDS measurement below is asking for.
        constexpr int Q = 2;
        u32 af[Q][TM], bf[Q][TN], afn[Q][TM], bfn[Q][TN];
        auto load_frag = [&](int kp, u32 (*a)[TM], u32 (*b)[TN]) {
#ifndef SKIP_LDS
#pragma unroll
            for (int q = 0; q < Q; q++) {
                *(uint4*)&a[q][0] = *(const uint4*)(ap + (kp + q) * APAD);
                *(uint4*)&a[q][4] = *(const uint4*)(ap + (kp + q) * APAD + 4);
                *(uint4*)&b[q][0] = *(const uint4*)(bp + (kp + q) * BPAD);
            }
#else
            (void)ap; (void)bp; (void)kp;
            for (int q = 0; q < Q; q++) {
                for (int i = 0; i < TM; i++) a[q][i] = 0x3c003c00u;
                for (int j = 0; j < TN; j++) b[q][j] = 0x3c003c00u;
            }
#endif
        };
        load_frag(0, af, bf);
#pragma unroll
        for (int kp = 0; kp < KP; kp += Q) {
            if (kp + Q < KP) load_frag(kp + Q, afn, bfn);
#pragma unroll
            for (int q = 0; q < Q; q++)
#pragma unroll
                for (int i = 0; i < TM; i++)
#pragma unroll
                    for (int j = 0; j < TN; j++)
                        asm volatile("v_dot2_f32_f16 %0, %1, %2, %0\n"
                                     : "+v"(facc[i][j]) : "v"(af[q][i]), "v"(bf[q][j]));
#pragma unroll
            for (int q = 0; q < Q; q++)
#pragma unroll
                for (int i = 0; i < TM; i++) af[q][i] = afn[q][i];
#pragma unroll
            for (int q = 0; q < Q; q++)
#pragma unroll
                for (int j = 0; j < TN; j++) bf[q][j] = bfn[q][j];
        }

        if (s + 1 < nsteps) {
            store_stage(cur ^ 1);
            __syncthreads();
        }
    }

#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
        for (int j = 0; j < TN; j++)
            C[(size_t)(bm + ty * TM + i) * N + (bn + tx * TN + j)] = facc[i][j];
}

// -------------------------------------------------------------------- host --
static inline float wval(int n, int k) {
    uint32_t h = (uint32_t)n * 2654435761u ^ (uint32_t)k * 40503u;
    h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
    return ((float)(h & 0xFFFF) / 65536.f - 0.5f) * 0.4f;
}
static inline float aval(int m, int k) {
    uint32_t h = (uint32_t)m * 2246822519u ^ (uint32_t)k * 3266489917u;
    h ^= h >> 13; h *= 2654435761u; h ^= h >> 16;
    return ((float)(h & 0xFFFF) / 65536.f - 0.5f) * 2.0f;
}

int main(int argc, char** argv) {
    int M = 512, N = 17408, K = 5120;
    if (argc > 3) { M = atoi(argv[1]); N = atoi(argv[2]); K = atoi(argv[3]); }
    printf("W4A16 GEMM  M=%d N=%d K=%d   (%.2f GMAC)\n", M, N, K,
           (double)M * N * K / 1e9);
    if (M % BM || N % BN || K % G) { printf("shapes must divide BM/BN/G\n"); return 1; }

    // ---- host-side packing: int4 + one fp16 scale per G elements ----
    const int KD = K / 8, NG = K / G;
    std::vector<u32> hBp((size_t)N * KD);
    std::vector<hf>  hsb((size_t)N * NG);
    std::vector<hf>  hA((size_t)M * K);
    for (size_t i = 0; i < hA.size(); i++) hA[i] = __float2half(aval(i / K, i % K));
    for (int n = 0; n < N; n++) {
        for (int g = 0; g < NG; g++) {
            float amax = 0.f;
            for (int i = 0; i < G; i++) amax = fmaxf(amax, fabsf(wval(n, g * G + i)));
            float sc = amax > 0.f ? amax / 7.f : 1.f;
            hsb[(size_t)n * NG + g] = __float2half(sc);
            for (int u = 0; u < G; u += 8) {
                u32 w = 0;
                for (int t = 0; t < 8; t++) {
                    int v = (int)lrintf(wval(n, g * G + u + t) / sc);
                    if (v > 7) v = 7; if (v < -8) v = -8;
                    w |= (u32)(v & 0xF) << (4 * t);
                }
                hBp[(size_t)n * KD + (g * G + u) / 8] = w;
            }
        }
    }

    hf *dA, *dsb; u32 *dBp; float* dC;
    CK(hipMalloc(&dA, hA.size() * 2));
    CK(hipMalloc(&dBp, hBp.size() * 4));
    CK(hipMalloc(&dsb, hsb.size() * 2));
    CK(hipMalloc(&dC, (size_t)M * N * 4));
    CK(hipMemcpy(dA, hA.data(), hA.size() * 2, hipMemcpyHostToDevice));
    CK(hipMemcpy(dBp, hBp.data(), hBp.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(dsb, hsb.data(), hsb.size() * 2, hipMemcpyHostToDevice));

    dim3 grid(N / BN, M / BM);
    {
        hipFuncAttributes at;
        CK(hipFuncGetAttributes(&at, (const void*)gemm_w4a16));
        int maxblk = 0, nsm = 0;
        hipDeviceProp_t dp; CK(hipGetDeviceProperties(&dp, 0)); nsm = dp.multiProcessorCount;
        CK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&maxblk, (const void*)gemm_w4a16,
                                                        NTHREADS, 0));
        printf("kernel: %d VGPR, %zu B smem/block, %d blocks/CU (%d waves), grid=%dx%d\n",
               at.numRegs, at.sharedSizeBytes, maxblk,
               maxblk * NTHREADS / 64, grid.x, grid.y);
    }
    gemm_w4a16<<<grid, NTHREADS>>>(dA, dBp, dsb, dC, M, N, K);
    CK(hipDeviceSynchronize());

    // ---- correctness: f64 reference recomputed from the packed int4 ----
    {
        std::vector<float> hC((size_t)M * N);
        CK(hipMemcpy(hC.data(), dC, (size_t)M * N * 4, hipMemcpyDeviceToHost));
        double worst = 0, rms = 0;
        const int NT = 200;
        for (int t = 0; t < NT; t++) {
            int i = (t * 7919) % M, j = (t * 104729 + 13) % N;
            double ref = 0;
            for (int k = 0; k < K; k++) {
                u32 w = hBp[(size_t)j * KD + k / 8];
                int nib = (int)((w >> (4 * (k % 8))) & 0xF);
                int sgn = (nib & 0x8) ? nib - 16 : nib;
                // the kernel rounds the *scaled* weight to fp16, so the
                // reference has to do the same or the test only measures that
                // rounding (it is ~6e-4 of rms|c|, which is expected and not a
                // kernel bug)
                double wv = __half2float(
                    __float2half(__half2float(hsb[(size_t)j * NG + k / G]) * (float)sgn));
                ref += wv * (double)__half2float(hA[(size_t)i * K + k]);
            }
            worst = fmax(worst, fabs((double)hC[(size_t)i * N + j] - ref));
            rms += ref * ref;
        }
        rms = sqrt(rms / NT);
        printf("correctness: worst |abs err| = %.3e  vs rms|c| = %.3e  ratio %.2e %s\n",
               worst, rms, worst / rms, worst / rms > 1e-4 ? "*** CHECK ***" : "ok");
    }

    {
        hipEvent_t s, e; CK(hipEventCreate(&s)); CK(hipEventCreate(&e));
        const int reps = 20;
        gemm_w4a16<<<grid, NTHREADS>>>(dA, dBp, dsb, dC, M, N, K);
        CK(hipDeviceSynchronize());
        CK(hipEventRecord(s));
        for (int r = 0; r < reps; r++)
            gemm_w4a16<<<grid, NTHREADS>>>(dA, dBp, dsb, dC, M, N, K);
        CK(hipEventRecord(e)); CK(hipEventSynchronize(e));
        float ms = 0; CK(hipEventElapsedTime(&ms, s, e)); ms /= reps;
        double macs = (double)M * N * K;
        double tmac = macs / (ms * 1e-3) / 1e12;
        printf("throughput : %.3f ms -> %.2f TMAC/s = %.1f TFLOPS  (%.1f%% of the "
               "16.9 TMAC/s dot2 ceiling)\n", ms, tmac, tmac * 2,
               100.0 * tmac / 16.9);
    }
    return 0;
}
