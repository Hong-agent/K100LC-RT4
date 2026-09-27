// W4A4 解码 GEMV 的内核部分（单一来源）。独立基准在 kernels/gemv_w4a4.hip，
// 运行时的 src/kernels.hip 也 include 这个文件，避免两份实现漂移。
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>

#define GRP 128                 // 权重分组（与 RT4 一致）
#define SG 32                   // 激活子分组 = 一个 lane 一次 uint4 覆盖的权重数
#define NTHREADS 256
#define NWARP (NTHREADS / 64)
#ifndef ROWS
#define ROWS 1                  // 每个 warp 处理几行
#endif

static inline __device__ int dot8_i4(int acc, int a, int b) {
    asm volatile("v_dot8_i32_i4 %0, %1, %2, %0\n" : "+v"(acc) : "v"(a), "v"(b));
    return acc;
}

// ---------------------------------------------------------------- 激活量化 --
// f32 a[M][K] → int4 打包 aq[M][K/2]（低半字节 = 偶数 k，与权重同序）
//              + 每 32 个元素一个 f32 尺度 as[M][K/32]
__global__ void quant_act4(const float* __restrict__ a, uint8_t* __restrict__ aq,
                           float* __restrict__ as, int M, int K) {
    const int g = blockIdx.x;                    // 32 元素一组
    const int m = blockIdx.y;
    const int t = threadIdx.x;                   // 0..31
    const int i = g * SG + t;                    // 全局元素下标
    if (i >= K) return;
    const float* src = a + (size_t)m * K + g * SG;
    float amax = fabsf(src[t]);
    for (int off = 16; off; off >>= 1) amax = fmaxf(amax, __shfl_xor(amax, off));
    const float s = amax > 0.f ? amax / 7.f : 1.f;
    if (t == 0) as[(size_t)m * (K / SG) + g] = s;
    float v = src[t] / s;
    int q = (int)lrintf(v);
    q = q > 7 ? 7 : (q < -8 ? -8 : q);
    // 偶数元素拿低半字节、奇数拿高半字节；由偶数线程合成整字节，避免写竞争
    const int qn = __shfl_down(q, 1);
    if ((t & 1) == 0)
        aq[(size_t)m * (K / 2) + i / 2] = (uint8_t)((q & 0xF) | ((qn & 0xF) << 4));
}

// ---------------------------------------------------------------- GEMV 主体 --
template <int M>
__global__ void gemv_w4a4(const uint32_t* __restrict__ wq, const uint16_t* __restrict__ ws,
                          const uint32_t* __restrict__ aq, const float* __restrict__ as,
                          float* __restrict__ out, int N, int K) {
    const int lane = threadIdx.x & 63;
    const int warp = threadIdx.x >> 6;
    const int row0 = (blockIdx.x * NWARP + warp) * ROWS;
    const int KD = K / 8;                          // 每行 dword 数
    const int NG = K / GRP;
    if (row0 >= N) return;
    const int nr = (row0 + ROWS <= N) ? ROWS : (N - row0);

    float acc[ROWS][M];
#pragma unroll
    for (int r = 0; r < ROWS; r++)
#pragma unroll
        for (int m = 0; m < M; m++) acc[r][m] = 0.f;

    // 每个 lane 一次拿 4 个 dword（= 32 个权重 = 16 字节），64 个 lane 连续 1KB。
    // 一个 128 权重的量化组正好由 4 个 lane 覆盖；激活尺度按 32 个一组，
    // 正好是 lane 私有 → 两者相乘即可，零 shuffle。
    const int STEP = 64 * 4;
    for (int base = 0; base < KD; base += STEP) {
        const int d0 = base + lane * 4;
        const bool ok = (d0 + 4 <= KD);
        float sa[M];
#pragma unroll
        for (int m = 0; m < M; m++) {
            // 注意：K 不是 STEP 的整数倍时，尾部 d0 会越界（w4 被置 0，但越界读到的
            // 垃圾若是 NaN/Inf，0*NaN 会把整行结果污染成 NaN）——所以这里必须夹住。
            sa[m] = ok ? as[(size_t)m * (K / SG) + d0 / 4] : 0.f;
        }
        uint4 w4[ROWS];
#pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const uint4* wp = reinterpret_cast<const uint4*>(wq + (size_t)(row0 + r) * KD + d0);
            if (r < nr && ok) {
#ifdef NT_LOAD
                // 非暂存加载：权重只读一次，别把 L1 占满（激活要常驻 L1）
                w4[r] = __builtin_nontemporal_load(wp);
#else
                w4[r] = *wp;
#endif
            } else {
                w4[r] = make_uint4(0, 0, 0, 0);
            }
        }
#pragma unroll
        for (int m = 0; m < M; m++) {
#ifdef NO_ACT_LOAD
            const uint4 a4 = make_uint4(0x11111111u, 0x11111111u, 0x11111111u, 0x11111111u);
#else
            const uint4 a4 = ok
                ? *reinterpret_cast<const uint4*>(aq + (size_t)m * (K / 8) + d0)
                : make_uint4(0, 0, 0, 0);
#endif
#pragma unroll
            for (int r = 0; r < ROWS; r++) {
                if (r >= nr) break;
                int v0 = 0, v1 = 0;
#pragma unroll
                for (int i = 0; i < 4; i += 2) {
                    v0 = dot8_i4(v0, (int)(&w4[r].x)[i], (int)(&a4.x)[i]);
                    v1 = dot8_i4(v1, (int)(&w4[r].x)[i + 1], (int)(&a4.x)[i + 1]);
                }
                const int v = v0 + v1;
                // 权重组尺度（每 128 一组）与激活子组尺度（每 32 一组）相乘
                // 注意：d0 越过行尾时（K 不是 STEP 的整数倍），d0/16 会跑到下一行的
                // 尺度里；对最后一行就是**越过整张尺度表**，读到的字节若是 Inf/NaN，
                // 0*NaN 会把该行结果整行污染成 NaN。所以这里也必须夹住（w4 已置 0）。
                float sw = ok ? __half2float(*reinterpret_cast<const __half*>(
                    ws + (size_t)(row0 + r) * NG + d0 / 16)) : 0.f;
                acc[r][m] += sw * sa[m] * (float)v;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < ROWS; r++) {
        if (r >= nr) break;
#pragma unroll
        for (int m = 0; m < M; m++) {
            float v = acc[r][m];
            for (int off = 32; off; off >>= 1) v += __shfl_xor(v, off);
            if (lane == 0) out[(size_t)m * N + (row0 + r)] = v;
        }
    }
}
