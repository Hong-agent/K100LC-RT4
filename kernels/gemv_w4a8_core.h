// W4A8 解码 GEMV 的内核部分（单一来源）。独立基准与实测见 kernels/gemv_int4.hip
// 与 docs/GEMV.md：int4 权重（每 128 组 f16 尺度）× int8 激活，M = 1..4。
//
// 算法：RT4 的 int4 是有符号 q∈[-8,7]，取 u = q+8 ∈ [0,15]，则
//   Σ q·a = Σ u·a - 8·Σa（u 与 a 都是合法 int8，直接走 v_dot4_i32_i8）。
//
// 运行时的选择：**解码路径一律走这个内核**（激活 8bit），因为解码是带宽瓶颈，
// 权重只读一次，int8 激活不额外花带宽，却把每层误差从 int4 激活的 12~16%
// 降到 ~1.4%（实测，见 tools/ref_stages.py）。
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>

#define GRP 128                 // 量化组大小（与 RT4 格式一致）
#ifndef NTHREADS
#define NTHREADS 256
#endif
#define NWARP (NTHREADS / 64)
// ROWS 是模板参数（每个 warp 处理几行）：
//   实测 M=1/2 时 ROWS=1 最快；M=3/4（MTP 校验批）时 ROWS=2 让两份权重行共享同一份
//   激活加载，带宽从 322/260 提到 358/317 GB/s（+11%/+22%）。ROWS≥3 会寄存器溢出，
//   实测掉到 20 GB/s，不能开。

static inline __device__ int dot4_i8(int a, int b) {
    int r, z = 0;
    asm volatile("v_dot4_i32_i8 %0, %1, %2, %3" : "=v"(r) : "v"(a), "v"(b), "v"(z));
    return r;
}

// ---------------------------------------------------------------- 激活量化 --
// 输入 a[M][K] (f32) → 输出 aq_even/aq_odd (int8, 交错拆分成两半)、
// ascale[M][K/GRP] (f32)、asum[M][K/GRP] (int32，Σq，供 -8Σa 修正用)
__global__ void quant_act(const float* __restrict__ a, int8_t* __restrict__ aq_even,
                          int8_t* __restrict__ aq_odd, float* __restrict__ ascale,
                          int* __restrict__ asum, int M, int K) {
    int g = blockIdx.x;                 // 组号
    int m = blockIdx.y;                 // token
    int t = threadIdx.x;
    if (g * GRP >= K) return;
    const float* src = a + (size_t)m * K + (size_t)g * GRP;
    int n = min(GRP, K - g * GRP);
    // 组内 amax（warp 归约）
    float amax = 0.f;
    for (int i = t; i < n; i += blockDim.x) amax = fmaxf(amax, fabsf(src[i]));
    for (int off = 16; off; off >>= 1) amax = fmaxf(amax, __shfl_xor(amax, off));
    float s = amax > 0.f ? amax / 127.f : 1.f;
    if (t == 0) ascale[m * (K / GRP) + g] = s;
    __shared__ float ss;
    if (t == 0) ss = s;
    __syncthreads();
    s = ss;
    int part = 0;
    // 两个半字节流：偶下标进 q_even，奇下标进 q_odd（与权重 lo/hi 对齐）
    for (int i = t; i < n; i += blockDim.x) {
        float v = src[i] / s;
        int q = (int)lrintf(v);
        q = q > 127 ? 127 : (q < -128 ? -128 : q);
        part += q;
        int8_t* dst = ((g * GRP + i) & 1) ? aq_odd : aq_even;
        dst[(size_t)m * (K / 2) + (g * GRP + i) / 2] = (int8_t)q;
    }
    // 组内求和
    for (int off = 16; off; off >>= 1) part += __shfl_xor(part, off);
    if (t == 0) asum[m * (K / GRP) + g] = part;
}

// ---------------------------------------------------------------- GEMV 主体 --
// wq: [N, K/2] int4 打包；ws: [N, K/GRP] f16 尺度；
// ae/ao: [M, K/2] int8；asc: [M, K/GRP] f32；asu: [M, K/GRP] int32；out: [M, N]
//
// SMEM=true（只在 M==1 的解码路径用）：把这一行激活（ae/ao/asc/asu，总共
// K + K/16 字节）先协作搬进共享内存。原来每个 warp 每个迭代都要自己发 2 个
// 128-bit 全局读去取激活，4 个 warp 就是在重复读同一份数据；搬进 LDS 之后
// 全局侧只剩权重那 1/3，LDS.128 也远比 LDG 便宜（实测见 docs/GEMV.md）。
template <int M, bool SMEM, int NROW>
__global__ void gemv_w4a8(const uint32_t* __restrict__ wq, const uint16_t* __restrict__ ws,
                          const uint32_t* __restrict__ ae, const uint32_t* __restrict__ ao,
                          const float* __restrict__ asc, const int* __restrict__ asu,
                          float* __restrict__ out, int N, int K) {
    const int lane = threadIdx.x & 63;
    const int warp = threadIdx.x >> 6;
    // 一个 warp 处理 NROW 行：行间共享同一份激活加载，且多条点积链互相独立，
    // 用来掩盖访存延迟（M=4 时每个 warp 本来只有 2.5 次迭代，延迟完全暴露）。
    const int row0 = (blockIdx.x * NWARP + warp) * NROW;
    const int KD = K / 8;                            // 每行 dword 数
    const int NG = K / GRP;                          // 每行组数
    if (row0 >= N) return;
    const int nr = (row0 + NROW <= N) ? NROW : (N - row0);

    extern __shared__ uint32_t smem[];
    const uint32_t *ea = ae, *oa = ao;
    const float* sca = asc;
    const int* sua = asu;
    if (SMEM) {
        uint32_t* s_e = smem;
        uint32_t* s_o = smem + KD;
        float* s_sc = reinterpret_cast<float*>(smem + 2 * KD);
        int* s_su = reinterpret_cast<int*>(s_sc + NG);
        for (int i = threadIdx.x; i < KD; i += blockDim.x) { s_e[i] = ae[i]; s_o[i] = ao[i]; }
        for (int i = threadIdx.x; i < NG; i += blockDim.x) { s_sc[i] = asc[i]; s_su[i] = asu[i]; }
        __syncthreads();
        ea = s_e; oa = s_o; sca = s_sc; sua = s_su;
    }

    float acc[NROW][M];
#pragma unroll
    for (int r = 0; r < NROW; r++)
#pragma unroll
        for (int m = 0; m < M; m++) acc[r][m] = 0.f;

    // 关键实现细节：每组（128 个权重 = 16 个 dword）要乘自己的尺度，
    // 组内 16 个 lane 必须做 4 步 shuffle 归约。若每次只处理 1 个 dword，
    // 这条 4 步依赖链会完全暴露（实测：纯读 688GB/s，带计算只剩 304GB/s）。
    // 所以这里一次处理 4 个 dword（跨 4 个组），得到 4 条互相独立的归约链，
    // 编译期展开后延迟可以互相掩盖。
    // 布局要点：每个 lane 一次拿 uint4（16 字节 = 4 个 dword = 32 个权重），
    // 而一个量化组是 128 个权重 = 16 个 dword = **正好 4 个 lane**。
    // 于是: (1) warp 内 64 个 lane 的 uint4 地址连续（64×16=1024B，完全合并）；
    //       (2) 组内归约只需 2 步 shuffle（而不是每个 dword 做 4 步）。
    // 之前的版本每个 lane 只拿 1 个 dword，组内 16 个 lane 要 4 步归约，
    // 实测那条依赖链把带宽从 687GB/s 压到 342GB/s（NO_REDUCE 对照组 477GB/s）。
    // 前提：K 必须是 128 的倍数（本模型全部满足：5120/17408/10240/6144/12288），
    // 这样组边界与 4-lane 边界对齐，尾巴不会切进组内部。
    const int LANES_PER_GROUP = 4;                 // 128 权重 / 32 权重每 lane
    const int STEP = 64 * 4;                       // 每 warp 每次迭代覆盖 256 个 dword
    for (int base = 0; base < KD; base += STEP) {
        const int d0 = base + lane * 4;
        // 用谓词而不是 break：提前退出会让编译器不敢做软件流水（预取下一轮的
        // 权重），而每个 warp 本来只有 2~3 次迭代，延迟完全暴露。
        // 组边界与 4-lane 边界对齐（K 是 128 的倍数），所以谓词在组内是一致的。
        const bool ok = (d0 + 4 <= KD);
        uint32_t lo[NROW][4], hi[NROW][4];
#pragma unroll
        for (int r = 0; r < NROW; r++) {
            if (r >= nr) break;
            const uint4* wp = reinterpret_cast<const uint4*>(wq + (size_t)(row0 + r) * KD + d0);
#ifdef USE_LDCS
            const uint4 w4 = ok ? __ldcs(wp) : make_uint4(0, 0, 0, 0);
#else
            const uint4 w4 = ok ? *wp : make_uint4(0, 0, 0, 0);
#endif
#pragma unroll
            for (int i = 0; i < 4; i++) {
                // 存的是二进制补码 int4（-1 存成 0xF），u = code + 8 等于把每个
                // 半字节的符号位翻过来（u = n ^ 8）；缺这个 XOR 时 -8·Σa
                // 修正会作用在错误的 u 上（实测差 1.67，与 Σa 同阶）。
                uint32_t w = (&w4.x)[i];
                lo[r][i] = (w & 0x0F0F0F0Fu) ^ 0x08080808u;
                hi[r][i] = ((w >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u;
            }
        }
        int p[NROW][M];
#pragma unroll
        for (int m = 0; m < M; m++) {
            const size_t eoff = SMEM ? 0 : (size_t)m * (K / 8);
            const uint4 e4 = *reinterpret_cast<const uint4*>(ea + eoff + d0);
            const uint4 o4 = *reinterpret_cast<const uint4*>(oa + eoff + d0);
            #pragma unroll
            for (int r = 0; r < NROW; r++) {
                if (r >= nr) break;
                int v0 = 0, v1 = 0;
#pragma unroll
                for (int i = 0; i < 4; i += 2) {
                    v0 += dot4_i8((int)lo[r][i], (int)(&e4.x)[i]) +
                          dot4_i8((int)hi[r][i], (int)(&o4.x)[i]);
                    v1 += dot4_i8((int)lo[r][i + 1], (int)(&e4.x)[i + 1]) +
                          dot4_i8((int)hi[r][i + 1], (int)(&o4.x)[i + 1]);
                }
                p[r][m] = v0 + v1;
            }
        }
#pragma unroll
        for (int r = 0; r < NROW; r++) {
            if (r >= nr) break;
#pragma unroll
            for (int m = 0; m < M; m++) {
                int v = p[r][m];
#pragma unroll
                for (int off = 1; off < LANES_PER_GROUP; off <<= 1) v += __shfl_xor(v, off);
                p[r][m] = v;
            }
        }
        if (ok && (lane & (LANES_PER_GROUP - 1)) == 0) {
            const int g = d0 / 16;                 // 本 lane 所属量化组
#pragma unroll
            for (int r = 0; r < NROW; r++) {
                if (r >= nr) break;
                float sw = __half2float(*reinterpret_cast<const __half*>(
                    ws + (size_t)(row0 + r) * NG + g));
#pragma unroll
                for (int m = 0; m < M; m++)
                    acc[r][m] += sw * sca[(SMEM ? 0 : (size_t)m * NG) + g] *
                                 (float)(p[r][m] - 8 * sua[(SMEM ? 0 : (size_t)m * NG) + g]);
            }
        }
    }
    // 最后的 warp 归约（只有 lane0/lane16 有值，但统一归约最省事）
#pragma unroll
    for (int r = 0; r < NROW; r++) {
        if (r >= nr) break;
#pragma unroll
        for (int m = 0; m < M; m++) {
            float v = acc[r][m];
            for (int off = 32; off; off >>= 1) v += __shfl_xor(v, off);
            if (lane == 0) out[(size_t)m * N + (row0 + r)] = v;
        }
    }
}
