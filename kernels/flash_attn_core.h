// int4 FlashAttention 内核部分（单一来源）。独立基准与三重正确性对照在
// kernels/flash_attn_int4.hip，运行时 src/kernels.hip 也 include 本文件。
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

#define HD   256           // head_dim（Qwen3.8-27B 的全注意力层）
#define KD   (HD / 8)      // 每行 32 个 dword
#define QG   128           // Q/K 沿 d 的量化组
#define NQG  (HD / QG)     // 2 组

// ---------------------------------------------------------------- 分块参数 ---
#ifndef BM
#define BM 64              // 每个 block 处理 64 行 query
#endif
#ifndef BN
#define BN 64              // 每个 KV tile 64 个 key（实测 64 比 128 快，见 docs/FLASH-ATTN.md）
#endif
#ifndef NT
#define NT 256             // 每 block 线程数（256 时 4 wave/CU 且不溢出寄存器）
#endif

// QK 阶段：每线程 TM 行 × TN 个 key
#ifndef TM
#define TM 2
#endif
#define TN 8
#define RG (BM / TM)       // 行组数
#define CG (BN / TN)       // key 组数
// PV 阶段：每线程 TMO 行 × TNO 个 d
#define TNO 8
#define TMO (BM * HD / NT / TNO)   // 由 (BM,NT) 推出，保证 oacc 与 oiacc 都是 BM*HD/NT 个
static_assert(BM * HD % (NT * TNO) == 0, "PV 每线程元素数不是 TNO 的整数倍");
#define RGO (BM / TMO)     // 行组数
#define CGO (HD / TNO)     // d 组数
static_assert(RG * CG == NT, "QK 线程映射不整除");
static_assert(RGO * CGO == NT, "PV 线程映射不整除");
static_assert(BN % 8 == 0 && QG % 8 == 0, "打包粒度必须是 8");

#define KDT (BN / 8)       // PV 的收缩维 dword 数（每 dword 8 个 key）
#define L2E 1.4426950408889634f

// 共享内存 pad：既要消 bank 冲突，又要保住 LDS.128 的 16B 对齐（所以 pad 是 4 的倍数）
#define QPAD 4
#define KPAD 4
#define VPAD 4
#define PPAD 4

// tile 列序交错：内核里每个线程要「连续 8 列」的 8 个 dword（K 的 key、V 的 dim），
// 而相邻 lane 的起点相隔 8 个 dword。按自然序存放时，一个 phase 的 16 个 lane 只会
// 用到 16 个 bank，LDS.128 退化成 4 路冲突。把每组 8 列拆成两个 4 列半段、前后半行
// 交错存放后，lane 的地址间隔是 4 个 dword，一个 phase 正好铺满 32 个 bank。
// 实测（BM=64/BN=64/NT=256）：QK 25.2 vs 22.3 TMAC/s。
__device__ __forceinline__ int xcol(int c, int n) {  // 自然列号 → 物理列号
    const int q = c & 7, t = c >> 3;
    return q < 4 ? 4 * t + q : n / 2 + 4 * t + q - 4;
}

static inline __device__ int dot8(int acc, int a, int b) {
    asm volatile("v_dot8_i32_i4 %0, %1, %2, %0\n" : "+v"(acc) : "v"(a), "v"(b));
    return acc;
}

// ================================ 内核 =======================================
// 时序（每 tile 两次 __syncthreads()）：
//   [t] LDG K/V(t+1)→寄存器   （跟在 QK(t) 前面发，延迟被 QK 盖住）
//       QK(t)  读 Qs / KsT(=K(t))
//       软最大 → 写 PsT(t)、RowAlpha、RowPB、RowL
//   BAR(a)  ← P 就绪；同时保证没人再读 KsT(t)
//       STS KsT ← K(t+1)
//       PV(t)  读 PsT(t) / Vs(=V(t)) / VscT(t)
//   BAR(b)  ← 保证没人再读 Vs(t)、PsT(t)
//       STS Vs ← V(t+1)     （下一次迭代的 QK(t+1) 用它；BAR(a) 之后对 PV(t+1) 可见）
__global__ void __launch_bounds__(NT)
fa_int4(const u32* __restrict__ Qg, const float* __restrict__ Qscg,
        const u32* __restrict__ Kg, const float* __restrict__ Kscg,
        const u32* __restrict__ Vc, const float* __restrict__ Vscg,
        float* __restrict__ Out,
        int n_q, int n_kv, int off, int n_tile, int ratio, int kv_cap)
{
    __shared__ u32   QsT[KD][BM + QPAD];        // [d/8][row]
    __shared__ u32   KsT[KD][BN + KPAD];        // [d/8][key]
    __shared__ u32   Vs[KDT][HD + VPAD];        // [key/8][d]  （V 已转置，沿 key 打包）
    __shared__ u32   PsT[KDT][BM + PPAD];       // [key/8][row]
    __shared__ float QscT[NQG][BM];
    __shared__ float KscT[NQG][BN];
    __shared__ float VscT[HD];
    __shared__ float RowRef[BM];                // running max（参考点）
    __shared__ float RowL[BM];                  // 分母
    __shared__ float RowAlpha[BM];              // exp2(r_old - r_new)
    __shared__ float RowPB[BM];                 // beta/7（折进 PV 的行尺度）

    const int tid = threadIdx.x;
    const int head = blockIdx.y;
    const int vh   = head / ratio;              // GQA：ratio 个 q 头共用一个 kv 头
    // 注意：KV cache 的**行距是分配容量 kv_cap**（运行时的 max_ctx），不是当前
    // 已缓存长度 n_kv。用 n_kv 当行距时第 0 个头还看得过去（vh=0 偏移为 0），
    // 其余头会整段读错（GQA 下 4 个 kv 头全错）。独立基准里 kv_cap == n_kv，
    // 所以两种用法都对。
    Qg    += (size_t)head * n_q * KD;
    Qscg  += (size_t)head * NQG * n_q;
    Kg    += (size_t)vh * kv_cap * KD;
    Kscg  += (size_t)vh * NQG * kv_cap;
    Vc    += (size_t)vh * (kv_cap / 8) * HD;
    Vscg  += (size_t)vh * (kv_cap / BN) * HD;
    Out   += (size_t)head * n_q * HD;
    const int q0  = blockIdx.x * BM;
    if (q0 >= n_q) return;

    const int rg = tid / CG, tx = tid % CG;     // QK 映射
    const int i0 = rg * TM,  j0 = tx * TN;
    const int ro = tid / CGO, co = tid % CGO;   // PV 映射
    const int io0 = ro * TMO, do0 = co * TNO;

    // ---- Q 块常驻 + 行状态初始化（一次） ----
    for (int t = tid; t < BM * KD; t += NT) {
        const int i = t / KD, kd = t % KD;
        QsT[kd][i] = Qg[(size_t)(q0 + i) * KD + kd];
    }
    for (int t = tid; t < BM * NQG; t += NT) {
        const int i = t / NQG, g = t % NQG;
        QscT[g][i] = Qscg[(size_t)g * n_q + q0 + i];
    }
    for (int i = tid; i < BM; i += NT) {
        RowRef[i] = -1e30f; RowL[i] = 0.f; RowAlpha[i] = 1.f; RowPB[i] = 0.f;
    }

    // ---- 预取寄存器：K 每 pass 每线程 1 个 uint4（8 线程覆盖一行 32 dword），
    //      V 每 pass 每线程 2 个 uint4（32 线程覆盖一行 HD 个 dword）----
    constexpr int KPP  = NT / 8;                 // K：每 pass 覆盖多少个 key
    constexpr int KPASS = BN / KPP;
    constexpr int VPASS = KDT * 32 / NT;
    static_assert(BN % KPP == 0 && (KDT * 32) % NT == 0, "staging 分块不整除");
    u32 kreg[KPASS][4], vreg[VPASS][8];
    const int kj  = tid >> 3;                   // 0..KPP-1：本 pass 第几个 key
    const int kd0 = (tid & 7) * 4;              // 每行 4 个 dword
    const int vr  = tid >> 5;                   // 本 pass 第几个 8-key 组（一行 32 线程）
    const int vd0 = (tid & 31) * 8;             // 0..HD-8：每线程连续 8 个 d
    auto ldg_tile = [&](int t) {
        const int kb = t * BN;
#pragma unroll
        for (int p = 0; p < KPASS; p++) {
            const int j = p * KPP + kj;
            if (kb + j < n_kv) {
                *(uint4*)&kreg[p][0] = *(const uint4*)(Kg + (size_t)(kb + j) * KD + kd0);
            } else {
                kreg[p][0] = kreg[p][1] = kreg[p][2] = kreg[p][3] = 0u;
            }
        }
#pragma unroll
        for (int p = 0; p < VPASS; p++) {
            const int k8 = kb / 8 + p * (NT / 32) + vr;   // 8 个 key 一组的行号
            if ((size_t)k8 * 8 + 8 <= (size_t)n_kv) {
                const u32* s = Vc + (size_t)k8 * HD + vd0;
                *(uint4*)&vreg[p][0] = *(const uint4*)s;
                *(uint4*)&vreg[p][4] = *(const uint4*)(s + 4);
            } else {
                for (int e = 0; e < 8; e++) vreg[p][e] = 0u;
            }
        }
    };
    auto sts_k = [&]() {                       // 共享内存里的 K 是 tile 局部的
#pragma unroll
        for (int p = 0; p < KPASS; p++) {
            const int j = p * KPP + kj;
            const int c = xcol(j, BN);
#pragma unroll
            for (int e = 0; e < 4; e++) KsT[kd0 + e][c] = kreg[p][e];
        }
    };
    auto sts_v = [&]() {
#pragma unroll
        for (int p = 0; p < VPASS; p++) {
            const int row = p * (NT / 32) + vr;
            const int c0 = (vd0 / 8) * 4, c1 = HD / 2 + (vd0 / 8) * 4;
            *(uint4*)&Vs[row][c0] = *(uint4*)&vreg[p][0];
            *(uint4*)&Vs[row][c1] = *(uint4*)&vreg[p][4];
        }
    };

    ldg_tile(0);
    sts_k(); sts_v();
    if (tid < NQG * BN) {                       // K 的尺度：组优先布局
        const int g = tid / BN, j = tid % BN;
        KscT[g][j] = Kscg[(size_t)g * kv_cap + j];
    }
    if (tid < HD) VscT[tid] = Vscg[tid];        // tile 0 的 V 分组尺度
    __syncthreads();

    int   iacc[TM][TN];
    float sacc[TM][TN];
    float oacc[TMO][TNO];
    int   oiacc[TMO][TNO];
#pragma unroll
    for (int i = 0; i < TMO; i++)
#pragma unroll
        for (int j = 0; j < TNO; j++) { oacc[i][j] = 0.f; oiacc[i][j] = 0; }

    for (int t = 0; t < n_tile; t++) {
        const int kb = t * BN;

        // ---------- 预取下一个 tile 的 K/V（延迟被下面的 QK 盖住） ----------
        if (t + 1 < n_tile) ldg_tile(t + 1);

        // ---------- QK^T ----------
#ifdef ABL_PV_ONLY                                  // 消融：跳过 QK，用假 S 喂后面
#pragma unroll
        for (int i = 0; i < TM; i++)
#pragma unroll
            for (int j = 0; j < TN; j++)
                sacc[i][j] = (float)((tid + i * 7 + j * 13) & 7) * 0.1f;
        (void)iacc;
#else
#pragma unroll
        for (int i = 0; i < TM; i++)
#pragma unroll
            for (int j = 0; j < TN; j++) { iacc[i][j] = 0; sacc[i][j] = 0.f; }

#pragma unroll
        for (int g = 0; g < NQG; g++) {
            float sq[TM], sk[TN];
#pragma unroll
            for (int i = 0; i < TM; i++) sq[i] = QscT[g][i0 + i];
#pragma unroll
            for (int j = 0; j < TN; j++) sk[j] = KscT[g][j0 + j];
            // 共享内存取数软件流水：2 级展开 + 两组交替寄存器，避免「a=an」这种搬移
            // （搬移不清掉的话能占掉 30% 的指令槽）；本 kd 的 dot8 盖住下一 kd 的 LDS 延迟。
            {
                u32 a0[TM], b0[TN], a1[TM], b1[TN];
                auto ldq = [&](int kd, u32* aa, u32* bb) {
#ifdef ABL_NOLDS_QK                        // 消融：QK 的源操作数不来自共享内存
                    (void)kd;
#pragma unroll
                    for (int i = 0; i < TM; i++) aa[i] = 0x11111111u + (u32)i;
#pragma unroll
                    for (int j = 0; j < TN; j++) bb[j] = 0x22222222u + (u32)j;
#else
                    if (TM % 4 == 0) {
#pragma unroll
                        for (int e = 0; e < TM; e += 4)
                            *(uint4*)&aa[e] = *(const uint4*)&QsT[kd][i0 + e];
                    } else {
#pragma unroll
                        for (int e = 0; e + 1 < TM; e += 2)
                            *(uint2*)&aa[e] = *(const uint2*)&QsT[kd][i0 + e];
#pragma unroll
                        for (int e = (TM / 2) * 2; e < TM; e++) aa[e] = QsT[kd][i0 + e];
                    }
                    const uint4 c0 = *(const uint4*)&KsT[kd][4 * tx];
                    const uint4 c1 = *(const uint4*)&KsT[kd][BN / 2 + 4 * tx];   // 交错后的后半行
                    bb[0] = c0.x; bb[1] = c0.y; bb[2] = c0.z; bb[3] = c0.w;
                    bb[4] = c1.x; bb[5] = c1.y; bb[6] = c1.z; bb[7] = c1.w;
#endif
                };
                constexpr int NQ8 = QG / 8;
                ldq(g * NQ8, a0, b0);
#pragma unroll
                for (int kd2 = 0; kd2 < NQ8; kd2 += 2) {
                    if (kd2 + 1 < NQ8) ldq(g * NQ8 + kd2 + 1, a1, b1);
#pragma unroll
                    for (int i = 0; i < TM; i++)
#pragma unroll
                        for (int j = 0; j < TN; j++)
                            iacc[i][j] = dot8(iacc[i][j], a0[i], b0[j]);
                    if (kd2 + 2 < NQ8) ldq(g * NQ8 + kd2 + 2, a0, b0);
#pragma unroll
                    for (int i = 0; i < TM; i++)
#pragma unroll
                        for (int j = 0; j < TN; j++)
                            iacc[i][j] = dot8(iacc[i][j], a1[i], b1[j]);
                }
            }
#pragma unroll
            for (int i = 0; i < TM; i++)
#pragma unroll
                for (int j = 0; j < TN; j++) {
                    sacc[i][j] += (float)iacc[i][j] * (sq[i] * sk[j]);
                    iacc[i][j] = 0;
                }
        }
#endif

        // ---------- 因果掩码 ----------
        // query 的绝对位置 = off + q0 + i0 + i；key 的绝对位置 = kb + j0 + j
#pragma unroll
        for (int i = 0; i < TM; i++) {
            const int maxk = off + q0 + i0 + i - kb;      // 该行在本 tile 里最大的合法 key 下标
#pragma unroll
            for (int j = 0; j < TN; j++)
                if (j0 + j > maxk) sacc[i][j] = -1e30f;
        }

        // ---------- 行最大 / P 量化 / 行和 ----------
        float mt[TM];
        int   iq[TM][TN];
#pragma unroll
        for (int i = 0; i < TM; i++) {
            float m = sacc[i][0];
#pragma unroll
            for (int j = 1; j < TN; j++) m = fmaxf(m, sacc[i][j]);
#pragma unroll
            for (int o = 1; o < CG; o <<= 1) m = fmaxf(m, __shfl_xor(m, o));
            mt[i] = m;
        }
        // 整行被掩掉时用 running 参考顶替：这样 alpha=1、beta=exp2(-1e30)=0，不污染 l
#pragma unroll
        for (int i = 0; i < TM; i++)
            if (mt[i] < -1e29f) mt[i] = RowRef[i0 + i];

        int isum[TM];
#pragma unroll
        for (int i = 0; i < TM; i++) {
            int s = 0;
#pragma unroll
            for (int j = 0; j < TN; j++) {
                const float p = exp2f((sacc[i][j] - mt[i]) * L2E) * 7.f;
                int q = (int)(p + 0.5f);
                q = q > 7 ? 7 : (q < 0 ? 0 : q);
                iq[i][j] = q;
                s += q;
            }
#pragma unroll
            for (int o = 1; o < CG; o <<= 1) s += __shfl_xor(s, o);
            isum[i] = s;
            u32 pk = 0;
#pragma unroll
            for (int j = 0; j < TN; j++) pk |= (u32)(iq[i][j] & 0xF) << (4 * j);
            PsT[tx][i0 + i] = pk;
        }

        if (tx == 0) {
#pragma unroll
            for (int i = 0; i < TM; i++) {
                const int row = i0 + i;
                const float r_old = RowRef[row];
                const float r_new = fmaxf(r_old, mt[i]);
                const float alpha = exp2f((r_old - r_new) * L2E);
                const float beta  = exp2f((mt[i] - r_new) * L2E);
                RowRef[row]   = r_new;
                RowAlpha[row] = alpha;
                RowPB[row]    = beta * (1.f / 7.f);
                RowL[row]     = RowL[row] * alpha + (float)isum[i] * (beta / 7.f);
            }
        }
        __syncthreads();                                  // BAR(a)

        // ---------- 下一 tile 的 K 落共享（QK(t) 全部读完，安全） ----------
        if (t + 1 < n_tile) {
            sts_k();
            if (tid < NQG * BN) {                     // K 的分组尺度跟着一起换
                const int g = tid / BN, j = tid % BN;
                KscT[g][j] = Kscg[(size_t)g * kv_cap + (t + 1) * BN + j];
            }
        }

        // ---------- PV：O = O·alpha + (P·V)·(beta/7)·sv ----------
#ifndef ABL_QK_ONLY
#pragma unroll
        for (int i = 0; i < TMO; i++)
#pragma unroll
            for (int j = 0; j < TNO; j++) oacc[i][j] *= RowAlpha[io0 + i];
#pragma unroll
        for (int i = 0; i < TMO; i++)
#pragma unroll
            for (int j = 0; j < TNO; j++) oiacc[i][j] = 0;

        float pb[TMO];
#pragma unroll
        for (int i = 0; i < TMO; i++) pb[i] = RowPB[io0 + i];
        u32 pa[TMO], pbv[TNO], pan[TMO], pbn[TNO];
        {
#pragma unroll
            for (int e = 0; e < TMO / 4; e++) {
                const uint4 q4 = *(const uint4*)&PsT[0][io0 + 4 * e];
                pa[4 * e + 0] = q4.x; pa[4 * e + 1] = q4.y;
                pa[4 * e + 2] = q4.z; pa[4 * e + 3] = q4.w;
            }
            const uint4 r0 = *(const uint4*)&Vs[0][4 * co];
            const uint4 r1 = *(const uint4*)&Vs[0][HD / 2 + 4 * co];
            pbv[0] = r0.x; pbv[1] = r0.y; pbv[2] = r0.z; pbv[3] = r0.w;
            pbv[4] = r1.x; pbv[5] = r1.y; pbv[6] = r1.z; pbv[7] = r1.w;
        }
#pragma unroll
        for (int kd = 0; kd < KDT; kd++) {
            if (kd + 1 < KDT) {
#pragma unroll
                for (int e = 0; e < TMO / 4; e++) {
                    const uint4 q4 = *(const uint4*)&PsT[kd + 1][io0 + 4 * e];
                    pan[4 * e + 0] = q4.x; pan[4 * e + 1] = q4.y;
                    pan[4 * e + 2] = q4.z; pan[4 * e + 3] = q4.w;
                }
                const uint4 r0 = *(const uint4*)&Vs[kd + 1][4 * co];
                const uint4 r1 = *(const uint4*)&Vs[kd + 1][HD / 2 + 4 * co];
                pbn[0] = r0.x; pbn[1] = r0.y; pbn[2] = r0.z; pbn[3] = r0.w;
                pbn[4] = r1.x; pbn[5] = r1.y; pbn[6] = r1.z; pbn[7] = r1.w;
            }
#pragma unroll
            for (int i = 0; i < TMO; i++)
#pragma unroll
                for (int j = 0; j < TNO; j++)
                    oiacc[i][j] = dot8(oiacc[i][j], pa[i], pbv[j]);
#pragma unroll
            for (int i = 0; i < TMO; i++) pa[i] = pan[i];
#pragma unroll
            for (int j = 0; j < TNO; j++) pbv[j] = pbn[j];
        }
#pragma unroll
        for (int i = 0; i < TMO; i++)
#pragma unroll
            for (int j = 0; j < TNO; j++)
                oacc[i][j] = fmaf((float)oiacc[i][j], pb[i] * VscT[do0 + j], oacc[i][j]);
#else
        (void)io0; (void)do0; (void)co;
#endif

        __syncthreads();                                  // BAR(b)

        // ---------- 下一 tile 的 V 落共享 + 它的分组尺度 ----------
        if (t + 1 < n_tile) {
            sts_v();
            if (tid < HD) VscT[tid] = Vscg[(size_t)(t + 1) * HD + tid];
        }
    }

    // ---- 写回：O / l ----
#pragma unroll
    for (int i = 0; i < TMO; i++) {
        const int row = io0 + i;
        if (q0 + row >= n_q) continue;
        const float inv = RowL[row] > 0.f ? 1.f / RowL[row] : 0.f;
#pragma unroll
        for (int j = 0; j < TNO; j++)
            Out[(size_t)(q0 + row) * HD + do0 + j] = oacc[i][j] * inv;
    }
}
