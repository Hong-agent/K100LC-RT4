// 注意力封装：直接复用现役 int4 FlashAttention 内核（单一来源，见 kernels/flash_attn_core.h）
#include "kernels.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include "../kernels/flash_attn_core.h"

// ===================== 解码注意力：单行 query 专用 ==========================
// 上面的 fa_int4 是给预填充写的（BM=64 行 query 一起算）。解码时只有 1 行真实
// query，却要按 64 行补齐 —— 白算 64 倍，而且上下文越长亏得越多（实测 4k 上下
// 文时解码从 38.6 → 48.3 ms/token，且线性恶化）。
//
// 这里一个 block 一个 head、256 个线程 = head_dim 一个线程：
//   QK：thread t 负责 key = t/4、dword 段 = (t%4)*8，4 个 lane 归约出一个 key 的
//       score。4 个 lane 正好覆盖两个 128 维量化组（quarter 0/1 → 组 0，
//       2/3 → 组 1），各自的 scale 在归约前乘上；
//   PV：thread t 负责输出维度 t，按 [key/8][D] 布局读 V（同一时刻 32 个 lane
//       读连续地址）。P 直接用 fp32，比预填充路径的 4bit P 还准。
// KV 布局与 fa_int4 完全一致（见 kernels/flash_attn_core.h 顶部）。
// split-KV：把 key 方向切成 n_split 段，每段一个 block（grid.y），各自算
// (局部 max, 局部 sum, 未归一化的输出)；再由 fa_decode_comb_k 合并。
// 只要 24 个头 × 4~8 段 = 96~192 个 block 才能填满 120 个 CU —— 单 block 串行
// 遍历整段 KV 是延迟受限的（实测 4k 上下文每 1000 key 要 1.24 ms，而带宽只要
// 0.02 ms）。
// 内核体：一个 block 负责 (head, split) 的一段 KV，线程 = head_dim。
// 抽成 __device__ 函数，让「单行解码」与「MTP 验证批多行」两个入口共用同一份算术
// —— 这是验证批与单 token 解码**逐位相同**的前提（见 docs/MTP.md 的等价性坑 1）。
// sc_stride 是 split-KV 暂存的段距（调用方两侧必须一致）。
__device__ __forceinline__ void fa_decode_body(
        float* __restrict__ pout, float* __restrict__ pmax, float* __restrict__ psum,
        const u32* __restrict__ Qg, const float* __restrict__ Qscg,
        const u32* __restrict__ Kg, const float* __restrict__ Kscg, const u32* __restrict__ Vc,
        const float* __restrict__ Vscg, const float* __restrict__ Vsnap, int n_kv, int kv_cap,
        int ratio, int q_dw_stride, int q_sc_stride, int q_sc_row, int n_split, int sc_stride,
        int head, int split) {
    static_assert(HD == 256 && KD == 32 && NQG == 2, "解码内核按 D=256 手工展开");
    const int t = threadIdx.x;                     // 0..255
    const int vh = head / ratio;
    const int kk = t >> 2;                         // 本 tile 内负责的 key（0..63）
    const int quarter = t & 3;
    const int g = quarter >> 1;                    // 128 维量化组

    const u32* Kgb = Kg + (size_t)vh * kv_cap * KD;
    const float* Kscb = Kscg + (size_t)vh * NQG * kv_cap;
    const u32* Vcb = Vc + (size_t)vh * (kv_cap / 8) * HD;
    const float* Vscb = Vscg + (size_t)vh * (kv_cap / BN) * HD;

    __shared__ u32 qsh[KD];
    __shared__ float qsc_sh[NQG];
    __shared__ float shm[8], ssh[8];
    __shared__ float psh[BN];
    if (t < KD) qsh[t] = Qg[(size_t)head * q_dw_stride + t];
    if (t < NQG) qsc_sh[t] = Qscg[(size_t)head * q_sc_stride + (size_t)t * q_sc_row];

    float o = 0.f, mrun = -1e30f, lrun = 0.f;
    const int ntile = (n_kv + BN - 1) / BN;
    // 本行 prefix 落在最后一格（tile）里：V 的 tile 尺度按 prefix 的 amax 定，
    // 所以「同一个 tile 在不同行眼里」量化结果不同（见下面 PV 里的重算）。
    const int row_tile = (n_kv - 1) / BN;
    const int nval = n_kv - row_tile * BN;
    const int tps = (ntile + n_split - 1) / n_split;      // 每段几个 tile
    const int t0 = split * tps, t1 = min(ntile, t0 + tps);
    for (int tile = t0; tile < t1; tile++) {
        const int kv0 = tile * BN;
        __syncthreads();                            // 上一轮的 shm/ssh/psh 已用完
        const int kvi = kv0 + kk;
        float s;
        if (kvi < n_kv) {                           // 本 tile 的 score
            int acc = 0;
            const u32* kr = Kgb + (size_t)kvi * KD + quarter * 8;
#pragma unroll
            for (int i = 0; i < 8; i++) acc = dot8(acc, qsh[quarter * 8 + i], kr[i]);
            s = (float)acc * qsc_sh[g] * Kscb[g * kv_cap + kvi];
        } else {
            s = -1e30f;
        }
#pragma unroll
        for (int off = 1; off <= 2; off <<= 1) s += __shfl_xor(s, off);
        float mx = s;                               // tile max（先 8 个 key，再跨 warp）
#pragma unroll
        for (int off = 4; off <= 16; off <<= 1) mx = fmaxf(mx, __shfl_xor(mx, off));
        if ((t & 31) == 0) shm[t >> 5] = mx;
        __syncthreads();
        float tmax = shm[0];
#pragma unroll
        for (int i = 1; i < 8; i++) tmax = fmaxf(tmax, shm[i]);
        if (tmax < -1e29f) tmax = mrun;             // 整块无效（尾块）：不污染
        const float mnew = fmaxf(mrun, tmax);
        const float alpha = exp2f((mrun - mnew) * L2E);
        // online：p 用更新后的 max，旧的输出按 alpha 缩放
        const float p = exp2f((s - mnew) * L2E);
        float psum = p;
#pragma unroll
        for (int off = 4; off <= 16; off <<= 1) psum += __shfl_xor(psum, off);
        if ((t & 31) == 0) ssh[t >> 5] = psum;
        __syncthreads();
        float tsum = 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++) tsum += ssh[i];
        lrun = lrun * alpha + tsum;
        o *= alpha;
        mrun = mnew;
        if (quarter == 0) psh[kk] = p;
        __syncthreads();
        if (t < HD) {                               // PV：维度 t
            u32 vsnap_buf[BN / 8];
            const bool use_snap = Vsnap != nullptr && tile == row_tile;
            const u32* vr = Vcb + (size_t)(kv0 / 8) * HD + t;
            float vsc_here = Vscb[(size_t)tile * HD + t];
            if (use_snap) {
                // Vsnap 是「本行 append 之后的 fp32 V tile 暂存」：按本行 prefix 重新
                // 量化一次，得到与单行内核逐位相同的打包值（append 当时写的就是这些）。
                // 暂存布局 [row][KV][BN][D]，与运行时的 V tile（BG=64）一致
                static_assert(BN == 64, "V tile 粒度必须与运行时 BG 相同");
                const float* st = Vsnap + (size_t)vh * BN * HD + t;
                float amax = 0.f;
                for (int i = 0; i < nval; i++) amax = fmaxf(amax, fabsf(st[(size_t)i * HD]));
                vsc_here = amax > 0.f ? amax / 7.f : 1.f;
                for (int i = 0; i < BN; i += 8) {
                    u32 w = 0;
#pragma unroll
                    for (int e = 0; e < 8; e++) {
                        const int key = i + e;
                        const float v = key < nval ? st[(size_t)key * HD] : 0.f;
                        int c = (int)lrintf(v / vsc_here);
                        c = c > 7 ? 7 : (c < -8 ? -8 : c);
                        w |= (u32)(c & 0xF) << (4 * e);
                    }
                    vsnap_buf[i / 8] = w;
                }
            }
            float a = 0.f;
#pragma unroll
            for (int j = 0; j < BN / 8; j++) {
                const u32 w = use_snap ? vsnap_buf[j] : vr[(size_t)j * HD];
                const float* pp = psh + j * 8;
#pragma unroll
                for (int e = 0; e < 8; e++) {
                    int c = (int)((w >> (4 * e)) & 0xF);
                    if (c >= 8) c -= 16;
                    a = fmaf(pp[e], (float)c, a);
                }
            }
            o = fmaf(vsc_here, a, o);
        }
    }
    if (t < HD) pout[((size_t)head * sc_stride + split) * HD + t] = o;
    if (t == 0) {
        pmax[(size_t)head * sc_stride + split] = mrun;
        psum[(size_t)head * sc_stride + split] = lrun;
    }
}

__global__ void fa_decode_k(float* __restrict__ pout, float* __restrict__ pmax,
                            float* __restrict__ psum, const u32* __restrict__ Qg,
                            const float* __restrict__ Qscg, const u32* __restrict__ Kg,
                            const float* __restrict__ Kscg, const u32* __restrict__ Vc,
                            const float* __restrict__ Vscg, int n_kv, int kv_cap, int ratio,
                            int q_dw_stride, int q_sc_stride, int q_sc_row, int n_split) {
    fa_decode_body(pout, pmax, psum, Qg, Qscg, Kg, Kscg, Vc, Vscg, nullptr, n_kv, kv_cap,
                   ratio, q_dw_stride, q_sc_stride, q_sc_row, n_split, n_split, blockIdx.x,
                   blockIdx.y);
}

// ===================== 验证批解码注意力：一次算 T 行 query ==================
// MTP 验证批是 T≤4 行、每行 KV 长度差 1 的因果解码。以前是「逐行调用单行内核」，
// 于是整段 KV 被读 T 遍，而且 T 次启动在长上下文下是串行延迟（实测 8k 上下文时
// 4 行合计 8.9 ms/轮，占解码 11.7%）。
// 这里把**行**放进 grid.z：每行仍由同一份 fa_decode_body 计算、同一套 split 规则，
// 因此逐行结果与单行内核按位相同；差别只是 T 行并行而不是串行。
// 注意第 r 行只能看到 kr = n_kv0 + r 个 key（因果），靠 body 里的 kvi < n_kv 掩码。
__global__ void fa_decode_rows_k(float* __restrict__ pout, float* __restrict__ pmax,
                                 float* __restrict__ psum, const u32* __restrict__ Qg,
                                 const float* __restrict__ Qscg, const u32* __restrict__ Kg,
                                 const float* __restrict__ Kscg, const u32* __restrict__ Vc,
                                 const float* __restrict__ Vscg,
                                 const float* __restrict__ Vsnap, size_t vsnap_row_stride,
                                 int n_kv0, int kv_cap, int ratio, int q_dw_stride,
                                 int q_sc_stride, int q_sc_row, int max_split, int force_split,
                                 int sc_stride, int n_heads, size_t q_row_stride,
                                 size_t sc_row_stride) {
    const int row = blockIdx.z;
    const int n_kv = n_kv0 + row;
    const int ntile = (n_kv + BN - 1) / BN;
    int n_split = force_split > 0 ? force_split : 1;
    while (force_split <= 0 && n_split < max_split && ntile >= n_split * 8)
        n_split *= 2;                                                    // 与主机侧同规则
    if (blockIdx.y >= n_split) return;                                   // 行尾多出来的段空转
    // 暂存按行切片：每行是 n_heads 个头的完整 split-KV 暂存（少乘 n_heads 就会让
    // 相邻两行互相踩，实测表现为「一次算 4 行」结果错乱且不确定）。
    const size_t pst = (size_t)row * n_heads * sc_stride * HD;
    const size_t sst = (size_t)row * n_heads * sc_stride;
    fa_decode_body(pout + pst, pmax + sst, psum + sst,
                   Qg + (size_t)row * q_row_stride, Qscg + (size_t)row * sc_row_stride, Kg,
                   Kscg, Vc, Vscg, Vsnap ? Vsnap + (size_t)row * vsnap_row_stride : nullptr,
                   n_kv, kv_cap, ratio, q_dw_stride, q_sc_stride, q_sc_row, n_split, sc_stride,
                   blockIdx.x, blockIdx.y);
}

// 合并各 split：(o, l) 按各自的 max 重标定后相加
__global__ void fa_decode_comb_k(float* __restrict__ Out, const float* __restrict__ pout,
                                 const float* __restrict__ pmax, const float* __restrict__ psum,
                                 int n_split, int out_stride) {
    const int head = blockIdx.x;
    const int t = threadIdx.x;
    float M = -1e30f;
    for (int s = 0; s < n_split; s++) M = fmaxf(M, pmax[(size_t)head * n_split + s]);
    float o = 0.f, l = 0.f;
    for (int s = 0; s < n_split; s++) {
        const float a = exp2f((pmax[(size_t)head * n_split + s] - M) * L2E);
        o += a * pout[((size_t)head * n_split + s) * HD + t];
        l += a * psum[(size_t)head * n_split + s];
    }
    if (l > 0.f) Out[(size_t)head * out_stride + t] = o / l;
}

// 同上，但按行（blockIdx.y）取各自的暂存、按行自己的有效段数 n_split 合并。
__global__ void fa_decode_rows_comb_k(float* __restrict__ Out, const float* __restrict__ pout,
                                      const float* __restrict__ pmax,
                                      const float* __restrict__ psum, int n_kv0, int n_heads,
                                      int max_split, int force_split, int sc_stride,
                                      int out_stride, size_t out_row_stride) {
    const int head = blockIdx.x;
    const int row = blockIdx.y;
    const int t = threadIdx.x;
    const int n_kv = n_kv0 + row;
    const int ntile = (n_kv + BN - 1) / BN;
    int n_split = force_split > 0 ? force_split : 1;
    while (force_split <= 0 && n_split < max_split && ntile >= n_split * 8) n_split *= 2;
    const float* po = pout + (size_t)row * n_heads * sc_stride * HD;
    const float* pm = pmax + (size_t)row * n_heads * sc_stride;
    const float* ps = psum + (size_t)row * n_heads * sc_stride;
    float M = -1e30f;
    for (int s = 0; s < n_split; s++) M = fmaxf(M, pm[(size_t)head * sc_stride + s]);
    float o = 0.f, l = 0.f;
    for (int s = 0; s < n_split; s++) {
        const float a = exp2f((pm[(size_t)head * sc_stride + s] - M) * L2E);
        o += a * po[((size_t)head * sc_stride + s) * HD + t];
        l += a * ps[(size_t)head * sc_stride + s];
    }
    if (l > 0.f) Out[(size_t)row * out_row_stride + (size_t)head * out_stride + t] = o / l;
}

void k_attention_decode(float* out, const u32* qq, const float* qsc,
                        const u32* kc, const float* ksc, const u32* vc, const float* vsc,
                        int n_kv, int kv_cap, int n_heads, int ratio, int n_q,
                        float* pout, float* pmax, float* psum, int max_split) {
    const int ntile = (n_kv + BN - 1) / BN;
    int n_split = 1;
    while (n_split < max_split && ntile >= n_split * 8) n_split *= 2;
    static const int forced = getenv("RT_NSPLIT") ? atoi(getenv("RT_NSPLIT")) : 0;
    if (forced > 0) n_split = forced > max_split ? max_split : forced;   // 缓冲按 max_split 分配
    fa_decode_k<<<dim3(n_heads, n_split), HD>>>(pout, pmax, psum, qq, qsc, kc, ksc, vc, vsc,
                                                n_kv, kv_cap, ratio, n_q * KD, n_q * NQG, n_q,
                                                n_split);
    if (hipGetLastError() != hipSuccess) { printf("decode split launch fail\n"); exit(1); }
    fa_decode_comb_k<<<n_heads, HD>>>(out, pout, pmax, psum, n_split, n_q * HD);
    if (hipGetLastError() != hipSuccess) { printf("decode comb launch fail\n"); exit(1); }
}

// T 行（T≤4）验证批：n_kv0 是第 0 行的 KV 长度，第 r 行看到 n_kv0+r 个 key。
// 暂存按行分片（每行 H*max_split*HD 与 H*max_split 个 float），需要调用方按 4 行分配。
void k_attention_decode_rows(float* out, const u32* qq, const float* qsc,
                             const u32* kc, const float* ksc, const u32* vc, const float* vsc,
                             const float* vstage_snap, size_t vsnap_row_stride, int n_q,
                             int n_rows, int n_kv0, int n_heads, int ratio, int kv_cap,
                             float* pout, float* pmax, float* psum, int max_split) {
    static const int forced = getenv("RT_NSPLIT") ? atoi(getenv("RT_NSPLIT")) : 0;
    const int split = forced > 0 ? (forced > max_split ? max_split : forced) : max_split;
    const int q_dw_stride = n_q * KD, q_sc_stride = n_q * NQG, q_sc_row = n_q;
    fa_decode_rows_k<<<dim3(n_heads, split, n_rows), HD>>>(
        pout, pmax, psum, qq, qsc, kc, ksc, vc, vsc, vstage_snap, vsnap_row_stride, n_kv0,
        kv_cap, ratio, q_dw_stride, q_sc_stride, q_sc_row, max_split, forced, max_split,
        n_heads, KD, 1);
    if (hipGetLastError() != hipSuccess) { printf("decode rows launch fail\n"); exit(1); }
    fa_decode_rows_comb_k<<<dim3(n_heads, n_rows), HD>>>(out, pout, pmax, psum, n_kv0,
                                                         n_heads, max_split, forced, max_split,
                                                         n_q * HD, HD);
    if (hipGetLastError() != hipSuccess) { printf("decode rows comb fail\n"); exit(1); }
}

void k_attention(float* out, const u32* qq, const float* qsc,
                 const u32* kc, const float* ksc, const u32* vc, const float* vsc,
                 int n_q, int n_row, int n_kv, int off, int n_tile, int n_heads, int ratio,
                 int kv_cap, float* pout, float* pmax, float* psum, int max_split) {
    // n_row == 1 且没有待算的因果未来（off+1 == n_kv）→ 解码单行路径
    static const int old = getenv("RT_ATTN_OLD") ? atoi(getenv("RT_ATTN_OLD")) : 0;
    if (n_row == 1 && off + 1 == n_kv && !old) {
        k_attention_decode(out, qq, qsc, kc, ksc, vc, vsc, n_kv, kv_cap, n_heads, ratio, n_q,
                           pout, pmax, psum, max_split);
        return;
    }
    const int nblk = (n_q + BM - 1) / BM;
    fa_int4<<<dim3(nblk, n_heads), NT>>>(qq, qsc, kc, ksc, vc, vsc, out,
                                         n_q, n_kv, off, n_tile, ratio, kv_cap);
    if (hipGetLastError() != hipSuccess) {
        printf("attention launch fail: %s\n", hipGetErrorString(hipGetLastError()));
        exit(1);
    }
}
