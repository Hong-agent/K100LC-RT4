// 自研运行时的算子实现（元素级 / 量化 / 词嵌入 / KV 写入 / gated delta net / 卷积）。
//
// 设计取舍（这一轮的目标是「先能对话」，性能第二步）：
//   * 元素级算子一律 grid-stride，简单够用；
//   * 量化与打包布局严格对齐现役内核的期望（见 kernels/*_core.h 顶部注释）；
//   * gated delta net 递推用「一线程一列」：每个 v head 的 D×D 状态按列切开，
//     每列 128 个状态元素常驻寄存器，列与列完全独立 → 不需要任何跨线程归约，
//     block = 64 线程正好一个 wave。
#include "kernels.h"
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdlib>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    printf("HIP ERR %s @%d: %s\n", #x, __LINE__, hipGetErrorString(e_)); exit(1);} } while (0)

#define GDN_MAX_D 128

// ================================ 基础 =====================================
__global__ void fill_k(float* y, float v, long long n) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x) y[i] = v;
}
__global__ void add_inplace_k(float* y, const float* x, long long n) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x) y[i] += x[i];
}
// 两段 [rows][D] 按最后一维拼成 [rows][2D]（MTP 的 [embedding | hidden]）
__global__ void concat2_k(float* __restrict__ y, const float* __restrict__ a,
                          const float* __restrict__ b, int D, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const long long row = i / D, d = i % D;
    y[row * 2 * D + d] = a[i];
    y[row * 2 * D + D + d] = b[i];
}
__global__ void silu_mul_k(float* y, const float* g, const float* u, long long n) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x) {
        const float x = g[i];
        y[i] = (x / (1.f + __expf(-x))) * u[i];
    }
}
__global__ void sigmoid_mul_k(float* y, const float* x, const float* g, long long n) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x)
        y[i] = x[i] / (1.f + __expf(-g[i]));
}
static int grid_for(long long n, int th) {
    long long bl = (n + th - 1) / th;
    return (int)(bl > 65535 ? 65535 : bl);
}
void k_fill(float* y, float v, long long n) {
    fill_k<<<grid_for(n, 256), 256>>>(y, v, n); CK(hipGetLastError());
}
void k_add_inplace(float* y, const float* x, long long n) {
    add_inplace_k<<<grid_for(n, 256), 256>>>(y, x, n); CK(hipGetLastError());
}
void k_concat2(float* y, const float* a, const float* b, int rows, int D) {
    const long long n = (long long)rows * D;
    concat2_k<<<grid_for(n, 256), 256>>>(y, a, b, D, n); CK(hipGetLastError());
}
void k_silu_mul(float* y, const float* g, const float* u, long long n) {
    silu_mul_k<<<grid_for(n, 256), 256>>>(y, g, u, n); CK(hipGetLastError());
}
void k_sigmoid_mul(float* y, const float* x, const float* g, long long n) {
    sigmoid_mul_k<<<grid_for(n, 256), 256>>>(y, x, g, n); CK(hipGetLastError());
}

// =============================== RMSNorm ===================================
// Qwen3.5 的普通 RMSNorm 是「零中心」权重：y = x̂ * (1 + w)
__global__ void rmsnorm_k(float* y, const float* x, const float* w, int D, float eps, int zc) {
    const int row = blockIdx.x;
    const float* xr = x + (size_t)row * D;
    float* yr = y + (size_t)row * D;
    float s = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) s += xr[i] * xr[i];
    __shared__ float sh[32];
    for (int o = 16; o; o >>= 1) s += __shfl_xor(s, o);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x < 32) {
        float t = (threadIdx.x < (int)(blockDim.x >> 5)) ? sh[threadIdx.x] : 0.f;
        for (int o = 16; o; o >>= 1) t += __shfl_xor(t, o);
        if (threadIdx.x == 0) sh[0] = rsqrtf(t / (float)D + eps);
    }
    __syncthreads();
    const float inv = sh[0];
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        yr[i] = xr[i] * inv * (zc ? (1.f + w[i]) : w[i]);
}
void k_rmsnorm(float* y, const float* x, const float* w, int rows, int D, float eps, bool zc) {
    rmsnorm_k<<<rows, D >= 256 ? 256 : 64>>>(y, x, w, D, eps, zc ? 1 : 0); CK(hipGetLastError());
}

// ============================== L2 norm ====================================
__global__ void l2norm_k(float* x, int S, float eps) {
    const size_t row = blockIdx.x;
    float* xr = x + row * S;
    float s = 0.f;
    for (int i = threadIdx.x; i < S; i += blockDim.x) s += xr[i] * xr[i];
    __shared__ float sh[32];
    for (int o = 16; o; o >>= 1) s += __shfl_xor(s, o);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x < 32) {
        float t = (threadIdx.x < (int)(blockDim.x >> 5)) ? sh[threadIdx.x] : 0.f;
        for (int o = 16; o; o >>= 1) t += __shfl_xor(t, o);
        if (threadIdx.x == 0) sh[0] = rsqrtf(t + eps);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < S; i += blockDim.x) xr[i] *= sh[0];
}
void k_l2norm(float* x, int rows, int S, float eps) {
    l2norm_k<<<rows, S >= 128 ? 128 : 64>>>(x, S, eps); CK(hipGetLastError());
}

// ============================== argmax =====================================
// 单 block、1024 线程做两阶段归约。vocab≈248k 时每线程约 243 次合并读，
// 实测远低于一次 1MB 的 D2H 拷贝。
__global__ void argmax_k(const float* __restrict__ x, long long n, int* __restrict__ out) {
    __shared__ float sv[32];
    __shared__ int si[32];
    float best = -INFINITY;
    int bi = -1;
    for (long long i = (long long)threadIdx.x; i < n; i += blockDim.x) {
        const float v = x[i];
        if (v > best || (v == best && (int)i < bi)) { best = v; bi = (int)i; }
    }
    for (int o = 16; o; o >>= 1) {
        const float ov = __shfl_xor(best, o);
        const int oi = __shfl_xor(bi, o);
        if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
    }
    if ((threadIdx.x & 31) == 0) { sv[threadIdx.x >> 5] = best; si[threadIdx.x >> 5] = bi; }
    __syncthreads();
    if (threadIdx.x < 32) {
        const int nw = blockDim.x >> 5;
        best = threadIdx.x < nw ? sv[threadIdx.x] : -INFINITY;
        bi = threadIdx.x < nw ? si[threadIdx.x] : -1;
        for (int o = 16; o; o >>= 1) {
            const float ov = __shfl_xor(best, o);
            const int oi = __shfl_xor(bi, o);
            if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
        }
        if (threadIdx.x == 0) *out = bi;
    }
}
void k_argmax(const float* x, long long n, int* out) {
    // gfx926 的默认每 block 线程上限是 256；248k 元素下每线程约 970 次循环，
    // 单 block 仍是亚毫秒级，不值得为 1024 线程另做多 block 归约。
    argmax_k<<<1, 256>>>(x, n, out);
    CK(hipGetLastError());
}

// ================================ RoPE =====================================
// 部分旋转（前 rot 维，rotate_half 约定）：
//   out[i]      = x[i]*cos - x[i+rot/2]*sin
//   out[i+rot/2]= x[i+rot/2]*cos + x[i]*sin ,  i < rot/2
// 频率与 HF 一致：1/theta^(2i/rot)
// pos 为空时位置取 pos0 + row（解码就是这一路：省掉每层一次的 hipMalloc +
// 阻塞式 H2D 拷贝 —— 那两次调用会把整条流水线抽干，实测每层 ~100 us）。
__global__ void rope_k(float* q, float* k, const int* pos, int pos0, int n_q, int n_kv,
                       int nqh, int nkvh, int D, int rot, float theta) {
    const int half = rot >> 1;
    const long long total = (long long)(n_q * nqh + n_kv * nkvh) * half;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (long long)gridDim.x * blockDim.x) {
        const int hh = (int)(idx % half);
        long long rest = idx / half;
        int row, head, isq;
        if (rest < (long long)n_q * nqh) { row = (int)(rest / nqh); head = (int)(rest % nqh); isq = 1; }
        else { rest -= (long long)n_q * nqh; row = (int)(rest / nkvh); head = (int)(rest % nkvh); isq = 0; }
        float* x = isq ? q + ((size_t)row * nqh + head) * D : k + ((size_t)row * nkvh + head) * D;
        const int p = pos ? pos[row] : pos0 + row;
        const float a = (float)p / powf(theta, (float)(2 * hh) / (float)rot);
        const float c = cosf(a), s = sinf(a);
        const float x1 = x[hh], x2 = x[hh + half];
        x[hh] = x1 * c - x2 * s;
        x[hh + half] = x2 * c + x1 * s;
    }
}
void k_rope(float* q, float* k, const int* pos, int pos0, int n_q, int n_kv, int nqh, int nkvh,
            int D, int rot, float theta) {
    const long long total = (long long)(n_q * nqh + n_kv * nkvh) * (rot / 2);
    rope_k<<<grid_for(total, 256), 256>>>(q, k, pos, pos0, n_q, n_kv, nqh, nkvh, D, rot, theta);
    CK(hipGetLastError());
}

// ========================= 激活量化（int4，权重式布局）======================
// f32 [rows][K] → q[row][K/8]（低半字节 = 偶数下标）+ 尺度 sc[g][row]（组优先）
__global__ void quant_rows_k(u32* q, float* sc, const float* x, int K, int G, int rows,
                             int in_stride, int in_off) {
    const int row = blockIdx.x;
    const int g = threadIdx.x;
    const float* xr = x + (size_t)row * in_stride + in_off + (size_t)g * G;
    float amax = 0.f;
#pragma unroll 4
    for (int i = 0; i < G; i++) amax = fmaxf(amax, fabsf(xr[i]));
    const float s = amax > 0.f ? amax / 7.f : 1.f;
    sc[(size_t)g * rows + row] = s;
    u32* qr = q + (size_t)row * (K / 8) + (size_t)(g * G) / 8;
    for (int i = 0; i < G; i += 8) {
        u32 w = 0;
#pragma unroll
        for (int e = 0; e < 8; e++) {
            int c = (int)lrintf(xr[i + e] / s);
            c = c > 7 ? 7 : (c < -8 ? -8 : c);
            w |= (u32)(c & 0xF) << (4 * e);
        }
        qr[i / 8] = w;
    }
}
void k_quant_rows(u32* q, float* sc, const float* x, int rows, int K, int G,
                  int in_stride, int in_off) {
    if (in_stride == 0) in_stride = K;
    quant_rows_k<<<rows, K / G>>>(q, sc, x, K, G, rows, in_stride, in_off);
    CK(hipGetLastError());
}

// ------------------- 注意力的 Q 量化：一次做完所有头 ------------------------
// 上面那个 k_quant_rows 一次只能做一个头（K=D），运行时要循环 H=24 次 —— 解码时
// 就是 24 次「64 block × 2 线程」的小启动，实测 0.72 ms/层（11.6 ms/token）。
// 这里换成单次启动：grid 铺满 H*T*(D/G) 个 (头, 行, 组) 三元组，每个线程自己
// 处理一整个 G=128 的组。输出布局与 FA 内核读的完全一致。
// 只写 row < T 的行；补齐到 TP 的那几行保持 reset_state 给的 0 数据 / 1 尺度
// （FA 是逐行独立的，补齐行的输出会被 k_scatter_heads 丢掉）。
__global__ void attn_q_quant_k(u32* qq, float* qs, const float* x, int T, int H, int D, int G,
                               int TP) {
    const int ng = D / G;
    const int total = H * T * ng;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += gridDim.x * blockDim.x) {
        const int g = i % ng;
        const int r = (i / ng) % T;
        const int h = i / (ng * T);
        const float* xr = x + ((size_t)r * H + h) * D + (size_t)g * G;
        float amax = 0.f;
#pragma unroll 4
        for (int j = 0; j < G; j++) amax = fmaxf(amax, fabsf(xr[j]));
        const float s = amax > 0.f ? amax / 7.f : 1.f;
        qs[((size_t)h * ng + g) * TP + r] = s;
        u32* qr = qq + ((size_t)h * TP + r) * (D / 8) + (size_t)(g * G) / 8;
        for (int j = 0; j < G; j += 8) {
            u32 w = 0;
#pragma unroll
            for (int e = 0; e < 8; e++) {
                int c = (int)lrintf(xr[j + e] / s);
                c = c > 7 ? 7 : (c < -8 ? -8 : c);
                w |= (u32)(c & 0xF) << (4 * e);
            }
            qr[j / 8] = w;
        }
    }
}
void k_attn_q_quant(u32* qq, float* qs, const float* x, int T, int H, int D, int G, int TP) {
    const int total = H * T * (D / G);
    const int th = 128;
    const int blk = (total + th - 1) / th;
    attn_q_quant_k<<<blk, th>>>(qq, qs, x, T, H, D, G, TP);
    CK(hipGetLastError());
}

// ==================== 激活量化（8bit → 两个 int4 数字）=======================
// q8 = clamp(round(x/s), -119, 119)，然后精确拆成 q8 = 16*h + l，h,l ∈ [-8,7]：
//   h = floor((q8 + 8) / 16)，l = q8 - 16h
// 于是 Σ w·q8 = 16·Σ w·h + Σ w·l，两个 int4 点积就能还原 8bit 激活的精度
// （每层误差从 int4 激活的 12~16% 降到 ~1.4%，代价是 GEMM 跑两遍）。
__global__ void quant_rows_a8_k(u32* qh, u32* ql, float* sc, float* sc16, const float* x,
                                int K, int G, int rows, int in_stride, int in_off) {
    const int row = blockIdx.x;
    const int g = threadIdx.x;
    const float* xr = x + (size_t)row * in_stride + in_off + (size_t)g * G;
    float amax = 0.f;
#pragma unroll 4
    for (int i = 0; i < G; i++) amax = fmaxf(amax, fabsf(xr[i]));
    const float s = amax > 0.f ? amax / 119.f : 1.f;
    sc[(size_t)g * rows + row] = s;
    sc16[(size_t)g * rows + row] = s * 16.f;
    u32* hr = qh + (size_t)row * (K / 8) + (size_t)(g * G) / 8;
    u32* lr = ql + (size_t)row * (K / 8) + (size_t)(g * G) / 8;
    for (int i = 0; i < G; i += 8) {
        u32 wh = 0, wl = 0;
#pragma unroll
        for (int e = 0; e < 8; e++) {
            int q8 = (int)lrintf(xr[i + e] / s);
            q8 = q8 > 119 ? 119 : (q8 < -128 ? -128 : q8);
            const int h = (q8 + 8) >> 4;               // floor，负数也正确
            const int l = q8 - 16 * h;                 // ∈ [-8,7]
            wh |= (u32)(h & 0xF) << (4 * e);
            wl |= (u32)(l & 0xF) << (4 * e);
        }
        hr[i / 8] = wh;
        lr[i / 8] = wl;
    }
}
void k_quant_rows_a8(u32* qh, u32* ql, float* sc, float* sc16, const float* x,
                     int rows, int K, int G, int in_stride, int in_off) {
    if (in_stride == 0) in_stride = K;
    quant_rows_a8_k<<<rows, K / G>>>(qh, ql, sc, sc16, x, K, G, rows, in_stride, in_off);
    CK(hipGetLastError());
}

// ============================== 词嵌入 =====================================
__global__ void embed_k(float* y, const u32* tbl, const unsigned short* sc, const int* ids,
                        int D, int G) {
    const int row = blockIdx.x;
    const int id = ids[row];
    for (int kd = threadIdx.x; kd < D / 8; kd += blockDim.x) {
        const u32 w = tbl[(size_t)id * (D / 8) + kd];
        const unsigned short sh = sc[(size_t)id * (D / G) + (kd * 8) / G];
        const uint32_t sg = (sh >> 15) & 1, ex = (sh >> 10) & 0x1F, ma = sh & 0x3FF;
        float fs = ex == 0 ? ldexpf(ma / 1024.f, -14) : ldexpf(1.f + ma / 1024.f, (int)ex - 15);
        const float s = sg ? -fs : fs;
        float* o = y + (size_t)row * D + kd * 8;
#pragma unroll
        for (int e = 0; e < 8; e++) {
            int c = (int)((w >> (4 * e)) & 0xF);
            if (c >= 8) c -= 16;
            o[e] = (float)c * s;
        }
    }
}
void k_embed(float* y, const u32* tbl, const unsigned short* sc, const int* ids, int n, int D, int G) {
    embed_k<<<n, 256>>>(y, tbl, sc, ids, D, G); CK(hipGetLastError());
}

// ============================= KV cache 写入 ===============================
// K：f32 [T][H][D] → kc[H][kv][D/8] + ksc[H][D/G][kv]（组优先，FA 内核直接用）
__global__ void kv_append_k_k(u32* kc, float* ksc, const float* kf, int t0, int T, int H,
                              int D, int G, int kv_total) {
    const int t = blockIdx.x, h = blockIdx.y, g = threadIdx.x;
    const int kv = t0 + t;
    const float* xr = kf + ((size_t)t * H + h) * D + (size_t)g * G;
    float amax = 0.f;
#pragma unroll 4
    for (int i = 0; i < G; i++) amax = fmaxf(amax, fabsf(xr[i]));
    const float s = amax > 0.f ? amax / 7.f : 1.f;
    ksc[((size_t)h * (D / G) + g) * kv_total + kv] = s;
    u32* qr = kc + ((size_t)h * kv_total + kv) * (D / 8) + (size_t)(g * G) / 8;
    for (int i = 0; i < G; i += 8) {
        u32 w = 0;
#pragma unroll
        for (int e = 0; e < 8; e++) {
            int c = (int)lrintf(xr[i + e] / s);
            c = c > 7 ? 7 : (c < -8 ? -8 : c);
            w |= (u32)(c & 0xF) << (4 * e);
        }
        qr[i / 8] = w;
    }
}
void k_kv_append_k(u32* kc, float* ksc, const float* kf, int t0, int T, int H, int D, int G, int kv_total) {
    kv_append_k_k<<<dim3(T, H), D / G>>>(kc, ksc, kf, t0, T, H, D, G, kv_total);
    CK(hipGetLastError());
}

// V：f32 [T][H][D] → 暂存 stage[H][BG][D]，整块重量化为
//    vc[H][kv/8][D]（dword 内 8 个 key，低半字节 = 偶数 key）+ vsc[H][tile][D]
__global__ void kv_append_v_k(u32* vc, float* vsc, float* stage, const float* vf,
                              int t0, int T, int H, int D, int BG, int kv_total) {
    const int h = blockIdx.x;
    const int d = threadIdx.x;
    float* st = stage + ((size_t)h * BG) * D + d;
    u32* out = vc + ((size_t)h * (kv_total / 8)) * D;
    // 一次 append 可能横跨多个 tile（预填充 T 可以远大于 BG），必须按 tile 分段处理：
    // 之前只按 slot0 + T 一路写 stage[H][BG][D]，T > BG-slot0 时直接越界（实测 T=100
    // 触发 VM fault）。
    int done = 0;
    while (done < T) {
        const int kv = t0 + done;
        const int tile = kv / BG;
        const int slot0 = kv % BG;
        const int n = (T - done < BG - slot0) ? (T - done) : (BG - slot0);
        for (int i = 0; i < n; i++)
            st[(size_t)(slot0 + i) * D] = vf[((size_t)(done + i) * H + h) * D + d];
        const int nval = slot0 + n;
        float amax = 0.f;
        for (int i = 0; i < nval; i++) amax = fmaxf(amax, fabsf(st[(size_t)i * D]));
        const float s = amax > 0.f ? amax / 7.f : 1.f;
        vsc[((size_t)h * (kv_total / BG) + tile) * D + d] = s;
        for (int i = 0; i < BG; i += 8) {
            u32 w = 0;
#pragma unroll
            for (int e = 0; e < 8; e++) {
                const float v = (i + e < nval) ? st[(size_t)(i + e) * D] : 0.f;
                int c = (int)lrintf(v / s);
                c = c > 7 ? 7 : (c < -8 ? -8 : c);
                w |= (u32)(c & 0xF) << (4 * e);
            }
            out[((size_t)tile * (BG / 8) + i / 8) * D + d] = w;
        }
        done += n;
    }
}
void k_kv_append_v(u32* vc, float* vsc, float* stage, const float* vf,
                   int t0, int T, int H, int D, int BG, int kv_total) {
    kv_append_v_k<<<H, D>>>(vc, vsc, stage, vf, t0, T, H, D, BG, kv_total);
    CK(hipGetLastError());
}

// ============================ 卷积（SSM 前置）==============================
__global__ void conv1d_silu_k(float* out, const float* in, const float* w, const float* state,
                              int T, int C, int K) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)T * C) return;
    const int c = (int)(idx % C);
    const int t = (int)(idx / C);
    float acc = 0.f;
    for (int j = 0; j < K; j++) {
        const int src = t - (K - 1) + j;
        const float x = src < 0 ? state[(size_t)(K - 1 + src) * C + c] : in[(size_t)src * C + c];
        acc += w[(size_t)c * K + j] * x;
    }
    out[idx] = acc / (1.f + __expf(-acc));
}
void k_conv1d_silu(float* out, const float* in, const float* w, float* state, int T, int C, int K) {
    conv1d_silu_k<<<grid_for((long long)T * C, 256), 256>>>(out, in, w, state, T, C, K);
    CK(hipGetLastError());
}
// 状态更新：state 保存整段序列（含本块）最后 K-1 行；snap 非空时把每个 token 之后
// 的 state 也写进 [T][K-1][C]，供 MTP 验证批按接受长度回滚。
__global__ void conv_state_update_k(float* state, float* snap, const float* in,
                                    const float* prev, int T, int C, int K) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)(K - 1) * C) return;
    const int c = (int)(idx % C);
    const int r = (int)(idx / C);
    const int src = T - (K - 1) + r;               // 目标行在「state + 本块」拼起来的序列里的位置
    state[(size_t)r * C + c] = src < 0 ? prev[(size_t)((K - 1) + src) * C + c]
                                       : in[(size_t)src * C + c];
    if (snap) {
        for (int t = 0; t < T; t++) {
            // 处理完第 t 个 token 后，state[r] 保存的是它之前第 K-2-r 个输入：
            // 起点是 t-(K-2)+r（这里比最后的整块更新多 1，之前漏了这个 +1）。
            const int st = t - (K - 2) + r;
            snap[((size_t)t * (K - 1) + r) * C + c] =
                st < 0 ? prev[(size_t)((K - 1) + st) * C + c] : in[(size_t)st * C + c];
        }
    }
}
void k_conv_state_update(float* state, const float* in, const float* prev, int T, int C, int K) {
    conv_state_update_k<<<grid_for((long long)(K - 1) * C, 128), 128>>>(state, nullptr, in, prev,
                                                                        T, C, K);
    CK(hipGetLastError());
}
void k_conv_state_update_snap(float* state, float* snap, const float* in, const float* prev,
                              int T, int C, int K) {
    conv_state_update_k<<<grid_for((long long)(K - 1) * C, 128), 128>>>(state, snap, in, prev,
                                                                        T, C, K);
    CK(hipGetLastError());
}

// a = W_a·x、b = W_b·x（两个 [Hv][K] 的 f32 权重），再顺带算
//   beta = sigmoid(b)、g = -exp(A_log[h])·softplus(a + dt_bias[h])
// 原来这一串是 4 个内核（2 个 f32 GEMV + sigmoid + ssm gate），解码时每层 4 次启动、
// 实测 3.9 ms/token（48 层 × 81 us，而权重总共只有 96 MB/token ≈ 0.12 ms）。
// 合成一个：grid = (Hv, T)，一个 block 管一个 (head, token)。
// 输出：gab[t*Hv+h] = a（保持 db 阶段 7 的 a|b 语义）、beta_out[t*Hv+h]、
//       gg[t*Hv+h] = g。
__global__ void ssm_ab_gate_k(float* __restrict__ gab, float* __restrict__ beta_out,
                              float* __restrict__ gg, const float* __restrict__ wa,
                              const float* __restrict__ wb, const float* __restrict__ x,
                              const float* __restrict__ dt, const float* __restrict__ alog,
                              int Hv, int K) {
    const int h = blockIdx.x, t = blockIdx.y;
    const float* xr = x + (size_t)t * K;
    const float* ar = wa + (size_t)h * K;
    const float* br = wb + (size_t)h * K;
    float sa = 0.f, sb = 0.f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        const float xv = xr[k];
        sa = fmaf(ar[k], xv, sa);
        sb = fmaf(br[k], xv, sb);
    }
    for (int o = 16; o; o >>= 1) { sa += __shfl_xor(sa, o); sb += __shfl_xor(sb, o); }
    __shared__ float red[16];
    const int w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    if ((threadIdx.x & 31) == 0) { red[w] = sa; red[nw + w] = sb; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float a = 0.f, b = 0.f;
        for (int i = 0; i < nw; i++) { a += red[i]; b += red[nw + i]; }
        gab[(size_t)t * Hv + h] = a;
        beta_out[(size_t)t * Hv + h] = 1.f / (1.f + __expf(-b));
        const float z = a + dt[h];
        const float sp = z > 20.f ? z : log1pf(__expf(z));
        gg[(size_t)t * Hv + h] = -__expf(alog[h]) * sp;
    }
}
void k_ssm_ab_gate(float* gab, float* beta_out, float* gg, const float* wa, const float* wb,
                   const float* x, const float* dt, const float* alog, int T, int Hv, int K) {
    ssm_ab_gate_k<<<dim3(Hv, T), 256>>>(gab, beta_out, gg, wa, wb, x, dt, alog, Hv, K);
    CK(hipGetLastError());
}

// ========================== gated delta net 递推 ============================
// 旧版：一线程一列，state 布局 [h][col][i]，st[128] 全进寄存器。
//   两个问题：(1) 同一时刻相邻线程（列号相邻）的地址相差 D=128 个 float，一个
//   warp 的每次 load 要碰 32 条 cache line（只用 4/32 字节）；(2) 每线程 128 个
//   活跃寄存器 + 96 个 64 线程的块 → 实测 0.09 ms/层（4.4 ms/token）。
// 新版：state 改成 [h][i][col]（列连续），一个 warp 沿列走、一个 warp 管一段 i
//   （i 切成 GDN_SPLIT 段），跨 warp 的两段归约走共享内存。合并访存 + 每线程
//   只有 32 个寄存器 + 更多并行块。
#define GDN_SPLIT 4
template <int DSUB>
__global__ void gdn_k2(float* out, const float* q, const float* k, const float* v,
                       const float* g, const float* beta, float* state, float* snap,
                       int T, int Hk, int Hv, int D, int rep) {
    const int h = blockIdx.x;
    const int sub = threadIdx.x >> 5;                  // 0..GDN_SPLIT-1：管哪一段 i
    const int lane = threadIdx.x & 31;
    const int col = blockIdx.y * 32 + lane;
    if (col >= D) return;                              // 只在 __syncthreads() 之前返回
    const int kh = h / rep;
    const float qscale = 1.f / sqrtf((float)D);
    float* sp = state + ((size_t)h * D + sub * DSUB) * D + col;   // [i][col] 布局
    float st[DSUB];
    for (int i = 0; i < DSUB; i++) st[i] = sp[(size_t)i * D];
    __shared__ float red[GDN_SPLIT][32];
    for (int t = 0; t < T; t++) {
        const float gt = __expf(g[(size_t)t * Hv + h]);
        const float bt = beta[(size_t)t * Hv + h];
        const float* kt = k + ((size_t)t * Hk + kh) * D + sub * DSUB;
        const float* qt = q + ((size_t)t * Hk + kh) * D + sub * DSUB;
        float sk = 0.f;
#pragma unroll 4
        for (int i = 0; i < DSUB; i++) { st[i] *= gt; sk = fmaf(st[i], kt[i], sk); }
        red[sub][lane] = sk;
        __syncthreads();
        float s = 0.f;
#pragma unroll
        for (int j = 0; j < GDN_SPLIT; j++) s += red[j][lane];
        __syncthreads();
        const float d = bt * (v[((size_t)t * Hv + h) * D + col] - s);
        float o = 0.f;
#pragma unroll 4
        for (int i = 0; i < DSUB; i++) { st[i] = fmaf(kt[i], d, st[i]); o = fmaf(st[i], qt[i], o); }
        red[sub][lane] = o;
        __syncthreads();
        float os = 0.f;
#pragma unroll
        for (int j = 0; j < GDN_SPLIT; j++) os += red[j][lane];
        if (sub == 0) out[((size_t)t * Hv + h) * D + col] = os * qscale;
        __syncthreads();
        if (snap) {
            float* sn = snap + (((size_t)t * Hv + h) * D + sub * DSUB) * D + col;
#pragma unroll 4
            for (int i = 0; i < DSUB; i++) sn[(size_t)i * D] = st[i];
        }
    }
    for (int i = 0; i < DSUB; i++) sp[(size_t)i * D] = st[i];
}

// 旧版（保留做 A/B 对照：RT_GDN_OLD=1）
__global__ void gdn_k(float* out, const float* q, const float* k, const float* v,
                      const float* g, const float* beta, float* state,
                      int T, int Hk, int Hv, int D, int rep, int khmod) {
    const int h   = blockIdx.x;
    const int col = blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= D) return;
    const int kh = khmod ? (h % Hk) : (h / rep);   // khmod=1 用 llama.cpp 的 j%Hk 语义
    const float qscale = 1.f / sqrtf((float)D);      // HF: query 先乘 1/sqrt(head_dim)
    float* sp = state + ((size_t)h * D + col) * D;
    float st[GDN_MAX_D];
#pragma unroll 4
    for (int i = 0; i < D; i++) st[i] = sp[i];
    for (int t = 0; t < T; t++) {
        const float gt = __expf(g[(size_t)t * Hv + h]);
        const float bt = beta[(size_t)t * Hv + h];
        const float* kt = k + ((size_t)t * Hk + kh) * D;
        const float* qt = q + ((size_t)t * Hk + kh) * D;
        float sk = 0.f;
#pragma unroll 4
        for (int i = 0; i < D; i++) { st[i] *= gt; sk += st[i] * kt[i]; }
        const float d = bt * (v[((size_t)t * Hv + h) * D + col] - sk);
        float o = 0.f;
#pragma unroll 4
        for (int i = 0; i < D; i++) { st[i] += kt[i] * d; o += st[i] * qt[i] * qscale; }
        out[((size_t)t * Hv + h) * D + col] = o;
    }
#pragma unroll 4
    for (int i = 0; i < D; i++) sp[i] = st[i];
}
void k_gdn(float* out, const float* q, const float* k, const float* v, const float* g,
           const float* beta, float* state, int T, int Hk, int Hv, int D, int rep) {
    k_gdn_snap(out, q, k, v, g, beta, state, nullptr, T, Hk, Hv, D, rep);
}
void k_gdn_snap(float* out, const float* q, const float* k, const float* v, const float* g,
                const float* beta, float* state, float* snap, int T, int Hk, int Hv, int D,
                int rep) {
    static const int khmod = getenv("RT_KH_MOD") ? atoi(getenv("RT_KH_MOD")) : 0;
    static const int old = getenv("RT_GDN_OLD") ? atoi(getenv("RT_GDN_OLD")) : 0;
    if (old || khmod || D != 128) {
        if (snap) { printf("k_gdn_snap: 回滚快照只支持 gdn_k2 路径\n"); exit(1); }
        gdn_k<<<dim3(Hv, (D + 63) / 64), 64>>>(out, q, k, v, g, beta, state, T, Hk, Hv, D, rep,
                                               khmod);
    } else {
        gdn_k2<128 / GDN_SPLIT><<<dim3(Hv, D / 32), 32 * GDN_SPLIT>>>(
            out, q, k, v, g, beta, state, snap, T, Hk, Hv, D, rep);
    }
    CK(hipGetLastError());
}

// ===================== 其它运行时要用的补充算子 =============================
// 抽头：从 [T][tok_stride] 的大张量里抽出每 (t,h) 的一段 D 维，紧凑成 [T][n_head][D]
__global__ void gather_heads_k(float* dst, const float* src, int n_head, int D,
                               int tok_stride, int head_off, int head_stride, long long n) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const int d = (int)(idx % D);
    const long long rest = idx / D;
    const int h = (int)(rest % n_head);
    const int t = (int)(rest / n_head);
    dst[idx] = src[(size_t)t * tok_stride + (size_t)h * head_stride + head_off + d];
}
void k_gather_heads(float* dst, const float* src, int T, int n_head, int D,
                    int tok_stride, int head_off, int head_stride) {
    const long long n = (long long)T * n_head * D;
    gather_heads_k<<<grid_for(n, 256), 256>>>(dst, src, n_head, D, tok_stride, head_off,
                                              head_stride, n);
    CK(hipGetLastError());
}
// 反操作：把 [T][n_head][D] 的紧凑结果塞回大张量的对应位置
// 注意：src 的实际行布局是 [n_head][src_rows][D]（注意力内核按补齐行数写），
// 所以这里必须用 src_rows 算头间距——用 T 会让 T 不是 64 的倍数时整段错位。
__global__ void scatter_heads_k(float* dst, const float* src, int n_head, int D,
                                int tok_stride, int head_off, int src_rows, long long n) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const int d = (int)(idx % D);
    const long long rest = idx / D;
    const int h = (int)(rest % n_head);
    const int t = (int)(rest / n_head);
    dst[(size_t)t * tok_stride + h * D + head_off + d] =
        src[((size_t)h * src_rows + t) * D + d];
}
void k_scatter_heads(float* dst, const float* src, int T, int n_head, int D,
                     int tok_stride, int head_off, int src_rows) {
    const long long n = (long long)T * n_head * D;
    scatter_heads_k<<<grid_for(n, 256), 256>>>(dst, src, n_head, D, tok_stride, head_off,
                                               src_rows, n);
    CK(hipGetLastError());
}
__global__ void scale_mul_k(float* x, float s, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= s;
}
void k_scale(float* x, float s, long long n) {
    scale_mul_k<<<grid_for(n, 256), 256>>>(x, s, n); CK(hipGetLastError());
}
// 门控 RMSNorm（权重非零中心，gate 用 silu）：y = rmsnorm(x,w) * silu(g)
__global__ void rmsnorm_gated_k(float* y, const float* x, const float* w, const float* g,
                                int D, float eps) {
    const int row = blockIdx.x;
    const float* xr = x + (size_t)row * D;
    const float* gr = g + (size_t)row * D;
    float* yr = y + (size_t)row * D;
    float s = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) s += xr[i] * xr[i];
    __shared__ float sh[32];
    for (int o = 16; o; o >>= 1) s += __shfl_xor(s, o);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x < 32) {
        float t = (threadIdx.x < (int)(blockDim.x >> 5)) ? sh[threadIdx.x] : 0.f;
        for (int o = 16; o; o >>= 1) t += __shfl_xor(t, o);
        if (threadIdx.x == 0) sh[0] = rsqrtf(t / (float)D + eps);
    }
    __syncthreads();
    const float inv = sh[0];
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        const float gg = gr[i];
        yr[i] = xr[i] * inv * w[i] * (gg / (1.f + __expf(-gg)));
    }
}
void k_rmsnorm_gated(float* y, const float* x, const float* w, const float* g, int rows, int D,
                     float eps) {
    rmsnorm_gated_k<<<rows, D >= 128 ? 128 : 64>>>(y, x, w, g, D, eps); CK(hipGetLastError());
}
// 卷积输出按 [q | k | v] 切成紧凑的 [T][H*D] 三份
__global__ void split_qkv_k(float* q, float* k, float* v, const float* c, int T,
                            int qn, int kn, int vn) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const int C = qn + kn + vn;
    if (idx >= (long long)T * C) return;
    const int c_ = (int)(idx % C);
    const int t = (int)(idx / C);
    const float x = c[(size_t)t * C + c_];
    if (c_ < qn) q[(size_t)t * qn + c_] = x;
    else if (c_ < qn + kn) k[(size_t)t * kn + (c_ - qn)] = x;
    else v[(size_t)t * vn + (c_ - qn - kn)] = x;
}
void k_split_qkv(float* q, float* k, float* v, const float* c, int T, int qn, int kn, int vn) {
    const long long n = (long long)T * (qn + kn + vn);
    split_qkv_k<<<grid_for(n, 256), 256>>>(q, k, v, c, T, qn, kn, vn); CK(hipGetLastError());
}
