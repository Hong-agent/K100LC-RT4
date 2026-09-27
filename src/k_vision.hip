// 视觉塔专用内核：LayerNorm / GELU / bias / 2D-RoPE / 全注意力。
// 线性层直接复用 src/k_gemm.hip 的 int4 GEMM。
#include "vision_kernels.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

// f16 视觉线性层：一个 block 算 64x64 输出，K 方向每次 32。
// 视觉塔对 int4 太敏感，RT4 里存 f16；这个内核不追求极限吞吐，5~10ms/图足够。
#define VBM 64
#define VBN 64
#define VBK 32

__global__ void vit_linear_f16_kernel(float* __restrict__ y, const __half* __restrict__ w,
                                      const float* __restrict__ x, int M, int N, int K,
                                      int x_stride, int y_stride) {
    __shared__ __half As[VBM][VBK];
    __shared__ __half Bs[VBN][VBK];
    const int bm = blockIdx.y * VBM, bn = blockIdx.x * VBN;
    const int tid = threadIdx.x;
    const int tr = tid / 16, tc = tid % 16;
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.f;
    for (int kb = 0; kb < K; kb += VBK) {
        for (int i = tid; i < VBM * VBK; i += blockDim.x) {
            const int r = i / VBK, c = i % VBK;
            const int m = bm + r, k = kb + c;
            As[r][c] = (m < M && k < K) ? __float2half(x[(size_t)m * x_stride + k])
                                        : __float2half(0.f);
        }
        for (int i = tid; i < VBN * VBK; i += blockDim.x) {
            const int r = i / VBK, c = i % VBK;
            const int n = bn + r, k = kb + c;
            Bs[r][c] = (n < N && k < K) ? w[(size_t)n * K + k] : __float2half(0.f);
        }
        __syncthreads();
        for (int kk = 0; kk < VBK; kk++) {
            __half a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; i++) a[i] = As[tr * 4 + i][kk];
#pragma unroll
            for (int j = 0; j < 4; j++) b[j] = Bs[tc * 4 + j][kk];
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++)
                    acc[i][j] += __half2float(a[i]) * __half2float(b[j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int m = bm + tr * 4 + i, n = bn + tc * 4 + j;
            if (m < M && n < N) y[(size_t)m * y_stride + n] = acc[i][j];
        }
}

void k_vit_linear_f16(float* y, const uint16_t* w, const float* x, int M, int N, int K,
                      int x_stride, int y_stride) {
    if (!x_stride) x_stride = K;
    if (!y_stride) y_stride = N;
    dim3 grid((N + VBN - 1) / VBN, (M + VBM - 1) / VBM);
    vit_linear_f16_kernel<<<grid, 256>>>(y, (const __half*)w, x, M, N, K,
                                         x_stride, y_stride);
}

__global__ void vit_ln_kernel(float* __restrict__ y, const float* __restrict__ x,
                              const float* __restrict__ w, const float* __restrict__ b,
                              int rows, int D, float eps) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    const float* xr = x + (size_t)r * D;
    float* yr = y + (size_t)r * D;
    float s = 0.f, s2 = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        const float v = xr[i];
        s += v;
        s2 += v * v;
    }
    __shared__ float red[256];
    red[threadIdx.x] = s;
    __syncthreads();
    for (int off = blockDim.x >> 1; off; off >>= 1) {
        if ((int)threadIdx.x < off) red[threadIdx.x] += red[threadIdx.x + off];
        __syncthreads();
    }
    const float mean = red[0] / D;
    __syncthreads();
    red[threadIdx.x] = s2;
    __syncthreads();
    for (int off = blockDim.x >> 1; off; off >>= 1) {
        if ((int)threadIdx.x < off) red[threadIdx.x] += red[threadIdx.x + off];
        __syncthreads();
    }
    const float var = red[0] / D - mean * mean;
    const float inv = rsqrtf(var + eps);
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        yr[i] = (xr[i] - mean) * inv * w[i] + b[i];
}

void k_vit_layernorm(float* y, const float* x, const float* w, const float* b,
                     int rows, int D, float eps) {
    vit_ln_kernel<<<rows, 256>>>(y, x, w, b, rows, D, eps);
}

__global__ void vit_gelu_kernel(float* __restrict__ y, const float* __restrict__ x,
                                long long n, int exact) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i];
    y[i] = exact ? 0.5f * v * (1.f + erff(v * 0.7071067811865476f))
                 : 0.5f * v * (1.f + tanhf(0.7978845608028654f *
                                           (v + 0.044715f * v * v * v)));
}

void k_vit_gelu(float* y, const float* x, long long n, int exact) {
    const int th = 256;
    vit_gelu_kernel<<<(int)((n + th - 1) / th), th>>>(y, x, n, exact);
}

__global__ void vit_bias_kernel(float* __restrict__ y, const float* __restrict__ b,
                                int rows, int N) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)rows * N) return;
    y[i] += b[i % N];
}

void k_vit_bias_add(float* y, const float* b, int rows, int N) {
    const long long n = (long long)rows * N;
    const int th = 256;
    vit_bias_kernel<<<(int)((n + th - 1) / th), th>>>(y, b, rows, N);
}

__global__ void vit_bias_s_kernel(float* __restrict__ y, const float* __restrict__ b,
                                  int rows, int N, int stride) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)rows * N) return;
    y[(i / N) * stride + (i % N)] += b[i % N];
}

void k_vit_bias_add_s(float* y, const float* b, int rows, int N, int stride) {
    const long long n = (long long)rows * N;
    const int th = 256;
    vit_bias_s_kernel<<<(int)((n + th - 1) / th), th>>>(y, b, rows, N, stride);
}

// 2D RoPE：cos/sin 已经是 [rows][dim]，且前后半段相同。
// qkv 行布局：[3][heads][dim]，对 q_offset=0 和 k_offset=heads*dim 原地旋转。
__global__ void vit_rope_kernel(float* __restrict__ qkv, const float* __restrict__ cos,
                                const float* __restrict__ sin, int rows, int heads,
                                int dim, int stride) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    const int half = dim / 2;
    const int nitem = heads * half;
    float* row = qkv + (size_t)r * stride;
    const float* cr = cos + (size_t)r * dim;
    const float* sr = sin + (size_t)r * dim;
    for (int idx = threadIdx.x; idx < nitem; idx += blockDim.x) {
        const int h = idx / half;
        const int d = idx % half;
        const int i0 = h * dim + d;
        const int i1 = i0 + half;
        const float c0 = cr[d], s0 = sr[d];
        const float c1 = cr[d + half], s1 = sr[d + half];
        // q 和 k 两块都要做
        for (int base = 0; base < 2; base++) {
            const int off = base * heads * dim;
            const float x0 = row[off + i0];
            const float x1 = row[off + i1];
            row[off + i0] = x0 * c0 - x1 * s0;
            row[off + i1] = x1 * c1 + x0 * s1;
        }
    }
}

void k_vit_rope(float* qkv, const float* cos, const float* sin, int rows, int heads,
                int dim, int stride) {
    vit_rope_kernel<<<rows, 256>>>(qkv, cos, sin, rows, heads, dim, stride);
}

// 每个 block 负责 (head, query row)。128 线程；acc 用 shared [128][96]（dim<=96）。
__global__ void vit_attn_kernel(float* __restrict__ out, const float* __restrict__ qkv,
                                int rows, int heads, int dim, int stride, float scale) {
    const int h = blockIdx.x;
    const int r = blockIdx.y;
    const int lane = threadIdx.x;
    const int q_off = 0;
    const int k_off = heads * dim;
    const int v_off = 2 * heads * dim;
    const float* q = qkv + (size_t)r * stride + h * dim;
    const int half_threads = blockDim.x;

    float local_max = -1e30f;
    for (int k = lane; k < rows; k += half_threads) {
        const float* kk = qkv + (size_t)k * stride + k_off + h * dim;
        float s = 0.f;
        for (int d = 0; d < dim; d++) s += q[d] * kk[d];
        s *= scale;
        local_max = fmaxf(local_max, s);
    }
    __shared__ float red[256];
    __shared__ float acc_sh[128][72];
    red[lane] = local_max;
    __syncthreads();
    for (int off = half_threads >> 1; off; off >>= 1) {
        if (lane < off) red[lane] = fmaxf(red[lane], red[lane + off]);
        __syncthreads();
    }
    const float vmax = red[0];
    __syncthreads();

    float local_sum = 0.f;
    for (int k = lane; k < rows; k += half_threads) {
        const float* kk = qkv + (size_t)k * stride + k_off + h * dim;
        float s = 0.f;
        for (int d = 0; d < dim; d++) s += q[d] * kk[d];
        local_sum += __expf(s * scale - vmax);
    }
    red[lane] = local_sum;
    __syncthreads();
    for (int off = half_threads >> 1; off; off >>= 1) {
        if (lane < off) red[lane] += red[lane + off];
        __syncthreads();
    }
    const float vsum = red[0];
    __syncthreads();

    float acc[72];
    for (int d = 0; d < dim; d++) acc[d] = 0.f;
    for (int k = lane; k < rows; k += half_threads) {
        const float* kk = qkv + (size_t)k * stride + k_off + h * dim;
        float s = 0.f;
        for (int d = 0; d < dim; d++) s += q[d] * kk[d];
        const float p = __expf(s * scale - vmax) / vsum;
        const float* vv = qkv + (size_t)k * stride + v_off + h * dim;
        for (int d = 0; d < dim; d++) acc[d] += p * vv[d];
    }
    for (int d = 0; d < dim; d++) acc_sh[lane][d] = acc[d];
    __syncthreads();
    if (lane < dim) {
        float s = 0.f;
        for (int t = 0; t < half_threads; t++) s += acc_sh[t][lane];
        out[(size_t)r * heads * dim + h * dim + lane] = s;
    }
}

void k_vit_attention(float* out, const float* qkv, int rows, int heads, int dim,
                     int stride, float scale) {
    if (dim > 72) { printf("vision attention dim=%d > 72\n", dim); exit(1); }
    vit_attn_kernel<<<dim3(heads, rows), 128>>>(out, qkv, rows, heads, dim, stride, scale);
}
