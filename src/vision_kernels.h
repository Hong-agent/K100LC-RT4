// 视觉塔专用元素级 / 注意力内核。线性层复用现有 int4 GEMM。
#pragma once
#include <cstdint>

// y[M,N] = x[M,K] @ W[N,K]^T；W 是 RT4 里的 f16。
// x_stride / y_stride 为 0 时分别按 K / N 紧凑排布（fc2 的输入有 padding 需要显式 stride）。
void k_vit_linear_f16(float* y, const uint16_t* w, const float* x, int M, int N, int K,
                      int x_stride, int y_stride);
void k_vit_layernorm(float* y, const float* x, const float* w, const float* b,
                     int rows, int D, float eps);
void k_vit_gelu(float* y, const float* x, long long n, int exact);
void k_vit_bias_add(float* y, const float* b, int rows, int N);
void k_vit_bias_add_s(float* y, const float* b, int rows, int N, int stride);
void k_vit_rope(float* qkv, const float* cos, const float* sin, int rows, int heads,
                int dim, int stride);
// qkv: [rows][3,heads,dim]；out: [rows][heads*dim]
void k_vit_attention(float* out, const float* qkv, int rows, int heads, int dim,
                     int stride, float scale);
