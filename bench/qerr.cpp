// What does a larger int4 quantization group cost in accuracy?
//
// x3_int4.cpp shows the GEMM runs 65% of the dot8 ceiling when the group is
// raised from 128 to 512 elements, because the per-group scale fold then costs
// a quarter as many instructions.  This probe isolates the accuracy side: it
// quantizes the same matrices at several group sizes and reports how far the
// int4 result lands from the fp32 one.
//
// Host only (no GPU): the point is the quantizer, not the kernel.
//   g++ -O2 -o qerr qerr.cpp
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>

static float quant_group(const float* x, int G, int8_t* q) {
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

// quantize + dequantize one matrix; returns relative Frobenius error
static double quant_err(const std::vector<float>& X, int R, int K, int G,
                        std::vector<float>& out) {
    out.assign((size_t)R * K, 0.f);
    std::vector<int8_t> q(G);
    double num = 0, den = 0;
    for (int r = 0; r < R; r++)
        for (int g = 0; g < K / G; g++) {
            const float* x = &X[(size_t)r * K + g * G];
            float s = quant_group(x, G, q.data());
            for (int i = 0; i < G; i++) {
                float d = q[i] * s;
                out[(size_t)r * K + g * G + i] = d;
                num += (double)(d - x[i]) * (d - x[i]);
                den += (double)x[i] * x[i];
            }
        }
    return sqrt(num / den);
}

int main(void) {
    const int K = 5120, ROWS_A = 64, ROWS_B = 64;
    const int groups[] = { 64, 128, 256, 512, 1024 };
    const int NG = (int)(sizeof(groups) / sizeof(groups[0]));

    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> uu(0.f, 1.f);

    // activations: unit normal with 1% outliers x10; weights: 0.2% x6
    std::vector<float> A((size_t)ROWS_A * K), B((size_t)ROWS_B * K);
    for (auto& v : A) { v = nd(rng); if (uu(rng) < 0.01f) v *= 10.f; }
    for (auto& v : B) { v = nd(rng); if (uu(rng) < 0.002f) v *= 6.f; }

    std::vector<double> C((size_t)ROWS_A * ROWS_B, 0.0);
    for (int i = 0; i < ROWS_A; i++)
        for (int j = 0; j < ROWS_B; j++) {
            double s = 0;
            for (int k = 0; k < K; k++)
                s += (double)A[(size_t)i * K + k] * B[(size_t)j * K + k];
            C[(size_t)i * ROWS_B + j] = s;
        }

    printf("int4 quantization error, 64x5120 by 5120x64, with outliers\n");
    printf("%-8s %15s %15s %16s\n", "group", "weights relFrob", "acts relFrob",
           "GEMM relFrob");
    for (int gi = 0; gi < NG; gi++) {
        int G = groups[gi];
        std::vector<float> Aq, Bq;
        double ea = quant_err(A, ROWS_A, K, G, Aq);
        double eb = quant_err(B, ROWS_B, K, G, Bq);
        double num = 0, den = 0;
        for (int i = 0; i < ROWS_A; i++)
            for (int j = 0; j < ROWS_B; j++) {
                double s = 0;
                for (int k = 0; k < K; k++)
                    s += (double)Aq[(size_t)i * K + k] * Bq[(size_t)j * K + k];
                double d = s - C[(size_t)i * ROWS_B + j];
                num += d * d;
                den += C[(size_t)i * ROWS_B + j] * C[(size_t)i * ROWS_B + j];
            }
        printf("%-8d %15.3e %15.3e %16.3e\n", G, eb, ea, sqrt(num / den));
    }
    return 0;
}
