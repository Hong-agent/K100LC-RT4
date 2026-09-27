// RT4 解码 GEMV，W4A4 版：int4 权重 × int4 激活，用 v_dot8_i32_i4。
//
// 为什么比 W4A8 版（gemv_int4.hip）更快：
//   1. int4 权重 dword 和 int4 激活 dword **直接**就是 dot8 的两个源操作数
//      （都是「8 个有符号 int4、低半字节在前」，见 DESIGN-INT4.md 2.2 的实测语义），
//      内循环里连 unpack 都不需要，也没有 W4A8 那条 -8·Σa 修正项。
//   2. 激活从 int8（1 字节/权重）降到 int4（0.5 字节/权重），
//      M=4 时激活流量是权重的 4 倍，这一项直接砍半。
//   3. 激活按 **32 个一组**（= 一个 lane 一次 uint4 拿到的 32 个权重）给尺度，
//      于是权重尺度（每 128 一组，4 个 lane 共用）与激活尺度（每 32 一组，lane 私有）
//      相乘即可，**内循环里一次 shuffle 都不需要**——上一版测出组内归约吃掉 40% 带宽。
//
// 精度：来源模型本身就是 NVFP4（MLP 激活也是 4bit、组 16），W4A4 在它的设计范围内；
// 本程序会同时打印 int4/32 与 int8/128 两种激活量化的相对误差供权衡。
//
// 编译: hipcc -O3 --offload-arch=gfx926 -o gemv_w4a4 gemv_w4a4.hip
// 用法: ./gemv_w4a4 <tensor> <M> [rt4] [manifest] [reps]
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <random>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    printf("HIP ERR %s @%d: %s\n", #x, __LINE__, hipGetErrorString(e_)); exit(1);} } while (0)
#include "gemv_w4a4_core.h"

// ---------------------------------------------------------------- 存取辅助 ---
// 与主内核完全同构的纯读探针（uint4/2.5 次迭代），用来区分
// 「访存结构本身的极限」与「dot8 计算/延迟」。
__global__ void bw_probe4(const uint4* __restrict__ wq4, uint32_t* __restrict__ sink,
                          int N, int KD4) {
    const int lane = threadIdx.x & 63, warp = threadIdx.x >> 6;
    const int row = blockIdx.x * NWARP + warp;
    if (row >= N) return;
    const uint4* wrow = wq4 + (size_t)row * KD4;
    uint32_t acc = 0;
    for (int d = lane; d < KD4; d += 64) {
        uint4 v = wrow[d];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0xdeadbeefu) sink[row] = acc;
}

struct Ent {
    std::string name, kind;
    long long N, K, grp, q_off, s_off;
    double relerr;
};

static bool find_ent(const char* manifest, const std::string& want, Ent& e) {
    FILE* f = fopen(manifest, "rb");
    if (!f) { printf("open %s failed\n", manifest); exit(1); }
    std::string js;
    char buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) js.append(buf, n);
    fclose(f);
    size_t p = 0;
    while ((p = js.find("\"name\": \"", p)) != std::string::npos) {
        size_t q0 = p + 9, q1 = js.find('"', q0);
        std::string name = js.substr(q0, q1 - q0);
        size_t obj_end = js.find('}', q1);
        std::string obj = js.substr(q1, obj_end - q1);
        auto num = [&](const char* key, long long def) -> long long {
            size_t k = obj.find(key);
            if (k == std::string::npos) return def;
            return atoll(obj.c_str() + k + strlen(key));
        };
        if (name == want) {
            e.name = name;
            size_t k = obj.find("\"kind\": \"");
            size_t k2 = obj.find('"', k + 9);
            e.kind = obj.substr(k + 9, k2 - k - 9);
            e.N = num("\"shape\": [", 0);
            size_t comma = obj.find(',', obj.find("\"shape\": ["));
            e.K = atoll(obj.c_str() + comma + 1);
            e.grp = num("\"group\": ", 128);
            e.q_off = num("\"q_off\": ", 0);
            e.s_off = num("\"s_off\": ", 0);
            size_t r = obj.find("\"relerr\": ");
            e.relerr = r == std::string::npos ? 0 : atof(obj.c_str() + r + 10);
            return true;
        }
        p = obj_end;
    }
    return false;
}

static std::vector<uint8_t> pread_region(const char* path, long long off, long long len) {
    FILE* f = fopen(path, "rb");
    if (!f) { printf("open %s failed\n", path); exit(1); }
    fseeko(f, off, SEEK_SET);
    std::vector<uint8_t> v(len);
    if (fread(v.data(), 1, len, f) != (size_t)len) { printf("short read\n"); exit(1); }
    fclose(f);
    return v;
}

static float f16f(uint16_t sh) {
    uint32_t sg = (sh >> 15) & 1, ex = (sh >> 10) & 0x1F, ma = sh & 0x3FF;
    float f = ex == 0 ? ldexpf(ma / 1024.f, -14) : ldexpf(1.f + ma / 1024.f, (int)ex - 15);
    return sg ? -f : f;
}

int main(int argc, char** argv) {
    const char* tname = argc > 1 ? argv[1] : "model.language_model.layers.0.mlp.gate_proj.weight";
    int M = argc > 2 ? atoi(argv[2]) : 1;
    const char* rt4 = argc > 3 ? argv[3] : "/rt/qwen38_27b.rt4";
    const char* man = argc > 4 ? argv[4] : "/rt/qwen38_27b.rt4.json";
    int reps = argc > 5 ? atoi(argv[5]) : 30;

    Ent e;
    if (!find_ent(man, tname, e)) { printf("tensor %s not in manifest\n", tname); return 1; }
    const int N = (int)e.N, K = (int)e.K;
    if (e.kind != "i4") { printf("not i4\n"); return 1; }
    printf("tensor %s [%d, %d]  M=%d  ROWS=%d\n", tname, N, K, M, ROWS);

    auto q = pread_region(rt4, e.q_off, (long long)N * (K / 2));
    auto s = pread_region(rt4, e.s_off, (long long)N * (K / GRP) * 2);
    const double wbytes = (double)q.size();

    // 模拟激活：高斯 + 少量离群通道（与 gemv_int4.hip 同一分布，便于对照）
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> a((size_t)M * K);
    for (size_t i = 0; i < a.size(); i++) a[i] = nd(rng);
    for (int m = 0; m < M; m++) for (int i = 0; i < K; i += 997) a[(size_t)m * K + i] *= 12.f;

    uint32_t *dq, *daq;
    uint16_t* ds;
    float *das, *dout, *da;
    CK(hipMalloc(&dq, q.size()));
    CK(hipMalloc(&ds, s.size()));
    CK(hipMalloc(&da, a.size() * 4));
    CK(hipMalloc(&daq, (size_t)M * (K / 2)));
    CK(hipMalloc(&das, (size_t)M * (K / SG) * 4));
    CK(hipMalloc(&dout, (size_t)M * N * 4));
    CK(hipMemcpy(dq, q.data(), q.size(), hipMemcpyHostToDevice));
    CK(hipMemcpy(ds, s.data(), s.size(), hipMemcpyHostToDevice));
    CK(hipMemcpy(da, a.data(), a.size() * 4, hipMemcpyHostToDevice));

    dim3 qgrid(K / SG, M);
    quant_act4<<<qgrid, 32>>>(da, (uint8_t*)daq, das, M, K);
    CK(hipDeviceSynchronize());

    dim3 grid((N + NWARP * ROWS - 1) / (NWARP * ROWS));
    auto launch = [&]() {
        switch (M) {
            case 1: gemv_w4a4<1><<<grid, NTHREADS>>>(dq, ds, daq, das, dout, N, K); break;
            case 2: gemv_w4a4<2><<<grid, NTHREADS>>>(dq, ds, daq, das, dout, N, K); break;
            case 3: gemv_w4a4<3><<<grid, NTHREADS>>>(dq, ds, daq, das, dout, N, K); break;
            case 4: gemv_w4a4<4><<<grid, NTHREADS>>>(dq, ds, daq, das, dout, N, K); break;
            default: printf("M must be 1..4\n"); exit(1);
        }
        CK(hipGetLastError());
    };
    launch();
    CK(hipDeviceSynchronize());
    hipFuncAttributes attr{};
    const void* fn = M == 1 ? (const void*)&gemv_w4a4<1> : M == 2 ? (const void*)&gemv_w4a4<2>
                   : M == 3 ? (const void*)&gemv_w4a4<3> : (const void*)&gemv_w4a4<4>;
    if (hipFuncGetAttributes(&attr, fn) == hipSuccess)
        printf("grid=%u threads=%d  VGPR=%d spill=%zu\n", grid.x, NTHREADS, attr.numRegs,
               (size_t)attr.localSizeBytes);
    {
        int occBlocks = 0;
        if (hipOccupancyMaxActiveBlocksPerMultiprocessor(&occBlocks, fn, NTHREADS, 0) == hipSuccess)
            printf("占用率: %d 个 block/CU = %d 个 wave/CU (%d 线程/block, 每 wave 64 lane)\n",
                   occBlocks, occBlocks * (NTHREADS / 64), NTHREADS);
    }

    hipEvent_t t0, t1;
    CK(hipEventCreate(&t0)); CK(hipEventCreate(&t1));
    CK(hipEventRecord(t0));
    for (int i = 0; i < reps; i++) launch();
    CK(hipEventRecord(t1));
    CK(hipEventSynchronize(t1));
    float ms = 0;
    CK(hipEventElapsedTime(&ms, t0, t1));
    ms /= reps;
    const double gbs = wbytes / (ms * 1e-3) / 1e9;
    printf("W4A4 kernel: %.3f ms -> 权重带宽 %.1f GB/s (%.1f%% of 804)   每步 %.3f ms/行\n",
           ms, gbs, gbs * 100 / 804.0, ms / N * 1000);
    {
        uint32_t* dsink;
        CK(hipMalloc(&dsink, (size_t)N * 4));
        bw_probe4<<<grid, NTHREADS>>>((const uint4*)dq, dsink, N, K / 8 / 4);
        CK(hipDeviceSynchronize());
        CK(hipEventRecord(t0));
        for (int i = 0; i < reps; i++)
            bw_probe4<<<grid, NTHREADS>>>((const uint4*)dq, dsink, N, K / 8 / 4);
        CK(hipEventRecord(t1));
        CK(hipEventSynchronize(t1));
        float pms = 0;
        CK(hipEventElapsedTime(&pms, t0, t1));
        pms /= reps;
        printf("同构纯读探针: %.3f ms -> %.1f GB/s (%.1f%% of 804)\n", pms,
               wbytes / (pms * 1e-3) / 1e9, wbytes / (pms * 1e-3) / 1e9 * 100 / 804);
        hipFree(dsink);
    }

    // 参考：double 精度，用同一批量化的权重/激活（含两处尺度）
    std::vector<float> out((size_t)M * N);
    CK(hipMemcpy(out.data(), dout, out.size() * 4, hipMemcpyDeviceToHost));
    std::vector<uint8_t> aq((size_t)M * (K / 2));
    std::vector<float> as((size_t)M * (K / SG));
    CK(hipMemcpy(aq.data(), daq, aq.size(), hipMemcpyDeviceToHost));
    CK(hipMemcpy(as.data(), das, as.size() * 4, hipMemcpyDeviceToHost));
    const int rows = N < 32 ? N : 32;
    double worst = 0, worst_rel = 0, maxv = 0;
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < rows; n++) {
            double r = 0;
            for (int k = 0; k < K; k++) {
                uint8_t bw = q[(size_t)n * (K / 2) + k / 2];
                int cw = (k & 1) ? (bw >> 4) : (bw & 0xF);
                if (cw >= 8) cw -= 16;
                uint8_t ba = aq[(size_t)m * (K / 2) + k / 2];
                int ca = (k & 1) ? (ba >> 4) : (ba & 0xF);
                if (ca >= 8) ca -= 16;
                uint16_t sh;
                memcpy(&sh, &s[((size_t)n * (K / GRP) + k / GRP) * 2], 2);
                r += (double)(cw * f16f(sh)) * (double)(ca * as[(size_t)m * (K / SG) + k / SG]);
            }
            double got = out[(size_t)m * N + n];
            double err = fabs(got - r), rel = err / (fabs(r) + 1e-6);
            if (err > worst) worst = err;
            if (rel > worst_rel) worst_rel = rel;
            maxv = fmax(maxv, fabs(got));
        }
    }
    printf("正确性(前 32 行): 最大绝对误差 %.3e  最大相对误差 %.3e  幅值 %.3f\n", worst, worst_rel, maxv);

    // 激活量化误差：int4/32 vs int8/128（同一份激活数据，相对 RMS）
    auto act_err = [&](bool int4) {
        double num = 0, den = 0;
        for (int m = 0; m < M; m++)
            for (int k = 0; k < K; k++) {
                float v = a[(size_t)m * K + k];
                den += (double)v * v;
                if (int4) {
                    uint8_t b = aq[(size_t)m * (K / 2) + k / 2];
                    int c = (k & 1) ? (b >> 4) : (b & 0xF);
                    if (c >= 8) c -= 16;
                    double dq = c * as[(size_t)m * (K / SG) + k / SG];
                    num += (v - dq) * (v - dq);
                } else {
                    // int8/128：现场重算（用同一组数据，公平对比）
                    int g = k / 128;
                    float amax = 0;
                    for (int j = g * 128; j < (g + 1) * 128 && j < K; j++)
                        amax = fmaxf(amax, fabsf(a[(size_t)m * K + j]));
                    float ss = amax > 0 ? amax / 127.f : 1.f;
                    int c = (int)lrintf(v / ss);
                    c = c > 127 ? 127 : (c < -128 ? -128 : c);
                    double dq = c * ss;
                    num += (v - dq) * (v - dq);
                }
            }
        return sqrt(num / den);
    };
    printf("激活量化相对 RMS: int4/32 = %.4f   int8/128 = %.4f\n", act_err(true), act_err(false));
    return 0;
}
