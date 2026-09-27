// RT4 解码 GEMV：int4 权重（每 128 组 f16 尺度） × int8 激活。
// 基准支持 M=1..6（运行时只用到 1..4；M=5/6 用来评估加大 MTP 草稿数）
//
// 为什么是这个形状：MTP3 一步要校验 4 个候选 token，而权重只该从显存读一遍。
// 吐字 100 t/s 的账（rt/FORMAT.md 第 6 节）：一步 13.87GB 权重 + 4.30GB KV(@128k)
// + 0.66GB MTP ≈ 18.8GB，804GB/s ⇒ 23.4ms/步 ⇒ 43 步/s ⇒ 需要平均 2.3 token/步。
// 所以本内核的唯一硬指标是**带宽**：把权重按 ≥700GB/s 喂进来。
//
// 算法（避免任何反量化开销）：
//   RT4 里 int4 是有符号的 q∈[-8,7]，取 u = q+8 ∈ [0,15]，
//   Σ q·a = Σ u·a - 8·Σa。u 与 a 都是 int8 合法值，于是内积直接用
//   v_dot4_i32_i8；每 128 组只做一次 -(8·Σa) 修正，Σa 在激活量化时顺带算好。
//   打包顺序（见 rt/FORMAT.md）：低半字节 = 偶数 k。因此一取 4 字节：
//     lo = w & 0x0F0F0F0F  → 4 个偶下标权重（正好占 4 个字节通道）
//     hi = (w>>4) & 0x0F0F0F0F → 4 个奇下标权重
//   激活侧同序拆成 a_even / a_odd 两个数组，按下标与 lo/hi 一一对应，
//   于是每次循环 = 3 次 4 字节加载 + 2 条 dot4（每 token），零 unpack 开销。
//
// 线程组织：一个 warp 一行输出；lane l 负责 dword l（= 8 个权重）。
//   16 个 lane 组成一个量化组（16×8 = 128 个权重）：组内 4 步 shuffle 归约后
//   乘该组尺度再累加。整块 256 线程 = 8 个 warp = 8 行/块。
//
// 编译: hipec -O3 --offload-arch=gfx926 -o gemv_int4 gemv_int4.hip
// 用法: ./gemv_int4 <tensor_name> <M> [rt4_file] [manifest] [reps]
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

#include "gemv_w4a8_core.h"

#ifndef A8_ROWS
#define A8_ROWS 1
#endif

// ---------------------------------------------------------------- manifest ---
// 纯读带宽探针：同样的「每 warp 一行、64 lane × 1 dword」访存模式，不做计算。
// 用来区分「访存模式/并行度不足」和「计算循环开销」。
__global__ void bw_probe(const uint32_t* __restrict__ wq, uint32_t* __restrict__ sink,
                         int N, int K) {
    const int lane = threadIdx.x & 63, warp = threadIdx.x >> 6;
    const int row = blockIdx.x * NWARP + warp;
    if (row >= N) return;
    const int KD = K / 8;
    const uint32_t* wrow = wq + (size_t)row * KD;
    uint32_t acc = 0;
    for (int d0 = 0; d0 < KD; d0 += 64) {
        int d = d0 + lane;
        if (d < KD) acc ^= wrow[d];
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

int main(int argc, char** argv) {
    const char* tname = argc > 1 ? argv[1] : "model.language_model.layers.0.mlp.gate_proj.weight";
    int M = argc > 2 ? atoi(argv[2]) : 1;
    const char* rt4 = argc > 3 ? argv[3] : "/rt/qwen38_27b.rt4";
    const char* man = argc > 4 ? argv[4] : "/rt/qwen38_27b.rt4.json";
    int reps = argc > 5 ? atoi(argv[5]) : 30;

    Ent e;
    if (!find_ent(man, tname, e)) { printf("tensor %s not in manifest\n", tname); return 1; }
    int N = (int)e.N, K = (int)e.K;
    printf("tensor %s  [%d, %d] kind=%s group=%lld relerr=%.4f  M=%d\n",
           tname, N, K, e.kind.c_str(), e.grp, e.relerr, M);
    if (e.kind != "i4") { printf("this kernel only handles i4\n"); return 1; }

    auto q = pread_region(rt4, e.q_off, (long long)N * (K / 2));
    auto s = pread_region(rt4, e.s_off, (long long)N * (K / GRP) * 2);
    double wbytes = (double)q.size();

    // 模拟激活（高斯 + 少量离群通道，接近真实激活分布）
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> a((size_t)M * K);
    for (size_t i = 0; i < a.size(); i++) a[i] = nd(rng);
    for (int m = 0; m < M; m++) for (int i = 0; i < K; i += 997) a[(size_t)m * K + i] *= 12.f;

    uint32_t *dq, *dae, *dao;
    uint16_t* ds;
    float *dasc, *dout, *da;
    int* dasu;
    CK(hipMalloc(&dq, q.size()));
    CK(hipMalloc(&ds, s.size()));
    CK(hipMalloc(&da, a.size() * 4));
    CK(hipMalloc(&dae, (size_t)M * (K / 2)));
    CK(hipMalloc(&dao, (size_t)M * (K / 2)));
    CK(hipMalloc(&dasc, (size_t)M * (K / GRP) * 4));
    CK(hipMalloc(&dasu, (size_t)M * (K / GRP) * 4));
    CK(hipMalloc(&dout, (size_t)M * N * 4));
    CK(hipMemcpy(dq, q.data(), q.size(), hipMemcpyHostToDevice));
    CK(hipMemcpy(ds, s.data(), s.size(), hipMemcpyHostToDevice));
    CK(hipMemcpy(da, a.data(), a.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemset(dae, 0, (size_t)M * (K / 2)));
    CK(hipMemset(dao, 0, (size_t)M * (K / 2)));

    dim3 qgrid(K / GRP, M);
    quant_act<<<qgrid, 32>>>(da, (int8_t*)dae, (int8_t*)dao, dasc, dasu, M, K);
    CK(hipDeviceSynchronize());

    dim3 grid((N + NWARP * A8_ROWS - 1) / (NWARP * A8_ROWS));
    auto launch = [&]() {
        switch (M) {
            case 1: gemv_w4a8<1, false, A8_ROWS><<<grid, NTHREADS>>>(dq, ds, dae, dao, dasc, dasu, dout, N, K); break;
            case 2: gemv_w4a8<2, false, A8_ROWS><<<grid, NTHREADS>>>(dq, ds, dae, dao, dasc, dasu, dout, N, K); break;
            case 3: gemv_w4a8<3, false, A8_ROWS><<<grid, NTHREADS>>>(dq, ds, dae, dao, dasc, dasu, dout, N, K); break;
            case 4: gemv_w4a8<4, false, A8_ROWS><<<grid, NTHREADS>>>(dq, ds, dae, dao, dasc, dasu, dout, N, K); break;
            case 5: gemv_w4a8<5, false, A8_ROWS><<<grid, NTHREADS>>>(dq, ds, dae, dao, dasc, dasu, dout, N, K); break;
            case 6: gemv_w4a8<6, false, A8_ROWS><<<grid, NTHREADS>>>(dq, ds, dae, dao, dasc, dasu, dout, N, K); break;
            default: printf("M must be 1..6\n"); exit(1);
        }
        CK(hipGetLastError());
    };
    launch();
    CK(hipDeviceSynchronize());
    {
        hipFuncAttributes attr{};
        const void* fn = nullptr;
        switch (M) {
            case 1: fn = (const void*)&gemv_w4a8<1, false, A8_ROWS>; break;
            case 2: fn = (const void*)&gemv_w4a8<2, false, A8_ROWS>; break;
            case 3: fn = (const void*)&gemv_w4a8<3, false, A8_ROWS>; break;
            case 4: fn = (const void*)&gemv_w4a8<4, false, A8_ROWS>; break;
            case 5: fn = (const void*)&gemv_w4a8<5, false, A8_ROWS>; break;
            case 6: fn = (const void*)&gemv_w4a8<6, false, A8_ROWS>; break;
        }
        if (hipFuncGetAttributes(&attr, fn) == hipSuccess)
            printf("launch ok: grid=%u threads=%d  VGPR=%d  spill=%d  maxThreads/block=%d\n",
                   grid.x, NTHREADS, attr.numRegs, attr.localSizeBytes, attr.maxThreadsPerBlock);
        else
            printf("launch ok: grid=%u threads=%d (attr 查询失败)\n", grid.x, NTHREADS);
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

    double gbs = wbytes / (ms * 1e-3) / 1e9;
    printf("kernel: %.3f ms/次  ->  权重带宽 %.1f GB/s  (%.1f%% of 804)   等效 %.1f token/s 的解码权重读取\n",
           ms, gbs, gbs * 100.0 / 804.0, gbs * 1000.0 / (wbytes / 1e9) / (1000.0 / ms));

    // 同访存模式的纯读带宽（去掉全部计算）：用来判断瓶颈在访存还是计算
    {
        uint32_t* dsink;
        CK(hipMalloc(&dsink, (size_t)N * 4));
        bw_probe<<<grid, NTHREADS>>>(dq, dsink, N, K);
        CK(hipDeviceSynchronize());
        CK(hipEventRecord(t0));
        for (int i = 0; i < reps; i++) bw_probe<<<grid, NTHREADS>>>(dq, dsink, N, K);
        CK(hipEventRecord(t1));
        CK(hipEventSynchronize(t1));
        float pms = 0;
        CK(hipEventElapsedTime(&pms, t0, t1));
        pms /= reps;
        printf("纯读探针: %.3f ms -> %.1f GB/s (%.1f%% of 804)\n", pms,
               wbytes / (pms * 1e-3) / 1e9, wbytes / (pms * 1e-3) / 1e9 * 100 / 804);
        hipFree(dsink);
    }

    std::vector<float> out((size_t)M * N);
    CK(hipMemcpy(out.data(), dout, out.size() * 4, hipMemcpyDeviceToHost));
    std::vector<float> asc((size_t)M * (K / GRP)), src_a((size_t)M * K);
    std::vector<int> asu((size_t)M * (K / GRP));
    std::vector<int8_t> qe((size_t)M * (K / 2)), qo((size_t)M * (K / 2));
    CK(hipMemcpy(asc.data(), dasc, asc.size() * 4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(asu.data(), dasu, asu.size() * 4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(qe.data(), dae, qe.size(), hipMemcpyDeviceToHost));
    CK(hipMemcpy(qo.data(), dao, qo.size(), hipMemcpyDeviceToHost));
    CK(hipMemcpy(src_a.data(), da, a.size() * 4, hipMemcpyDeviceToHost));

    // CPU 参考：前 32 行，double 精度，用同一套量化值
    int rows = N < 32 ? N : 32;
    double worst = 0, worst_rel = 0, maxv = 0;
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < rows; n++) {
            double acc = 0;
            for (int k = 0; k < K; k++) {
                uint8_t byte = q[(size_t)n * (K / 2) + k / 2];
                int code = (k & 1) ? (byte >> 4) : (byte & 0xF);
                if (code >= 8) code -= 16;
                uint16_t sh;
                memcpy(&sh, &s[((size_t)n * (K / GRP) + k / GRP) * 2], 2);
                float sw = (float)sh;                       // f16 位模式直接用（下面转）
                // f16 → f32
                uint32_t sign = (sh >> 15) & 1, ex = (sh >> 10) & 0x1F, mant = sh & 0x3FF;
                float f = ex == 0 ? ldexpf(mant / 1024.f, -14)
                                  : ldexpf(1.f + mant / 1024.f, (int)ex - 15);
                sw = sign ? -f : f;
                int8_t aq = ((k & 1) ? qo : qe)[(size_t)m * (K / 2) + k / 2];
                acc += (double)(code * sw) * (double)(aq * asc[m * (K / GRP) + k / GRP]);
            }
            double got = out[(size_t)m * N + n];
            double err = fabs(got - acc);
            double rel = err / (fabs(acc) + 1e-6);
            if (err > worst) worst = err;
            if (rel > worst_rel) worst_rel = rel;
            maxv = fmax(maxv, fabs(got));
        }
    }
    printf("正确性(前 32 行, vs double 参考): 最大绝对误差 %.3e  最大相对误差 %.3e  输出幅值 %.3f\n",
           worst, worst_rel, maxv);
    printf("权重读取 %.1f MB, 激活量化尺度样例 %.5f（首个）\n", wbytes / 1e6, asc[0]);

    if (getenv("RT_DBG")) {
        auto rdw32 = [](const void* p) { uint32_t x; memcpy(&x, p, 4); return x; };
        printf("[host] row0 dword0=%08x lo=%08x hi=%08x e=%08x o=%08x\n",
               rdw32(&q[0]), rdw32(&q[0]) & 0x0F0F0F0Fu, (rdw32(&q[0]) >> 4) & 0x0F0F0F0Fu,
               rdw32(&qe[0]), rdw32(&qo[0]));
        auto dot4h = [](uint32_t x, uint32_t y) {
            int s = 0;
            for (int b = 0; b < 4; b++) {
                int xb = (x >> (8 * b)) & 0xFF, yb = (y >> (8 * b)) & 0xFF;
                if (xb >= 128) xb -= 256;
                if (yb >= 128) yb -= 256;
                s += xb * yb;
            }
            return s;
        };
        auto f16h = [](uint16_t sh) {
            uint32_t sg = (sh >> 15) & 1, ex = (sh >> 10) & 0x1F, ma = sh & 0x3FF;
            float f = ex == 0 ? ldexpf(ma / 1024.f, -14) : ldexpf(1.f + ma / 1024.f, (int)ex - 15);
            return sg ? -f : f;
        };
        // row 0：真值参考（逐元素反量化相乘） vs u 形式参考（内核用的算法）
        double true_ref = 0, u_ref = 0;
        for (int k = 0; k < K; k++) {
            uint8_t byte = q[k / 2];
            int code = (k & 1) ? (byte >> 4) : (byte & 0xF);
            if (code >= 8) code -= 16;
            uint16_t sh; memcpy(&sh, &s[(size_t)(k / GRP) * 2], 2);
            int8_t aq = ((k & 1) ? qo : qe)[k / 2];
            true_ref += (double)(code * f16h(sh)) * (double)(aq * asc[k / GRP]);
        }
        for (int g = 0; g < K / GRP; g++) {
            double s_u = 0;
            for (int d = g * 16; d < g * 16 + 16; d++) {
                uint32_t ww = rdw32(&q[4 * d]);
                s_u += dot4h((ww & 0x0F0F0F0Fu) ^ 0x08080808u, rdw32(&qe[4 * d]));
                s_u += dot4h(((ww >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u, rdw32(&qo[4 * d]));
            }
            uint16_t sh; memcpy(&sh, &s[(size_t)g * 2], 2);
            u_ref += (double)f16h(sh) * (double)asc[g] * (s_u - 8.0 * asu[g]);
        }
        printf("[host] row0: 真值参考=%.6f  u形式参考=%.6f  kernel 输出=%.6f\n", true_ref, u_ref, out[0]);
        printf("[host] row1: kernel 输出=%.6f\n", out[1]);
        double int_true = 0, int_u = 0, corr = 0;
        for (int k = 0; k < K; k++) {
            uint8_t byte = q[k / 2];
            int code = (k & 1) ? (byte >> 4) : (byte & 0xF);
            if (code >= 8) code -= 16;
            int8_t aq = ((k & 1) ? qo : qe)[k / 2];
            int_true += (double)code * aq;
        }
        for (int g = 0; g < K / GRP; g++) {
            double s_u = 0;
            for (int d = g * 16; d < g * 16 + 16; d++) {
                uint32_t ww = rdw32(&q[4 * d]);
                s_u += dot4h((ww & 0x0F0F0F0Fu) ^ 0x08080808u, rdw32(&qe[4 * d]));
                s_u += dot4h(((ww >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u, rdw32(&qo[4 * d]));
            }
            int_u += s_u - 8.0 * asu[g];
            corr += 8.0 * asu[g];
        }
        printf("[host] 定点: Σcode·aq=%.0f   Σu·a-8Σa=%.0f   (8Σa=%.0f, su 总和=%.0f)\n",
               int_true, int_u, corr, corr / 8.0);
        printf("[host] row0 dword0: lo=%08x hi=%08x\n", rdw32(&q[0]) & 0x0F0F0F0Fu,
               (rdw32(&q[0]) >> 4) & 0x0F0F0F0Fu);
    }
    return 0;
}
