// 预填充 int4 FlashAttention（K100_LC / gfx926，自研）
// =============================================================================
// 为什么要有这个文件：见 docs/FORMAT.md 第 7 节。128k 上下文下预填充有一半算力
// 在注意力上，而通用 GEMM 打「K=256 窄 K」的注意力形状只有 18.66 TMAC/s
// （dot8 上限的 24.7%），折算预填充只有 519 t/s。窄 K 让 K 方向的流水摊不开，
// 所以这里改成：**head_dim 256 整个放在内层、Q 块常驻共享内存、外层沿 131072
// 个 key 长流水**，一个 block 一次把一整段 KV 走完（online softmax）。
//
// 结构（每个 block = 一个 q 头 × 64 行 query，沿 key 方向长流水）
// -----------------------------------------------------------------------------
//   QK^T : S[BM,BN] = Q·K^T   —— dot8，Q/K 都 int4，沿 d 每 128 一组尺度
//   软最大: 行最大/行和走 warp shuffle 归约，P = round(7·exp2(S-m_tile))
//   PV   : O += P·V           —— dot8，P/V 都 int4，沿 key 分组（= 一个 tile）
//
// 三个关键设计（都为了一件事：不让 dot8 空转）
// -----------------------------------------------------------------------------
// 1. 共享内存里所有 tile 都按 **K 维优先** 排（KsT[kd][j]，见 int4_gemm2.cpp
//    的教训），线程的 TM/TN 个 dot8 源操作数都是连续 dword → LDS.64/LDS.128，
//    每 dot8 的 LDS 字节数最少（默认配置 QK 2.5 B、PV 1.5 B；TM 越大越低）。
//    另外还做了列序交错（xcol，消 LDS.128 的 4 路 bank 冲突，+13%）。
// 2. Q 块常驻共享；K/V 走「全局→寄存器→STS」两级流水，LDG 的延迟被同一 tile
//    的 QK 计算盖住；每 tile 只有 2 次 __syncthreads()（时序见内核注释）。
// 3. P 的定点尺度取 **本 tile 的行最大**（不是 running max）：tile 内最大值正好
//    映射到 7，不丢动态范围；tile 之间的差以 beta = exp2(m_tile - r_new) ≤ 1
//    折进 PV 的分组尺度，**不多花一条指令**（本来就有一个 (行,列) 尺度乘积）。
//    若改用 running max，后段 tile 的 P 会整体被压成 0——这是 int4 注意力最常见的坑。
//
// 数据格式（KV cache 由本项目自己写，所以按内核最舒服的方式存）
// -----------------------------------------------------------------------------
//   Q : [n_q][KD] u32（dword 内 8 个 d 的 int4，低半字节 = 偶数 d）+ Qsc[g][n_q] f32
//   K : [n_kv][KD] u32（同上，沿 d 分组）+ Ksc[g][n_kv] f32
//   V : [n_kv/8][HD] u32（**转置**：dword 内 8 个 key 的 int4，低半字节 = 偶数 key）
//       + Vsc[tile][HD] f32（沿 key 分组 = 一个 tile）
//   V 之所以要转置存：PV 的收缩维是 key，而 dot8 要求「收缩维就是打包维」，所以 V
//   必须沿 key 打包。把它放到 cache 写入侧（每个 key 写一次）比放在注意力内核里
//   （每个 key 被约 96 个 q 块读到）便宜得多，解码路径同样吃这个布局。
//
// 编译: hipcc -O3 --offload-arch=gfx926 -o flash_attn_int4 flash_attn_int4.hip
// 用法: ./flash_attn_int4 check      # 小规模正确性（对 double 参考）
//       ./flash_attn_int4 bench      # 目标形状吞吐（24 头 / 128k 上下文）
//       ./flash_attn_int4 bench 256  # 指定 q 行数
//
// 实测（2026-09-27，默认 BM=64/BN=64/NT=256）：
//   吞吐 65.20 ms/层 = 25.49 TMAC/s（dot8 上限的 33.8%），VGPR=245 无溢出；
//   正确性 内核 vs 定点算法模型 5.5e-08（短）~ 3.0e-06（128k 深度）。
//   完整记录、消融表与下一步见 docs/FLASH-ATTN.md。
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <functional>
#include <string>
#include <vector>
#include <random>
#include <algorithm>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    printf("HIP ERR %s @%d: %s\n", #x, __LINE__, hipGetErrorString(e_)); exit(1);} } while (0)

typedef uint32_t u32;

#include "flash_attn_core.h"

// ================================ host =======================================

// 值按需生成（本机 host 只有 7GB 内存，不落地大数组）。用一张高斯表 + 哈希取值，
// 再给少量通道乘 12 造离群值（和 bench 里其它程序同一套做法，便于对照）。
static float g_tab[4096];
static void init_tab() {
    std::mt19937 rng(20260926);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (int i = 0; i < 4096; i++) g_tab[i] = nd(rng);
}
static inline float gval(size_t j, size_t d, size_t salt) {
    uint64_t h = (uint64_t)j * 0x9E3779B97F4A7C15ull + ((uint64_t)d + salt) * 0xC2B2AE3D27D4EB4Full;
    h ^= h >> 29; h *= 0xBF58476D1CE4E5B9ull; h ^= h >> 32;
    float v = g_tab[(h >> 7) & 4095];
    if (d % 997 == 0) v *= 12.f;
    return v;
}

static inline int nib(const u32* row, int idx, int e) {
    const int c = (int)((row[idx] >> (4 * e)) & 0xF);
    return c >= 8 ? c - 16 : c;
}

// [rows][HD] f32 → int4（dword 内 8 个 d，低半字节 = 偶数 d）+ 每 QG 一组尺度（组优先）
static void quant_d(int rows, size_t base_j, size_t salt,
                    std::vector<u32>& q, std::vector<float>& sc) {
    q.assign((size_t)rows * KD, 0u);
    sc.assign((size_t)NQG * rows, 0.f);
    std::vector<float> row(HD);
    for (int r = 0; r < rows; r++) {
        for (int d = 0; d < HD; d++) row[d] = gval(base_j + r, d, salt);
        for (int g = 0; g < NQG; g++) {
            const float* s = &row[g * QG];
            float amax = 0.f;
            for (int t = 0; t < QG; t++) amax = fmaxf(amax, fabsf(s[t]));
            const float sca = amax > 0.f ? amax / 7.f : 1.f;
            sc[(size_t)g * rows + r] = sca;
            for (int t = 0; t < QG; t += 8) {
                u32 w = 0;
                for (int e = 0; e < 8; e++) {
                    int c = (int)lrintf(s[t + e] / sca);
                    c = c > 7 ? 7 : (c < -8 ? -8 : c);
                    w |= (u32)(c & 0xF) << (4 * e);
                }
                q[(size_t)r * KD + (g * QG + t) / 8] = w;
            }
        }
    }
}

// V：[n_kv][HD] f32 → 转置打包 [n_kv/8][HD]（dword 内 8 个 key，低半字节 = 偶数 key）
//    + 沿 key 分组的尺度 sc[tile][HD]（tile = BN 个 key，和内核的 tile 对齐）
static void quant_v(int n_kv, size_t salt, std::vector<u32>& vc, std::vector<float>& sc) {
    const int n8 = n_kv / 8, ntl = n_kv / BN;
    vc.assign((size_t)n8 * HD, 0u);
    sc.assign((size_t)ntl * HD, 0.f);
    std::vector<float> blk((size_t)BN * HD);
    for (int t = 0; t < ntl; t++) {
        for (int j = 0; j < BN; j++)
            for (int d = 0; d < HD; d++) blk[(size_t)j * HD + d] = gval((size_t)t * BN + j, d, salt);
        for (int d = 0; d < HD; d++) {
            float amax = 0.f;
            for (int j = 0; j < BN; j++) amax = fmaxf(amax, fabsf(blk[(size_t)j * HD + d]));
            const float sca = amax > 0.f ? amax / 7.f : 1.f;
            sc[(size_t)t * HD + d] = sca;
            for (int j = 0; j < BN; j += 8) {
                u32 w = 0;
                for (int e = 0; e < 8; e++) {
                    int c = (int)lrintf(blk[(size_t)(j + e) * HD + d] / sca);
                    c = c > 7 ? 7 : (c < -8 ? -8 : c);
                    w |= (u32)(c & 0xF) << (4 * e);
                }
                vc[(size_t)((t * BN + j) / 8) * HD + d] = w;
            }
        }
    }
}

// double 参考：softmax(Q·K^T)·V（因果）。Q/K/V 的取值由回调给，于是同一份代码
// 既能算「内核看到的量化值」（分离算法误差），也能算「原始 f32 值」（看总误差）。
static void ref_rows(const std::function<double(int, int)>& qv,
                     const std::function<double(int, int)>& kv,
                     const std::function<double(int, int)>& vv,
                     int n_kv, int off, const std::vector<int>& rows,
                     std::vector<double>& out)
{
    out.assign((size_t)rows.size() * HD, 0.0);
    std::vector<double> kd(HD), s(n_kv), p(n_kv);
    for (size_t r = 0; r < rows.size(); r++) {
        const int gi = rows[r], pos = std::min(off + gi, n_kv - 1);
        for (int j = 0; j <= pos; j++) {
            double acc = 0;
            for (int d = 0; d < HD; d++) acc += qv(gi, d) * kv(j, d);
            s[j] = acc;
        }
        double mx = -1e300, sum = 0;
        for (int j = 0; j <= pos; j++) mx = std::max(mx, s[j]);
        for (int j = 0; j <= pos; j++) { p[j] = exp(s[j] - mx); sum += p[j]; }
        for (int d = 0; d < HD; d++) {
            double acc = 0;
            for (int j = 0; j <= pos; j++) acc += p[j] * vv(j, d);
            out[r * HD + d] = acc / sum;
        }
    }
}

// 「算法模型」参考：把内核的定点算法（每 tile 行最大做 P 尺度、beta 折进 PV、
// online 重标定）用 double 原样重算一遍。内核与它的差 = fp32 舍入；
// 它与量化参考的差 = int4 P 的量化误差（这是算法本身的代价，不是 bug）。
static void ref_alg(const std::function<double(int, int)>& qv,
                    const std::function<double(int, int)>& kv,
                    const std::function<double(int, int)>& vv,
                    int n_kv, int off, const std::vector<int>& rows,
                    std::vector<double>& out)
{
    out.assign((size_t)rows.size() * HD, 0.0);
    const int ntl = n_kv / BN;
    std::vector<double> s(BN), p(BN, 0.0), O(HD);
    for (size_t r = 0; r < rows.size(); r++) {
        const int gi = rows[r], pos = std::min(off + gi, n_kv - 1);
        std::fill(O.begin(), O.end(), 0.0);
        double r_old = -1e300, L = 0;
        for (int t = 0; t < ntl; t++) {
            const int j0 = t * BN, j1 = std::min(j0 + BN - 1, pos);
            if (j1 < j0) break;
            std::fill(p.begin(), p.end(), 0.0);
            double mt = -1e300;
            for (int j = j0; j <= j1; j++) {
                double acc = 0;
                for (int d = 0; d < HD; d++) acc += qv(gi, d) * kv(j, d);
                s[j - j0] = acc;
                mt = std::max(mt, acc);
            }
            double psum = 0;
            for (int j = j0; j <= j1; j++) {
                const double e = round(7.0 * exp(s[j - j0] - mt));   // int4 P
                p[j - j0] = e;
                psum += e;
            }
            const double r_new = std::max(r_old, mt);
            const double alpha = exp(r_old - r_new), beta = exp(mt - r_new) / 7.0;
            for (int d = 0; d < HD; d++) {
                double acc = 0;
                for (int j = j0; j <= j1; j++) acc += p[j - j0] * vv(j, d);
                O[d] = O[d] * alpha + beta * acc;
            }
            L = L * alpha + beta * psum;
            r_old = r_new;
        }
        for (int d = 0; d < HD; d++) out[r * HD + d] = O[d] / L;
    }
}

// 误差报告：相对 RMS（对参考自身的 RMS）与余弦相似度
// 参考是按「抽样行序号」紧凑存的，报告前统一摊成按全局行号索引
static std::vector<double> to_global(const std::vector<double>& v, const std::vector<int>& rows, int n_q) {
    std::vector<double> g((size_t)n_q * HD, 0.0);
    for (size_t r = 0; r < rows.size(); r++)
        std::copy(v.begin() + r * HD, v.begin() + (r + 1) * HD, g.begin() + (size_t)rows[r] * HD);
    return g;
}

template <typename T>
static void report(const char* tag, const std::vector<T>& got, const std::vector<double>& ref,
                   const std::vector<int>& rows)
{
    double num = 0, den = 0, dotv = 0, na = 0, nb = 0, maxd = 0;
    for (size_t r = 0; r < rows.size(); r++) {
        for (int d = 0; d < HD; d++) {
            const double a = ref[(size_t)rows[r] * HD + d], b = got[(size_t)rows[r] * HD + d];
            num += (a - b) * (a - b); den += a * a;
            dotv += a * b; na += a * a; nb += b * b;
            maxd = std::max(maxd, fabs(a - b));
        }
    }
    const double rel = sqrt(num / den);
    const double cos = dotv / sqrt(na * nb + 1e-300);
    printf("  %-22s rel_rms=%.4e  cos=%.9f  max|Δ|=%.4e\n", tag, rel, cos, maxd);
}

int main(int argc, char** argv) {
    init_tab();
    const std::string mode = argc > 1 ? argv[1] : "check";

    hipDeviceProp_t prop;
    CK(hipGetDeviceProperties(&prop, 0));
    printf("device %s | CU=%d | LDS/block=%zu B | wavefront=%d | clock=%d MHz\n",
           prop.name, prop.multiProcessorCount, prop.sharedMemPerBlock, prop.warpSize,
           prop.clockRate / 1000);
    printf("config: BM=%d BN=%d NT=%d  QK映射 %dx%d  PV映射 %dx%d\n", BM, BN, NT, RG, CG, RGO, CGO);

    int n_heads = 1, ratio = 1, n_q = BM, ctx = 320;
    if (mode == "bench") {
        n_heads = 24; ratio = 6;
        n_q = argc > 2 ? atoi(argv[2]) : 1024;
        ctx = 131072;
    } else if (argc > 2) {
        ctx = atoi(argv[2]);
    }
    n_q = ((n_q + BM - 1) / BM) * BM;
    const int n_vh = n_heads / ratio;
    const int n_kv = (((ctx + n_q) + BN - 1) / BN) * BN;
    const int ntl = n_kv / BN;
    const int off = n_kv - n_q;                 // query 的绝对位置 = off + 行号

    printf("%s: n_heads=%d(ratio %d) n_q=%d ctx=%d n_kv=%d tiles=%d off=%d\n",
           mode.c_str(), n_heads, ratio, n_q, ctx, n_kv, ntl, off);

    // ---------------- 量化（每个 head 一段，布局与内核的偏移约定一致） ----------------
    std::vector<u32> hQ, hK, hV;
    std::vector<float> hQs, hKs, hVs;
    {
        std::vector<u32> q; std::vector<float> qs, vs; std::vector<u32> v;
        for (int h = 0; h < n_heads; h++) {
            quant_d(n_q, (size_t)h * n_q, 11, q, qs);
            hQ.insert(hQ.end(), q.begin(), q.end());
            hQs.insert(hQs.end(), qs.begin(), qs.end());
        }
        for (int vh = 0; vh < n_vh; vh++) {
            quant_d(n_kv, (size_t)vh * n_kv, 1000003, q, qs);
            hK.insert(hK.end(), q.begin(), q.end());
            hKs.insert(hKs.end(), qs.begin(), qs.end());
            quant_v(n_kv, 2000003, v, vs);
            hV.insert(hV.end(), v.begin(), v.end());
            hVs.insert(hVs.end(), vs.begin(), vs.end());
        }
    }

    // ---------------- 上卡 ----------------
    u32 *dQ, *dK, *dV; float *dQs, *dKs, *dVs, *dOut;
    CK(hipMalloc(&dQ, hQ.size() * 4));   CK(hipMalloc(&dQs, hQs.size() * 4));
    CK(hipMalloc(&dK, hK.size() * 4));   CK(hipMalloc(&dKs, hKs.size() * 4));
    CK(hipMalloc(&dV, hV.size() * 4));   CK(hipMalloc(&dVs, hVs.size() * 4));
    CK(hipMalloc(&dOut, (size_t)n_heads * n_q * HD * 4));
    CK(hipMemcpy(dQ, hQ.data(), hQ.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(dQs, hQs.data(), hQs.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(dK, hK.data(), hK.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(dKs, hKs.data(), hKs.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(dV, hV.data(), hV.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(dVs, hVs.data(), hVs.size() * 4, hipMemcpyHostToDevice));

    dim3 grid(n_q / BM, n_heads);
    auto launch = [&]() {
        fa_int4<<<grid, NT>>>(dQ, dQs, dK, dKs, dV, dVs, dOut, n_q, n_kv, off, ntl, ratio, n_kv);
        CK(hipGetLastError());
    };
    launch();
    CK(hipDeviceSynchronize());
    {
        hipFuncAttributes attr{};
        if (hipFuncGetAttributes(&attr, (const void*)fa_int4) == hipSuccess)
            printf("kernel: VGPR=%d spill=%zu B\n", attr.numRegs, (size_t)attr.localSizeBytes);
    }

    std::vector<float> out((size_t)n_heads * n_q * HD);
    CK(hipMemcpy(out.data(), dOut, out.size() * 4, hipMemcpyDeviceToHost));
    if (getenv("FA_DEBUG")) {
        int nann = 0; double mx = 0;
        for (size_t i = 0; i < out.size(); i++) {
            if (std::isnan(out[i])) nann++; else mx = std::max(mx, (double)fabs(out[i]));
        }
        printf("debug out: nan=%d max|.|=%.4f\n", nann, mx);
    }

    // ---------------- 正确性：抽几行与 double 参考比 ----------------
    {
        std::vector<int> rows;
        for (int r = 0; r < n_q && rows.size() < 6; r += std::max(1, n_q / 6)) rows.push_back(r);
        if (std::find(rows.begin(), rows.end(), n_q - 1) == rows.end()) rows.push_back(n_q - 1);
        const int h = 0, vh = 0;
        const u32* Q = hQ.data() + (size_t)h * n_q * KD;
        const float* Qs = hQs.data() + (size_t)h * NQG * n_q;
        const u32* K = hK.data() + (size_t)vh * n_kv * KD;
        const float* Ks = hKs.data() + (size_t)vh * NQG * n_kv;
        const u32* V = hV.data() + (size_t)vh * (n_kv / 8) * HD;
        const float* Vs = hVs.data() + (size_t)vh * ntl * HD;

        auto q_quant = [&](int gi, int d) {
            return (double)nib(&Q[(size_t)gi * KD], d / 8, d % 8) * (double)Qs[(size_t)(d / QG) * n_q + gi];
        };
        auto k_quant = [&](int j, int d) {
            return (double)nib(&K[(size_t)j * KD], d / 8, d % 8) * (double)Ks[(size_t)(d / QG) * n_kv + j];
        };
        auto v_quant = [&](int j, int d) {
            return (double)nib(&V[(size_t)(j / 8) * HD], d, j % 8) * (double)Vs[(size_t)(j / BN) * HD + d];
        };
        auto q_src = [&](int gi, int d) { return (double)gval((size_t)h * n_q + gi, d, 11); };
        auto k_src = [&](int j, int d) { return (double)gval((size_t)vh * n_kv + j, d, 1000003); };
        auto v_src = [&](int j, int d) { return (double)gval((size_t)vh * n_kv + j, d, 2000003); };

        std::vector<double> rq0, rf0, ra0, rq, rf, ra;
        ref_rows(q_quant, k_quant, v_quant, n_kv, off, rows, rq0);
        ref_alg (q_quant, k_quant, v_quant, n_kv, off, rows, ra0);
        rq = to_global(rq0, rows, n_q);
        ra = to_global(ra0, rows, n_q);
        if (getenv("FA_DEBUG")) {
            int nann = 0; double mx = 0;
            for (size_t i = 0; i < rq.size(); i++) {
                if (std::isnan(rq[i])) nann++; else mx = std::max(mx, fabs(rq[i]));
            }
            printf("debug ref: nan=%d max|.|=%.4f [0..5]=%.4f %.4f %.4f %.4f %.4f %.4f\n",
                   nann, mx, rq0[0], rq0[1], rq0[2], rq0[3], rq0[4], rq0[5]);
        }
        printf("正确性（head 0，%zu 行 × %d 维）:\n", rows.size(), HD);
        report("内核 vs 算法模型", out, ra, rows);
        report("内核 vs 量化精确参考", out, rq, rows);
        if (n_kv <= 4096) {                      // 大形状下这条太慢，只在 check 里算
            ref_rows(q_src, k_src, v_src, n_kv, off, rows, rf0);
            rf = to_global(rf0, rows, n_q);
            report("内核 vs 原始 f32", out, rf, rows);
            report("算法模型 vs 原始 f32", ra, rf, rows);
        report("算法模型 vs 量化精确", ra, rq, rows);
        if (getenv("FA_DEBUG_ROWS")) {
            printf("逐行 rel_rms（内核/模型/精确三者两两）:\n");
            for (size_t r = 0; r < rows.size(); r++) {
                const int gi = rows[r];
                double ka = 0, kb = 0, kc = 0, kb2 = 0, kc2 = 0, mx = 0;
                for (int d = 0; d < HD; d++) {
                    const double o = out[(size_t)gi * HD + d], m = ra[(size_t)gi * HD + d],
                                 x = rq[(size_t)gi * HD + d];
                    ka += (o - m) * (o - m); kb += (o - x) * (o - x);
                    kc += (m - x) * (m - x); kb2 += x * x; kc2 += x * x;
                    mx = std::max(mx, fabs(x));
                }
                printf("  row %2d: |内核-模型|/rms=%.2e  |内核-精确|/rms=%.2e  |模型-精确|/rms=%.2e  rms|x|=%.3f\n",
                       gi, sqrt(ka / kb2), sqrt(kb / kb2), sqrt(kc / kc2), sqrt(kb2 / HD));
            }
        }
        }
    }

    // ---------------- 吞吐 ----------------
    {
        const int reps = mode == "bench" ? 3 : 5;
        hipEvent_t t0, t1;
        CK(hipEventCreate(&t0)); CK(hipEventCreate(&t1));
        CK(hipEventRecord(t0));
        for (int i = 0; i < reps; i++) launch();
        CK(hipEventRecord(t1));
        CK(hipEventSynchronize(t1));
        float ms = 0; CK(hipEventElapsedTime(&ms, t0, t1)); ms /= reps;

        const double mac_all = (double)n_heads * (n_q / BM) * ntl * BM * BN * HD * 2;
        double mac_causal = 0;                   // 有效（因果）MAC：每行只算到自己的位置
        for (int h = 0; h < n_heads; h++)
            for (int gi = 0; gi < n_q; gi++) mac_causal += (double)(off + gi + 1) * HD * 2;
        const double tm = mac_all / (ms * 1e-3) / 1e12;
        const double tc = mac_causal / (ms * 1e-3) / 1e12;
        const double kvbytes = (double)n_heads / ratio * ntl * (double)BN * (KD * 4 + HD * 4);
        printf("吞吐: %.3f ms/层  →  执行 %.2f TMAC/s (%.1f%% of 75.5)  有效 %.2f TMAC/s\n",
               ms, tm, 100.0 * tm / 75.5, tc);
        printf("      KV 流量 %.1f GB/s（%.0f MB/层）\n",
               kvbytes / (ms * 1e-3) / 1e9, kvbytes / 1e6);
        printf("      折算 16 层预填充 %d token: %.3fs → %.0f t/s（仅注意力部分）\n",
               n_q, ms * 16 / 1000.0, n_q / (ms * 16 / 1000.0));
    }
    return 0;
}
