// 验证批解码注意力（fa_decode_rows_k）自检：一次算 4 行 vs 逐行调用，逐位比较。
//   hipcc -O3 --offload-arch=gfx926 -o /tmp/fa_rows_check fa_rows_check.hip && /tmp/fa_rows_check
// 这是纯算子级自检：输入合成随机数据，不依赖权重。
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <random>
#include <cmath>

#include "../src/k_fa.hip"

#define CHK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    printf("HIP ERR %s @%d: %s\n", #x, __LINE__, hipGetErrorString(e_)); exit(1);} } while (0)

int main(int argc, char** argv) {
    const int H = 24, KVH = 4, ratio = H / KVH, D = HD, KD_ = HD / 8;
    const int kv_cap = 512;
    const int n_kv0 = 130;                     // 第 0 行看到的 key 数（第 r 行 +r）
    const int n_rows = 4;
    const int TP = 64;
    const int max_split = 8;
    const float scq = 0.01f;

    std::mt19937 rng(1234);
    auto rndf = [&](float s) { return std::uniform_real_distribution<float>(-s, s)(rng); };
    std::vector<float> hq((size_t)H * TP * D), hqs((size_t)H * 2 * TP);
    std::vector<uint32_t> hqq((size_t)H * TP * KD_);
    std::vector<float> hk((size_t)KVH * kv_cap * D), hks((size_t)KVH * 2 * kv_cap);
    std::vector<uint32_t> hkc((size_t)KVH * kv_cap * KD_);
    std::vector<uint32_t> hvc((size_t)KVH * (kv_cap / 8) * D);
    std::vector<float> hvs((size_t)KVH * (kv_cap / 64) * D);
    std::vector<float> hsnap((size_t)n_rows * KVH * 64 * D);
    std::vector<float> hpout((size_t)n_rows * H * max_split * D);
    std::vector<float> hpmax((size_t)n_rows * H * max_split), hpsum(hpmax.size());
    // 合成数据：Q/ K 直接给量化后的 dword 与尺度；V 由 fp32 stage 量化得到
    for (auto& v : hq) v = rndf(1.0f);
    for (size_t i = 0; i < hqs.size(); i++) hqs[i] = scq;
    for (size_t i = 0; i < hqq.size(); i++) hqq[i] = (uint32_t)rng();
    for (auto& v : hk) v = rndf(1.0f);
    for (size_t i = 0; i < hks.size(); i++) hks[i] = 0.02f;
    for (size_t i = 0; i < hkc.size(); i++) hkc[i] = (uint32_t)rng();
    for (auto& v : hvs) v = 0.05f;
    for (auto& v : hvc) v = (uint32_t)rng();
    for (auto& v : hsnap) v = rndf(1.5f);
    // 让每行最后一格的 stage 结构与运行时一致：超出本行 prefix 的槽位是别的行的值（真实数据）
    for (int r = 0; r < n_rows; r++) {
        const int tile = (n_kv0 + r - 1) / 64, nval = n_kv0 + r - tile * 64;
        for (int h = 0; h < KVH; h++)
            for (int key = nval; key < 64; key++)
                for (int d = 0; d < D; d++)
                    hsnap[(((size_t)r * KVH + h) * 64 + key) * D + d] = 0.9f;   // 后一行的数据
    }

    uint32_t *dqq, *dkc, *dvc; float *dqs, *dk, *dks, *dvs, *dsnap, *dpout, *dpmax, *dpsum, *dout;
    CHK(hipMalloc(&dqq, hqq.size() * 4)); CHK(hipMalloc(&dqs, hqs.size() * 4));
    CHK(hipMalloc(&dkc, hkc.size() * 4)); CHK(hipMalloc(&dks, hks.size() * 4));
    CHK(hipMalloc(&dvc, hvc.size() * 4)); CHK(hipMalloc(&dvs, hvs.size() * 4));
    CHK(hipMalloc(&dsnap, hsnap.size() * 4)); CHK(hipMalloc(&dpout, hpout.size() * 4));
    CHK(hipMalloc(&dpmax, hpmax.size() * 4)); CHK(hipMalloc(&dpsum, hpsum.size() * 4));
    CHK(hipMalloc(&dout, (size_t)H * TP * D * 4));
    CHK(hipMalloc(&dk, hk.size() * 4)); (void)dk;
    CHK(hipMemcpy(dqq, hqq.data(), hqq.size() * 4, hipMemcpyHostToDevice));
    CHK(hipMemcpy(dqs, hqs.data(), hqs.size() * 4, hipMemcpyHostToDevice));
    CHK(hipMemcpy(dkc, hkc.data(), hkc.size() * 4, hipMemcpyHostToDevice));
    CHK(hipMemcpy(dks, hks.data(), hks.size() * 4, hipMemcpyHostToDevice));
    CHK(hipMemcpy(dvc, hvc.data(), hvc.size() * 4, hipMemcpyHostToDevice));
    CHK(hipMemcpy(dvs, hvs.data(), hvs.size() * 4, hipMemcpyHostToDevice));
    CHK(hipMemcpy(dsnap, hsnap.data(), hsnap.size() * 4, hipMemcpyHostToDevice));

    std::vector<float> out_batch((size_t)H * TP * D, 0.f), out_seq((size_t)H * TP * D, 0.f);
    const size_t vsz = (size_t)KVH * 64 * D;
    const size_t pout_slice = (size_t)H * max_split * D, sc_slice = (size_t)H * max_split;
    // RT_NO_SNAP=1：两边都关掉「按 prefix 重算最后一格 V」的路径，用来定位差异来源
    const bool no_snap = getenv("RT_NO_SNAP") != nullptr;
    float* snapA = no_snap ? nullptr : dsnap;
    float* snapB = no_snap ? nullptr : dsnap;

    // A：逐行（每行单独一次 launch，用各自的暂存切片）
    for (int r = 0; r < n_rows; r++) {
        k_attention_decode_rows(dout + (size_t)r * D, dqq + (size_t)r * KD_, dqs + r, dkc, dks,
                                dvc, dvs, snapA ? snapA + (size_t)r * vsz : nullptr, 0, TP, 1,
                                n_kv0 + r, H, ratio,
                                kv_cap, dpout + (size_t)r * pout_slice,
                                dpmax + (size_t)r * sc_slice, dpsum + (size_t)r * sc_slice,
                                max_split);
    }
    CHK(hipDeviceSynchronize());
    CHK(hipMemcpy(out_seq.data(), dout, out_seq.size() * 4, hipMemcpyDeviceToHost));

    // B：一次算 4 行
    CHK(hipMemset(dpout, 0xCD, hpout.size() * 4));
    CHK(hipMemset(dpmax, 0xCD, hpmax.size() * 4));
    CHK(hipMemset(dpsum, 0xCD, hpsum.size() * 4));
    k_attention_decode_rows(dout, dqq, dqs, dkc, dks, dvc, dvs, snapB, vsz, TP, n_rows, n_kv0,
                            H, ratio, kv_cap, dpout, dpmax, dpsum, max_split);
    CHK(hipDeviceSynchronize());
    CHK(hipMemcpy(out_batch.data(), dout, out_batch.size() * 4, hipMemcpyDeviceToHost));

    std::vector<float> out_batch2((size_t)H * TP * D, 0.f);
    CHK(hipMemset(dout, 0x3F, (size_t)H * TP * D * 4));
    k_attention_decode_rows(dout, dqq, dqs, dkc, dks, dvc, dvs, snapB, vsz, TP, n_rows, n_kv0,
                            H, ratio, kv_cap, dpout, dpmax, dpsum, max_split);
    CHK(hipDeviceSynchronize());
    CHK(hipMemcpy(out_batch2.data(), dout, out_batch2.size() * 4, hipMemcpyDeviceToHost));

    int bad = 0; float mx = 0.f;
    for (int r = 0; r < n_rows; r++) {
        int rb = 0; float rm = 0.f;
        for (int h = 0; h < H; h++)
            for (int t = 0; t < D; t++) {
                const size_t i = ((size_t)h * TP + r) * D + t;
                const float d = fabsf(out_seq[i] - out_batch[i]);
                if (d != 0.f) { bad++; rb++; mx = fmaxf(mx, d); rm = fmaxf(rm, d); }
            }
        printf("  行 %d：不等 %d，max|Δ| = %.6g\n", r, rb, rm);
    }
    int bad2 = 0; float mx2 = 0.f;
    for (int r = 0; r < n_rows; r++)
        for (int h = 0; h < H; h++)
            for (int t = 0; t < D; t++) {
                const size_t i = ((size_t)h * TP + r) * D + t;
                const float d = fabsf(out_batch[i] - out_batch2[i]);
                if (d != 0.f) { bad2++; mx2 = fmaxf(mx2, d); }
            }
    printf("批内两次运行：不等 %d，max|Δ| = %.6g（0 则内核确定）\n", bad2, mx2);
    printf("逐行 vs 一次算 4 行：不等元素 %d / %d，max|Δ| = %.6g\n", bad,
           (int)((size_t)H * TP * D), mx);
    return bad == 0 ? 0 : 1;
}
