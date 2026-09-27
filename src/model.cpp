// 自研运行时：RT4 权重 → 全模型前向（Qwen3.8-27B，混合 SSM + 注意力）
//
// 结构（与 HF transformers/models/qwen3_5/modeling_qwen3_5.py 逐行对齐）：
//   每层：x += mixer(rmsnorm(x, input_ln, 零中心))；x += mlp(rmsnorm(x, post_attn_ln))
//   mixer 两种：
//     full_attention（每 4 层一个）：q_proj → [q, gate] 交错 → q/k 各自 RMSNorm(256)
//       → 部分 RoPE(前 64 维) → int4 FlashAttention → * sigmoid(gate) → o_proj
//     linear_attention（其余）：in_proj_qkv → 因果卷积(核 4)+SiLU → q/k 各头 L2 归一
//       → gated delta net 递推 → 门控 RMSNorm(乘 silu(z)) → out_proj
//
// 用法：
//   rt --model <rt4文件> --json <manifest> --ids 1,2,3        # 前向 + 打印 top-k
//   rt --model ... --ids ... --dump out.bin --dump-layers 0,3 # 导出中间激活做对照
#include "kernels.h"
#include "vision_kernels.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <poll.h>
#include <unistd.h>
#include <fstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <chrono>
#include <algorithm>

#define BM_ALIGN 128
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    printf("HIP ERR %s @%d: %s\n", #x, __LINE__, hipGetErrorString(e_)); exit(1);} } while (0)

// =========================== 分阶段计时（RT_PROF=1） ========================
// 用途：回答「这一步到底把时间花在哪」。默认全关（一次 getenv 之外零开销）。
// 实现：每个作用域在进入/退出时各记一个 hipEvent（同一条流上按程序顺序记录），
// **中途不同步**（同步会抽干流水线、把小内核的时间放大好几倍），最后只同步一次，
// 再按「相邻事件时间戳之差」把时间归到当时最内层的作用域。这样得到的是各段
// GPU 真实占用时间，加总是整步墙钟。
enum {
    P_EMBED = 0, P_NORM, P_LINEAR, P_FA, P_KV, P_GDN, P_HEAD, P_COPY, P_SAMPLE,
    P_CONV, P_SSM, P_GDNA, P_GNORM, P_ATTNPRE, P_QK, P_MTP, P_NCAT,
};
static const char* P_NAME[P_NCAT] = {
    "embed", "norm/elem", "linear", "flash-attn", "kv/quant", "gdn", "head", "logits-copy",
    "sample", "gdn-conv", "gdn-ssm", "gdn-recur", "gdn-norm", "attn-pre", "kv-quant", "mtp",
};

#define PROF_POOL 65536
static hipEvent_t g_prof_ev[PROF_POOL];
static int g_prof_next = 0;
static bool g_prof_init = false;
// 时间轴：(类别, 事件)。类别 <0 表示「作用域结束」。
static std::vector<std::pair<int, hipEvent_t>> g_prof_tl;

struct ProfTick {
    int cat = -1;
    bool on = false;
    explicit ProfTick(int c) : cat(c) {
        if (cat < 0) return;
        if (!g_prof_init) { for (int k = 0; k < PROF_POOL; k++) hipEventCreate(&g_prof_ev[k]); g_prof_init = true; }
        if (g_prof_next + 1 >= PROF_POOL) return;          // 池子不够就放弃这段（不会发生）
        on = true;
        const hipEvent_t e = g_prof_ev[g_prof_next++];
        hipEventRecord(e, 0);
        g_prof_tl.push_back({cat, e});
    }
    ~ProfTick() {
        if (!on) return;
        const hipEvent_t e = g_prof_ev[g_prof_next++];
        hipEventRecord(e, 0);
        g_prof_tl.push_back({-1 - cat, e});
    }
};

// 把时间轴折叠成每类目的毫秒数（在最后一个事件处同步一次）。
static void prof_fold(double* acc) {
    if (g_prof_tl.empty()) return;
    hipEventSynchronize(g_prof_tl.back().second);
    std::vector<int> stack;
    for (size_t k = 0; k + 1 < g_prof_tl.size(); k++) {
        const int cat = g_prof_tl[k].first;
        if (cat >= 0) stack.push_back(cat); else if (!stack.empty()) stack.pop_back();
        if (stack.empty()) continue;
        float ms = 0.f;
        if (hipEventElapsedTime(&ms, g_prof_tl[k].second, g_prof_tl[k + 1].second) == hipSuccess)
            acc[stack.back()] += ms;
    }
    g_prof_tl.clear();
    g_prof_next = 0;
}


// ============================== 配置 =======================================
struct Cfg {
    int hidden = 5120, n_layer = 64, n_head = 24, n_kv = 4, head_dim = 256;
    int inter = 17408, vocab = 248320;
    int lk_head = 16, lv_head = 48, ldim = 128, conv_k = 4;
    float eps = 1e-6f, rope_theta = 1e7f;
    int rot = 64;                    // partial_rotary_factor 0.25 × 256
    int full_interval = 4;
    bool is_full(int il) const { return (il + 1) % full_interval == 0; }
};

// ============================== RT4 加载 ===================================
struct RT4Tensor {
    std::string name, kind;
    long long N, K, group, q_off, s_off, nbytes;
};
struct RT4 {
    // 本机宿主只有 ~7GB 内存，13.9GB 的权重必须 mmap（虚拟地址），不能整份读进来
    const uint8_t* map = nullptr;
    size_t map_size = 0;
    std::vector<RT4Tensor> tensors;
    std::map<std::string, int> index;
    uint8_t* dev = nullptr;                      // 设备上的整份权重

    const RT4Tensor* find(const std::string& n) const {
        auto it = index.find(n);
        return it == index.end() ? nullptr : &tensors[it->second];
    }
    void load(const std::string& path, const std::string& json);
    void upload();
    const u32* qptr(const RT4Tensor* t) const { return (const u32*)(dev + t->q_off); }
    const float* sptr(const RT4Tensor* t) const { return (const float*)(dev + t->s_off); }
    const uint8_t* hptr(long long off) const { return map + off; }
};

static std::string read_file(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) { printf("无法打开 %s\n", p.c_str()); exit(1); }
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

void RT4::load(const std::string& path, const std::string& json) {
    const std::string js = read_file(json);
    size_t p = 0;
    while ((p = js.find("\"name\": \"", p)) != std::string::npos) {
        const size_t q0 = p + 9, q1 = js.find('"', q0);
        const size_t obj_end = js.find('}', q1);
        const std::string obj = js.substr(q1, obj_end - q1);
        RT4Tensor t;
        t.name = js.substr(q0, q1 - q0);
        auto num = [&](const char* key, long long def) -> long long {
            size_t k = obj.find(key);
            return k == std::string::npos ? def : atoll(obj.c_str() + k + strlen(key));
        };
        size_t k = obj.find("\"kind\": \"");
        if (k != std::string::npos) t.kind = obj.substr(k + 9, obj.find('"', k + 9) - k - 9);
        t.N = num("\"shape\": [", 0);
        size_t comma = obj.find(',', obj.find("\"shape\": ["));
        t.K = comma == std::string::npos ? 1 : atoll(obj.c_str() + comma + 1);
        t.group = num("\"group\": ", 128);
        t.q_off = num("\"q_off\": ", 0);
        t.s_off = num("\"s_off\": ", 0);
        t.nbytes = num("\"nbytes\": ", 0);
        index[t.name] = (int)tensors.size();
        tensors.push_back(t);
        p = obj_end;
    }
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) { printf("无法打开 %s\n", path.c_str()); exit(1); }
    struct stat st{};
    fstat(fd, &st);
    map_size = (size_t)st.st_size;
    map = (const uint8_t*)mmap(nullptr, map_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) { printf("mmap 失败 %s\n", path.c_str()); exit(1); }
    printf("RT4: %zu 张量, %.2f GB (mmap)\n", tensors.size(), map_size / 1e9);
}

void RT4::upload() {
    CK(hipMalloc(&dev, map_size));
    // 分块拷贝：一整块 13.9GB 的 hipMemcpy 从 pageable mmap 出发容易失败/很慢
    const size_t CH = 256ull << 20;
    for (size_t off = 0; off < map_size; off += CH) {
        const size_t n = std::min(CH, map_size - off);
        CK(hipMemcpy(dev + off, map + off, n, hipMemcpyHostToDevice));
    }
}

// ========================= 组优先尺度的转换 ================================
// RT4 的 i4 尺度是行优先 s[n*(K/G)+g]；int4 GEMM 要组优先 s[g*N+n]。
// 只给走 GEMM 路径的权重额外准备一份（尺度总量 ~0.2GB）。
static std::vector<float> make_group_major(const RT4& rt, const RT4Tensor* t) {
    const int ng = (int)(t->K / t->group);
    std::vector<float> out((size_t)t->N * ng);
    const uint16_t* s = (const uint16_t*)rt.hptr(t->s_off);
    for (int n = 0; n < t->N; n++)
        for (int g = 0; g < ng; g++) {
            const uint16_t h = s[(size_t)n * ng + g];
            const uint32_t sg = (h >> 15) & 1, ex = (h >> 10) & 0x1F, ma = h & 0x3FF;
            float f = ex == 0 ? ldexpf(ma / 1024.f, -14) : ldexpf(1.f + ma / 1024.f, (int)ex - 15);
            out[(size_t)g * t->N + n] = sg ? -f : f;
        }
    return out;
}

// ============================== 权重视图 ===================================
struct Wq {                       // 一个 int4 权重矩阵
    const u32* q = nullptr;
    const float* s = nullptr;     // 行优先（GEMV）
    float* s_gm = nullptr;        // 组优先（GEMM，设备指针）
    int N = 0, K = 0, group = 128;
};
struct Wf { const void* p = nullptr; int n = 0; bool is_f16 = false; };

// ============================== 视觉塔（RT4） ==============================
// 权重来自 tools/convert_vision_rt4.py：线性层 int4/128（N 补 64、K 补 128），
// norm/bias/pos_embed 是 f32。线性层直接复用文本运行时的 int4 GEMM。
struct VisionModel {
    struct VW {
        const uint16_t* w = nullptr;      // f16 [N,K]
        int N = 0, K = 0;
    };
    struct VLayer {
        VW qkv, proj, fc1, fc2;
        const float *n1w = nullptr, *n1b = nullptr;
        const float *n2w = nullptr, *n2b = nullptr;
        const float *qkvb = nullptr, *projb = nullptr;
        const float *fc1b = nullptr, *fc2b = nullptr;
    };

    static constexpr int H = 1152, HEADS = 16, HD = 72;
    static constexpr int QKV = 3456, INTER = 4304, INTER_PAD = 4352;
    static constexpr int MERGE_IN = 4608, OUT_H = 5120;
    static constexpr int PATCH_DIM = 3 * 2 * 16 * 16;
    static constexpr int POS_N = 2304, POS_SIDE = 48, DEPTH = 27;

    RT4 rt;
    std::vector<VLayer> layers;
    VW patch_w, m_fc1, m_fc2;
    const float *patch_b = nullptr, *m_nw = nullptr, *m_nb = nullptr;
    const float *m_fc1b = nullptr, *m_fc2b = nullptr;
    const float* pos_host = nullptr;
    bool loaded = false;
    int max_patches = 0, Mpad = 0, MMpad = 0;
    std::vector<float*> dev_f32;

    float *d_patch = nullptr, *d_h = nullptr, *d_norm = nullptr, *d_qkv = nullptr;
    float *d_attn = nullptr, *d_mlp = nullptr, *d_tmp = nullptr, *d_pos = nullptr;
    float *d_cos = nullptr, *d_sin = nullptr, *d_merger = nullptr, *d_merger2 = nullptr;
    float *d_out = nullptr;

    static int align128(int n) { return (n + 127) / 128 * 128; }

    VW W(const std::string& name) {
        const RT4Tensor* t = rt.find(name);
        if (!t || t->kind != "f16") {
            printf("视觉权重缺失或不是 f16：%s\n", name.c_str());
            exit(1);
        }
        VW w;
        w.w = (const uint16_t*)(rt.dev + t->q_off);
        w.N = (int)t->N;
        w.K = (int)t->K;
        return w;
    }

    const float* F(const std::string& name) {
        const RT4Tensor* t = rt.find(name);
        if (!t || t->kind != "f32") {
            printf("视觉张量缺失或不是 f32：%s\n", name.c_str());
            exit(1);
        }
        float* dev = nullptr;
        CK(hipMalloc(&dev, t->nbytes));
        CK(hipMemcpy(dev, rt.hptr(t->q_off), t->nbytes, hipMemcpyHostToDevice));
        dev_f32.push_back(dev);
        return dev;
    }

    void init(const std::string& path, const std::string& json) {
        rt.load(path, json);
        rt.upload();
        max_patches = getenv("RT_VISION_MAX_PATCHES")
                          ? atoi(getenv("RT_VISION_MAX_PATCHES")) : 4096;
        Mpad = align128(max_patches);
        MMpad = align128(max_patches / 4);
        layers.resize(DEPTH);
        patch_w = W("model.visual.patch_embed.proj.weight");
        patch_b = F("model.visual.patch_embed.proj.bias");
        const RT4Tensor* pe = rt.find("model.visual.pos_embed.weight");
        if (!pe || pe->kind != "f32" || pe->N != POS_N || pe->K != H) {
            printf("pos_embed 形状不对\n"); exit(1);
        }
        pos_host = (const float*)rt.hptr(pe->q_off);
        for (int il = 0; il < DEPTH; il++) {
            auto n = [&](const char* s) {
                char b[256]; snprintf(b, sizeof(b), "model.visual.blocks.%d.%s", il, s);
                return std::string(b);
            };
            VLayer& L = layers[il];
            L.qkv = W(n("attn.qkv.weight"));
            L.proj = W(n("attn.proj.weight"));
            L.fc1 = W(n("mlp.linear_fc1.weight"));
            L.fc2 = W(n("mlp.linear_fc2.weight"));
            L.n1w = F(n("norm1.weight")); L.n1b = F(n("norm1.bias"));
            L.n2w = F(n("norm2.weight")); L.n2b = F(n("norm2.bias"));
            L.qkvb = F(n("attn.qkv.bias")); L.projb = F(n("attn.proj.bias"));
            L.fc1b = F(n("mlp.linear_fc1.bias")); L.fc2b = F(n("mlp.linear_fc2.bias"));
        }
        m_fc1 = W("model.visual.merger.linear_fc1.weight");
        m_fc2 = W("model.visual.merger.linear_fc2.weight");
        m_nw = F("model.visual.merger.norm.weight");
        m_nb = F("model.visual.merger.norm.bias");
        m_fc1b = F("model.visual.merger.linear_fc1.bias");
        m_fc2b = F("model.visual.merger.linear_fc2.bias");

        CK(hipMalloc(&d_patch, (size_t)Mpad * PATCH_DIM * 4));
        CK(hipMalloc(&d_h, (size_t)Mpad * H * 4));
        CK(hipMalloc(&d_norm, (size_t)Mpad * H * 4));
        CK(hipMalloc(&d_qkv, (size_t)Mpad * QKV * 4));
        CK(hipMalloc(&d_attn, (size_t)Mpad * H * 4));
        CK(hipMalloc(&d_mlp, (size_t)Mpad * INTER_PAD * 4));
        CK(hipMalloc(&d_tmp, (size_t)Mpad * H * 4));
        CK(hipMalloc(&d_pos, (size_t)Mpad * H * 4));
        CK(hipMalloc(&d_cos, (size_t)Mpad * HD * 4));
        CK(hipMalloc(&d_sin, (size_t)Mpad * HD * 4));
        CK(hipMalloc(&d_merger, (size_t)MMpad * MERGE_IN * 4));
        CK(hipMalloc(&d_merger2, (size_t)MMpad * MERGE_IN * 4));
        CK(hipMalloc(&d_out, (size_t)MMpad * OUT_H * 4));
        CK(hipMemset(d_patch, 0, (size_t)Mpad * PATCH_DIM * 4));
        CK(hipMemset(d_attn, 0, (size_t)Mpad * H * 4));
        CK(hipMemset(d_mlp, 0, (size_t)Mpad * INTER_PAD * 4));
        CK(hipMemset(d_merger, 0, (size_t)MMpad * MERGE_IN * 4));
        loaded = true;
        printf("视觉塔 RT4 就绪：%d 层，max_patches=%d\n", DEPTH, max_patches);
    }

    void linear(float* y, const VW& w, const float* x, int M) {
        linear_s(y, w, x, M, w.K);
    }

    void linear_s(float* y, const VW& w, const float* x, int M, int x_stride) {
        k_vit_linear_f16(y, w.w, x, M, w.N, w.K, x_stride, w.N);
    }

    void dump_dev(const char* tag, const float* p, int rows, int D) {
        if (!getenv("RT_VISION_DUMP")) return;
        std::vector<float> h((size_t)rows * D);
        CK(hipMemcpy(h.data(), p, h.size() * 4, hipMemcpyDeviceToHost));
        char path[256];
        snprintf(path, sizeof(path), "/tmp/vit_%s.f32", tag);
        FILE* f = fopen(path, "wb");
        if (!f) return;
        fwrite(h.data(), 4, h.size(), f);
        fclose(f);
        fprintf(stderr, "  dump %s [%d,%d]\n", path, rows, D);
    }

    static float lin_value(int i, int n) {
        return n <= 1 ? 0.f : (float)i * (POS_SIDE - 1) / (n - 1);
    }

    void make_pos(int h, int w, int n, std::vector<float>& out) {
        const int gh = h / 2, gw = w / 2;
        out.resize((size_t)n * H);
        for (int idx = 0; idx < n; idx++) {
            int rem = idx;
            const int mw = rem % 2; rem /= 2;
            const int mh = rem % 2; rem /= 2;
            const int gc = rem % gw; rem /= gw;
            const int gr = rem % gh;
            const int row = gr * 2 + mh;
            const int col = gc * 2 + mw;
            const float hf = lin_value(row, h);
            const float wf = lin_value(col, w);
            int h0 = (int)hf, w0 = (int)wf;
            int h1 = std::min(h0 + 1, POS_SIDE - 1), w1 = std::min(w0 + 1, POS_SIDE - 1);
            const float dh = hf - h0, dw = wf - w0;
            const float* p00 = pos_host + ((size_t)h0 * POS_SIDE + w0) * H;
            const float* p01 = pos_host + ((size_t)h0 * POS_SIDE + w1) * H;
            const float* p10 = pos_host + ((size_t)h1 * POS_SIDE + w0) * H;
            const float* p11 = pos_host + ((size_t)h1 * POS_SIDE + w1) * H;
            float* o = out.data() + (size_t)idx * H;
            for (int d = 0; d < H; d++)
                o[d] = (1 - dh) * (1 - dw) * p00[d] + (1 - dh) * dw * p01[d] +
                       dh * (1 - dw) * p10[d] + dh * dw * p11[d];
        }
    }

    void make_rope(int h, int w, int n, std::vector<float>& cos, std::vector<float>& sin) {
        const int maxhw = std::max(h, w);
        const int nf = 18;
        std::vector<float> inv(nf);
        for (int i = 0; i < nf; i++) inv[i] = 1.f / powf(10000.f, (2.f * i) / 36.f);
        cos.resize((size_t)n * HD);
        sin.resize((size_t)n * HD);
        std::vector<float> ft((size_t)maxhw * nf);
        for (int j = 0; j < maxhw; j++)
            for (int i = 0; i < nf; i++) ft[(size_t)j * nf + i] = j * inv[i];
        for (int idx = 0; idx < n; idx++) {
            int rem = idx;
            const int mw = rem % 2; rem /= 2;
            const int mh = rem % 2; rem /= 2;
            const int gc = rem % (w / 2); rem /= (w / 2);
            const int gr = rem;
            const int row = gr * 2 + mh, col = gc * 2 + mw;
            float* c = cos.data() + (size_t)idx * HD;
            float* s = sin.data() + (size_t)idx * HD;
            for (int i = 0; i < nf; i++) {
                const float a = ft[(size_t)row * nf + i];
                const float b = ft[(size_t)col * nf + i];
                c[i] = cosf(a); s[i] = sinf(a);
                c[nf + i] = cosf(b); s[nf + i] = sinf(b);
                c[2 * nf + i] = c[i]; s[2 * nf + i] = s[i];
                c[3 * nf + i] = c[nf + i]; s[3 * nf + i] = s[nf + i];
            }
        }
    }

    bool encode(const float* patches, int n, std::vector<float>& out, int& n_tokens,
                int gh, int gw) {
        if (!loaded || n <= 0 || n % 4 != 0) {
            printf("vision: 非法 patch 数 %d\n", n); return false;
        }
        if (n > max_patches) {
            printf("vision: patch 数 %d > max_patches %d\n", n, max_patches); return false;
        }
        const int Mp = align128(n);
        CK(hipMemset(d_patch, 0, (size_t)Mpad * PATCH_DIM * 4));
        CK(hipMemcpy(d_patch, patches, (size_t)n * PATCH_DIM * 4, hipMemcpyHostToDevice));
        linear(d_h, patch_w, d_patch, n);
        k_vit_bias_add(d_h, patch_b, n, H);
        std::vector<float> pos;
        make_pos(gh, gw, n, pos);
        CK(hipMemcpy(d_pos, pos.data(), (size_t)n * H * 4, hipMemcpyHostToDevice));
        k_add_inplace(d_h, d_pos, (long long)n * H);
        dump_dev("h00", d_h, n, H);
        std::vector<float> cos, sin;
        make_rope(gh, gw, n, cos, sin);
        CK(hipMemcpy(d_cos, cos.data(), (size_t)n * HD * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(d_sin, sin.data(), (size_t)n * HD * 4, hipMemcpyHostToDevice));
        for (int il = 0; il < DEPTH; il++) {
            VLayer& L = layers[il];
            k_vit_layernorm(d_norm, d_h, L.n1w, L.n1b, n, H, 1e-6f);
            if (il == 0) dump_dev("l0_norm1", d_norm, n, H);
            linear(d_qkv, L.qkv, d_norm, n);
            k_vit_bias_add(d_qkv, L.qkvb, n, QKV);
            if (il == 0) dump_dev("l0_qkv", d_qkv, n, QKV);
            k_vit_rope(d_qkv, d_cos, d_sin, n, HEADS, HD, QKV);
            if (il == 0) dump_dev("l0_qkv_rope", d_qkv, n, QKV);
            k_vit_attention(d_attn, d_qkv, n, HEADS, HD, QKV, 1.f / sqrtf(72.f));
            if (il == 0) dump_dev("l0_attn", d_attn, n, H);
            linear(d_tmp, L.proj, d_attn, n);
            k_vit_bias_add(d_tmp, L.projb, n, H);
            k_add_inplace(d_h, d_tmp, (long long)n * H);
            if (il == 0) dump_dev("l0_res", d_h, n, H);
            k_vit_layernorm(d_norm, d_h, L.n2w, L.n2b, n, H, 1e-6f);
            if (il == 0) dump_dev("l0_norm2", d_norm, n, H);
            linear(d_mlp, L.fc1, d_norm, n);                 // [n, 4352]
            if (il == 0) dump_dev("l0_fc1", d_mlp, n, INTER_PAD);
            k_vit_bias_add_s(d_mlp, L.fc1b, n, INTER, INTER_PAD);  // 只加逻辑 4304 列
            k_vit_gelu(d_mlp, d_mlp, (long long)n * INTER_PAD, 0);
            if (il == 0) dump_dev("l0_gelu", d_mlp, n, INTER_PAD);
            linear_s(d_tmp, L.fc2, d_mlp, n, INTER_PAD);
            k_vit_bias_add(d_tmp, L.fc2b, n, H);
            k_add_inplace(d_h, d_tmp, (long long)n * H);
            if (il == 0) dump_dev("h01", d_h, n, H);
            if (il == 1) dump_dev("h02", d_h, n, H);
            if ((il + 1) % 9 == 0)
                fprintf(stderr, "  视觉层 %d/%d\n", il + 1, DEPTH);
        }
        k_vit_layernorm(d_norm, d_h, m_nw, m_nb, n, H, 1e-6f);
        n_tokens = n / 4;
        const int MMp = align128(n_tokens);
        CK(hipMemset(d_merger, 0, (size_t)MMpad * MERGE_IN * 4));
        CK(hipMemcpy(d_merger, d_norm, (size_t)n * H * 4, hipMemcpyDeviceToDevice));
        linear(d_merger2, m_fc1, d_merger, n_tokens);
        k_vit_bias_add(d_merger2, m_fc1b, n_tokens, MERGE_IN);
        k_vit_gelu(d_merger2, d_merger2, (long long)n_tokens * MERGE_IN, 1);
        linear(d_out, m_fc2, d_merger2, n_tokens);
        k_vit_bias_add(d_out, m_fc2b, n_tokens, OUT_H);
        out.resize((size_t)n_tokens * OUT_H);
        CK(hipMemcpy(out.data(), d_out, out.size() * 4, hipMemcpyDeviceToHost));
        dump_dev("hemb", d_out, n_tokens, OUT_H);
        return true;
    }

    bool encode_file(const std::string& patch_path, const std::string& out_path,
                     int& n_tokens, int gh, int gw) {
        std::ifstream f(patch_path, std::ios::binary);
        if (!f) { printf("无法打开 patch 文件 %s\n", patch_path.c_str()); return false; }
        f.seekg(0, std::ios::end);
        const size_t bytes = (size_t)f.tellg();
        f.seekg(0, std::ios::beg);
        if (bytes == 0 || bytes % (PATCH_DIM * 4)) {
            printf("patch 文件大小不对 %zu\n", bytes); return false;
        }
        const int n = (int)(bytes / (PATCH_DIM * 4));
        std::vector<float> patches(bytes / 4);
        f.read((char*)patches.data(), bytes);
        std::vector<float> out;
        if (!encode(patches.data(), n, out, n_tokens, gh, gw)) return false;
        std::ofstream o(out_path, std::ios::binary);
        if (!o) { printf("无法写 embedding 文件 %s\n", out_path.c_str()); return false; }
        o.write((const char*)out.data(), out.size() * 4);
        return true;
    }
};

struct Layer {
    bool full = false;
    Wf in_ln, post_ln;
    Wq mlp_gate, mlp_up, mlp_down;
    // full attention
    Wq q_proj, k_proj, v_proj, o_proj;
    Wf q_norm, k_norm;
    // linear attention
    Wq in_qkv, in_z, out_proj;
    Wf conv1d, dt_bias, a_log, ssm_norm;
    Wf ssm_alpha, ssm_beta;         // [48][5120] f16：太小不量化，加载时转 f32
};

// MTP 头（Qwen3.5 NextN）：[embedding | hidden] → fc → 一层标准全注意力 + MLP → norm
struct Mtp {
    Wq fc;
    Wf pre_norm_embedding, pre_norm_hidden, norm;
    Wf in_ln, post_ln, q_norm, k_norm;
    Wq q_proj, k_proj, v_proj, o_proj;
    Wq mlp_gate, mlp_up, mlp_down;
    bool loaded = false;
};

struct Dev {
    float *x = nullptr, *xb = nullptr;            // 残差流 / norm 输出
    float *m_gate = nullptr, *m_up = nullptr;     // MLP 中间 (17408)
    float *qfull = nullptr;                       // q_proj 输出 [T][12288]
    float *hq = nullptr, *hgate = nullptr, *hout = nullptr;   // [T][6144]
    float *hkk = nullptr, *hvv = nullptr;         // [T][1024]
    float *hfa = nullptr;                         // FA 输出 [24][TP][256]
    float *qkv3 = nullptr, *conv = nullptr, *gz = nullptr;    // GDN: [T][10240]/[T][6144]
    float *gq = nullptr, *gk = nullptr, *gv = nullptr;        // GDN 紧凑 q/k/v
    float *gab = nullptr, *gg = nullptr;          // a/b/gate [T][96]
    float *gbeta = nullptr;                       // sigmoid(b) [T][48]
    float *logits = nullptr;
    u32   *qq = nullptr;                          // 量化后的 Q [24][TP][32]
    float *qs = nullptr;                          // 其尺度 [24][2][TP]
    float *conv_prev = nullptr;                   // 卷积状态快照（更新前）
    u32   *aq_big = nullptr;                      // GEMM 激活量化暂存
    float *asc_big = nullptr;
    u32   *aq_h = nullptr, *aq_l = nullptr;       // W4A8：两个 int4 数字
    float *asc8 = nullptr, *asc16 = nullptr;      // 对应的两组尺度（sc16 = 16*sc）
    float *c_tmp = nullptr;                       // 第二遍 GEMM 的输出（之后相加）
    int   *dids = nullptr;                        // 输入 token id（常驻，避免每步 malloc）
    // 解码注意力的 split-KV 暂存：24 头 × 最多 8 段（验证批按 4 行分片）
    float *fa_pout = nullptr, *fa_pmax = nullptr, *fa_psum = nullptr;
    // 主模型每行的最终 norm 隐藏态（MTP 的 hidden 输入）与上一 token 的隐藏态
    float *h_norm = nullptr, *h_prev = nullptr;
    // MTP 草稿 / 预填充用的临时激活
    float *mtp_e = nullptr, *mtp_hin = nullptr, *mtp_hn = nullptr, *mtp_cat = nullptr;
    float *mtp_fc = nullptr, *mtp_tmp = nullptr, *mtp_out = nullptr;
    float *mtp_logits = nullptr;
    // MTP 验证批的逐 token 状态快照（GDN / 卷积），以及逐行 logits
    float *ssm_snap = nullptr, *conv_snap = nullptr;
    // 主模型 / MTP 的 V tile 暂存快照（V 的尺度按 64-key tile 共享，跨 tile 回滚必须恢复）
    float *vstage_snap = nullptr, *mtp_vstage_snap = nullptr;
    float *logits_all = nullptr;
    int   *argmax = nullptr;
    int   *mtp_dids = nullptr;
    void alloc(int T);
};
void Dev::alloc(int T) {
    const size_t H = 5120, F = 17408, C6 = 6144, C10 = 10240;
    const int TP = ((T + 63) / 64) * 64;
    T = ((T + 127) / 128) * 128;                 // GEMM 的 M 要按 BM=128 对齐
    CK(hipMalloc(&x, (size_t)T * H * 4));
    CK(hipMalloc(&xb, (size_t)T * H * 4));
    CK(hipMalloc(&m_gate, (size_t)T * F * 4));
    CK(hipMalloc(&m_up, (size_t)T * F * 4));
    CK(hipMalloc(&qfull, (size_t)T * 12288 * 4));
    CK(hipMalloc(&hq, (size_t)T * C6 * 4));
    CK(hipMalloc(&hgate, (size_t)T * C6 * 4));
    CK(hipMalloc(&hout, (size_t)T * C6 * 4));
    CK(hipMalloc(&hkk, (size_t)T * 1024 * 4));
    CK(hipMalloc(&hvv, (size_t)T * 1024 * 4));
    CK(hipMalloc(&hfa, (size_t)24 * TP * 256 * 4));
    CK(hipMalloc(&qkv3, (size_t)T * C10 * 4));
    CK(hipMalloc(&conv, (size_t)T * C10 * 4));
    CK(hipMalloc(&gz, (size_t)T * C6 * 4));
    CK(hipMalloc(&gq, (size_t)T * 2048 * 4));
    CK(hipMalloc(&gk, (size_t)T * 2048 * 4));
    CK(hipMalloc(&gv, (size_t)T * C6 * 4));
    CK(hipMalloc(&gab, (size_t)T * 96 * 4));
    CK(hipMalloc(&gg, (size_t)T * 48 * 4));
    CK(hipMalloc(&gbeta, (size_t)T * 48 * 4));
    CK(hipMalloc(&logits, (size_t)248320 * 4));
    CK(hipMalloc(&qq, (size_t)24 * TP * 32 * 4));
    CK(hipMalloc(&qs, (size_t)24 * 2 * TP * 4));
    CK(hipMalloc(&conv_prev, (size_t)3 * C10 * 4));
    CK(hipMalloc(&aq_big, (size_t)T * F / 2 + 64));
    CK(hipMalloc(&asc_big, (size_t)T * (F / 128) * 4 + 64));
    CK(hipMalloc(&aq_h, (size_t)T * F / 2 + 64));
    CK(hipMalloc(&aq_l, (size_t)T * F / 2 + 64));
    CK(hipMalloc(&asc8, (size_t)T * (F / 128) * 4 + 64));
    CK(hipMalloc(&asc16, (size_t)T * (F / 128) * 4 + 64));
    CK(hipMalloc(&c_tmp, (size_t)T * F * 4));
    CK(hipMalloc(&dids, (size_t)T * 4 + 64));
    CK(hipMalloc(&fa_pout, (size_t)4 * 24 * 8 * 256 * 4));
    CK(hipMalloc(&fa_pmax, (size_t)4 * 24 * 8 * 4));
    CK(hipMalloc(&fa_psum, (size_t)4 * 24 * 8 * 4));
    CK(hipMalloc(&h_norm, (size_t)T * H * 4));
    CK(hipMalloc(&h_prev, (size_t)H * 4));
    CK(hipMalloc(&mtp_e, (size_t)T * H * 4));
    CK(hipMalloc(&mtp_hin, (size_t)T * H * 4));
    CK(hipMalloc(&mtp_hn, (size_t)T * H * 4));
    CK(hipMalloc(&mtp_cat, (size_t)T * 2 * H * 4));
    CK(hipMalloc(&mtp_fc, (size_t)T * H * 4));
    CK(hipMalloc(&mtp_tmp, (size_t)T * H * 4));
    CK(hipMalloc(&mtp_out, (size_t)T * H * 4));
    CK(hipMalloc(&mtp_logits, (size_t)248320 * 4));
    CK(hipMalloc(&logits_all, (size_t)4 * 248320 * 4));
    CK(hipMalloc(&argmax, 4 * sizeof(int)));
    CK(hipMalloc(&mtp_dids, (size_t)4 * sizeof(int) + 64));
}

// ============================== 模型 =======================================
// 视觉塔输出的图像 embedding：覆盖 d.x 中 [start, start+count) 行的词嵌入。
// data 是宿主 f32 [count][hidden]；start 是整段 prompt 内的 token 下标。
struct EmbSpan {
    const float* data;
    int start;
    int count;
};

struct Model {
    Cfg cfg;
    RT4 rt;
    RT4 mtp_rt;
    std::vector<Layer> layers;
    Wq embed; Wf out_norm;
    Wq lm_head;
    std::vector<std::vector<float>> gm_scale;     // 组优先尺度（host 侧持有直到上传）
    std::vector<float*> gm_dev;

    // KV cache（16 个注意力层）与 GDN 状态（48 个线性层）
    int max_ctx = 131072, G = 128, BG = 64;
    std::vector<u32*> kcache, vcache; std::vector<float*> ksc, vsc, vstage;
    std::vector<float*> ssm_state, conv_state;
    int seq_len = 0;                              // 已缓存 token 数
    std::vector<int> lin_slot;                    // 线性层 → snap 槽位（-1=全注意力）
    std::vector<int> attn_slot;                   // 全注意力层 → vstage 快照槽位
    int n_lin = 0, n_attn = 0;

    // MTP 状态
    Mtp mtp;
    bool mtp_on = false;                          // 权重加载成功且本次运行启用
    int  mtp_n = 3;                               // 每轮最多草稿 token 数
    std::string mtp_path, mtp_json;               // MTP RT4 权重 / manifest（空则不启用）
    int  mtp_len = 0;                             // MTP KV 已写入条数（= 主模型 seq_len-1）
    u32* mtp_kc = nullptr; u32* mtp_vc = nullptr;
    float* mtp_ksc = nullptr; float* mtp_vsc = nullptr; float* mtp_vstage = nullptr;
    bool snap_mode = false;                       // 验证批：GDN/卷积保存逐 token 状态快照

    // 视觉塔（独立 RT4 文件；不参与文本层）
    VisionModel vm;
    bool vision_on = false;

    Dev d;
    int T_MAX = 4096;
    std::string dump_file;
    std::vector<int>* dump_layers = nullptr;
    bool stats = false;
    void dump_x(int il, int n);
    void dump_buf(int tag, const float* p, long long cnt);
    void dump_raw(const std::string& path, const void* p, size_t bytes);
    void db(int il, int stage, const float* p, long long cnt);   // 中间量 dump（RT_DUMP_BUF=1 时才写）
    void print_stats(int il, int n);
    void print_buf(const char* tag, const float* p, long long n);
    int dbg_layer = -1;

    // 分阶段计时（RT_PROF=1 打开；见文件头的 ProfTick）
    bool prof = false;
    double prof_acc[P_NCAT] = {0};
    long long prof_tokens = 0;               // 本次统计包含多少次 forward
    int pa(int cat) { return prof ? cat : -1; }
    void prof_reset() {
        for (int i = 0; i < P_NCAT; i++) prof_acc[i] = 0;
        prof_tokens = 0;
        g_prof_tl.clear();
        g_prof_next = 0;
    }
    void prof_flush() { if (prof) prof_fold(prof_acc); }
    void prof_print(const char* tag);

    void init(const std::string& path, const std::string& json);
    bool load_mtp(const std::string& path, const std::string& json);
    void reset_state();
    // 前向：ids[n]，positions 为绝对位置；返回最后一个 token 的 logits（设备指针）
    void forward(const int* ids, int n, bool log_last, int log_rows = 0,
                 bool extend_mtp = true, bool snap_states = false,
                 const EmbSpan* emb_spans = nullptr, int n_emb_spans = 0, int base_off = 0);
    void attention_layer(int il, int n);
    void gdn_layer(int il, int n);
    void mlp_layer(int il, int n);
    // MTP：把 n 行 (token, hidden) 送进 MTP 层并追加它自己的 KV；rows==1 时可输出 logits
    void mtp_layer_rows(const int* ids_dev, const float* hin, int rows, int pos0,
                        bool want_logits, int* argmax_out);
    int  mtp_draft_one(int token, const float* hin, int snap_slot = -1);
    void mtp_extend_context(const int* ids_dev, int n, int abs_off);
    void mtp_extra_entry(int token, bool want_logits, int* argmax_out, int snap_slot = -1);
    void mtp_restore_stage(int slot);
    void rollback_state(int keep);
    // T 行 × N 输出的线性层：M 小走 GEMV，M 大走 GEMM（a 已经是量化好的 int4）
    void linear(float* y, const Wq& w, const float* x, int M);
    void linear_quant(float* y, const Wq& w, const float* x, int M, u32* aq, float* asc, int row_stride, int row_off);
};

// 取权重（名字带上语言模型的完整前缀）
static std::string tn(int il, const char* suffix) {
    char buf[256];
    snprintf(buf, sizeof(buf), "model.language_model.layers.%d.%s", il, suffix);
    return buf;
}

void Model::init(const std::string& path, const std::string& json) {
    rt.load(path, json);
    rt.upload();
    layers.resize(cfg.n_layer);
    // 组优先尺度（先做 host 侧转换，之后再上传）
    std::vector<std::pair<Wq*, RT4Tensor*>> need_gm;
    auto W = [&](const std::string& n) -> Wq {
        const RT4Tensor* t = rt.find(n);
        if (!t) { printf("缺张量 %s\n", n.c_str()); exit(1); }
        Wq w;
        if (t->kind == "i4") {
            w.q = rt.qptr(t); w.s = rt.sptr(t); w.N = (int)t->N; w.K = (int)t->K; w.group = (int)t->group;
        } else { printf("权重 %s 不是 i4（%s），运行时暂不支持\n", n.c_str(), t->kind.c_str()); exit(1); }
        return w;
    };
    // 小张量（norm/conv/A_log/dt_bias/未量化的投影）：统一转 f32 放显存
    auto F = [&](const std::string& n) -> Wf {
        const RT4Tensor* t = rt.find(n);
        if (!t) { printf("缺张量 %s\n", n.c_str()); exit(1); }
        Wf w;
        if (t->kind == "f16") {
            const uint16_t* src = (const uint16_t*)(rt.hptr(t->q_off));
            const int cnt = (int)(t->nbytes / 2);
            std::vector<float> tmp(cnt);
            for (int i = 0; i < cnt; i++) {
                const uint16_t h = src[i];
                const uint32_t sg = (h >> 15) & 1, ex = (h >> 10) & 0x1F, ma = h & 0x3FF;
                float f = ex == 0 ? ldexpf(ma / 1024.f, -14) : ldexpf(1.f + ma / 1024.f, (int)ex - 15);
                tmp[i] = sg ? -f : f;
            }
            float* dp; CK(hipMalloc(&dp, (size_t)cnt * 4));
            CK(hipMemcpy(dp, tmp.data(), (size_t)cnt * 4, hipMemcpyHostToDevice));
            w.p = dp; w.n = cnt;
        } else {
            const int cnt = (int)(t->nbytes / 4);
            float* dp; CK(hipMalloc(&dp, (size_t)cnt * 4));
            CK(hipMemcpy(dp, rt.hptr(t->q_off), (size_t)cnt * 4, hipMemcpyHostToDevice));
            w.p = dp; w.n = cnt;
        }
        return w;
    };
    for (int il = 0; il < cfg.n_layer; il++) {
        Layer& L = layers[il];
        L.full = cfg.is_full(il);
        L.in_ln = F(tn(il, "input_layernorm.weight"));
        L.post_ln = F(tn(il, "post_attention_layernorm.weight"));
        L.mlp_gate = W(tn(il, "mlp.gate_proj.weight"));
        L.mlp_up = W(tn(il, "mlp.up_proj.weight"));
        L.mlp_down = W(tn(il, "mlp.down_proj.weight"));
        if (L.full) {
            L.q_proj = W(tn(il, "self_attn.q_proj.weight"));
            L.k_proj = W(tn(il, "self_attn.k_proj.weight"));
            L.v_proj = W(tn(il, "self_attn.v_proj.weight"));
            L.o_proj = W(tn(il, "self_attn.o_proj.weight"));
            L.q_norm = F(tn(il, "self_attn.q_norm.weight"));
            L.k_norm = F(tn(il, "self_attn.k_norm.weight"));
        } else {
            L.in_qkv = W(tn(il, "linear_attn.in_proj_qkv.weight"));
            L.in_z = W(tn(il, "linear_attn.in_proj_z.weight"));
            L.out_proj = W(tn(il, "linear_attn.out_proj.weight"));
            L.conv1d = F(tn(il, "linear_attn.conv1d.weight"));
            L.dt_bias = F(tn(il, "linear_attn.dt_bias"));
            L.a_log = F(tn(il, "linear_attn.A_log"));
            L.ssm_norm = F(tn(il, "linear_attn.norm.weight"));
            L.ssm_alpha = F(tn(il, "linear_attn.in_proj_a.weight"));
            L.ssm_beta = F(tn(il, "linear_attn.in_proj_b.weight"));
        }
    }
    embed = W("model.language_model.embed_tokens.weight");
    out_norm = F("model.language_model.norm.weight");
    lm_head = W("lm_head.weight");

    // ---- 组优先尺度：给 GEMM 路径准备 ----
    gm_scale.resize(rt.tensors.size());
    gm_dev.assign(rt.tensors.size(), nullptr);
    auto prep_gm = [&](const std::string& n) {
        const RT4Tensor* t = rt.find(n);
        if (!t || t->kind != "i4") return;
        const int idx = rt.index[n];
        if (gm_dev[idx]) return;
        gm_scale[idx] = make_group_major(rt, t);
        CK(hipMalloc(&gm_dev[idx], gm_scale[idx].size() * 4));
        CK(hipMemcpy(gm_dev[idx], gm_scale[idx].data(), gm_scale[idx].size() * 4, hipMemcpyHostToDevice));
        gm_scale[idx].clear(); gm_scale[idx].shrink_to_fit();
    };
    for (int il = 0; il < cfg.n_layer; il++) {
        prep_gm(tn(il, "mlp.gate_proj.weight"));
        prep_gm(tn(il, "mlp.up_proj.weight"));
        prep_gm(tn(il, "mlp.down_proj.weight"));
        if (layers[il].full) {
            prep_gm(tn(il, "self_attn.q_proj.weight"));
            prep_gm(tn(il, "self_attn.k_proj.weight"));
            prep_gm(tn(il, "self_attn.v_proj.weight"));
            prep_gm(tn(il, "self_attn.o_proj.weight"));
        } else {
            prep_gm(tn(il, "linear_attn.in_proj_qkv.weight"));
            prep_gm(tn(il, "linear_attn.in_proj_z.weight"));
            prep_gm(tn(il, "linear_attn.out_proj.weight"));
        }
    }
    prep_gm("lm_head.weight");
    for (int il = 0; il < cfg.n_layer; il++) {
        Layer& L = layers[il];
        L.mlp_gate.s_gm = gm_dev[rt.index[tn(il, "mlp.gate_proj.weight")]];
        L.mlp_up.s_gm   = gm_dev[rt.index[tn(il, "mlp.up_proj.weight")]];
        L.mlp_down.s_gm = gm_dev[rt.index[tn(il, "mlp.down_proj.weight")]];
        if (L.full) {
            L.q_proj.s_gm = gm_dev[rt.index[tn(il, "self_attn.q_proj.weight")]];
            L.k_proj.s_gm = gm_dev[rt.index[tn(il, "self_attn.k_proj.weight")]];
            L.v_proj.s_gm = gm_dev[rt.index[tn(il, "self_attn.v_proj.weight")]];
            L.o_proj.s_gm = gm_dev[rt.index[tn(il, "self_attn.o_proj.weight")]];
        } else {
            L.in_qkv.s_gm   = gm_dev[rt.index[tn(il, "linear_attn.in_proj_qkv.weight")]];
            L.in_z.s_gm     = gm_dev[rt.index[tn(il, "linear_attn.in_proj_z.weight")]];
            L.out_proj.s_gm = gm_dev[rt.index[tn(il, "linear_attn.out_proj.weight")]];
        }
    }
    lm_head.s_gm = gm_dev[rt.index["lm_head.weight"]];

    // ---- 激活缓冲 ----
    d.alloc(T_MAX);

    // ---- MTP 验证批的逐 token 状态快照 ----
    lin_slot.assign(cfg.n_layer, -1);
    attn_slot.assign(cfg.n_layer, -1);
    n_lin = 0; n_attn = 0;
    for (int il = 0; il < cfg.n_layer; il++) {
        if (!cfg.is_full(il)) lin_slot[il] = n_lin++;
        else                 attn_slot[il] = n_attn++;
    }
    const size_t ssm_one = (size_t)cfg.lv_head * cfg.ldim * cfg.ldim;
    const size_t conv_one = (size_t)(cfg.conv_k - 1) * 10240;
    // 快照只在 MTP 验证批（K+1 ≤ 4 行）里用；--no-mtp 时不分配，省 ~645MB 显存。
    const bool want_snap = !mtp_path.empty() || !mtp_json.empty();
    if (want_snap && n_lin > 0) {
        CK(hipMalloc(&d.ssm_snap, (size_t)n_lin * 4 * ssm_one * 4));
        CK(hipMalloc(&d.conv_snap, (size_t)n_lin * 4 * conv_one * 4));
    }
    if (want_snap && n_attn > 0) {
        const size_t vs = (size_t)cfg.n_kv * BG * cfg.head_dim;
        CK(hipMalloc(&d.vstage_snap, (size_t)n_attn * 4 * vs * 4));
    }

    // ---- KV cache / GDN 状态 ----
    kcache.resize(cfg.n_layer, nullptr); vcache.resize(cfg.n_layer, nullptr);
    ksc.resize(cfg.n_layer, nullptr); vsc.resize(cfg.n_layer, nullptr);
    vstage.resize(cfg.n_layer, nullptr);
    ssm_state.resize(cfg.n_layer, nullptr); conv_state.resize(cfg.n_layer, nullptr);
    const int n_tile = max_ctx / BG;
    for (int il = 0; il < cfg.n_layer; il++) {
        if (cfg.is_full(il)) {
            CK(hipMalloc(&kcache[il], (size_t)cfg.n_kv * max_ctx * (cfg.head_dim / 8) * 4));
            CK(hipMalloc(&ksc[il], (size_t)cfg.n_kv * 2 * max_ctx * 4));
            CK(hipMalloc(&vcache[il], (size_t)cfg.n_kv * (max_ctx / 8) * cfg.head_dim * 4));
            CK(hipMalloc(&vsc[il], (size_t)cfg.n_kv * n_tile * cfg.head_dim * 4));
            CK(hipMalloc(&vstage[il], (size_t)cfg.n_kv * BG * cfg.head_dim * 4));
        } else {
            const size_t sd = (size_t)cfg.lv_head * cfg.ldim * cfg.ldim;
            CK(hipMalloc(&ssm_state[il], sd * 4));
            CK(hipMalloc(&conv_state[il], (size_t)(cfg.conv_k - 1) * 10240 * 4));
        }
    }
    printf("模型就绪：%d 层（%d 注意力 / %d 线性），max_ctx=%d\n",
           cfg.n_layer, cfg.n_layer / cfg.full_interval, cfg.n_layer - cfg.n_layer / cfg.full_interval,
           max_ctx);
    if (!mtp_path.empty() || !mtp_json.empty()) {
        if (load_mtp(mtp_path, mtp_json)) {
            printf("MTP 就绪：1 层，草稿数 %d，KV 容量 %d\n", mtp_n, max_ctx);
        } else {
            printf("MTP 未启用（权重加载失败）\n");
        }
    }
}

// 加载 MTP 头（独立 RT4 文件，15 张量 / 0.22GB）。权重命名与 HF 一致：
//   mtp.fc.weight, mtp.layers.0.*, mtp.norm.weight,
//   mtp.pre_fc_norm_{embedding,hidden}.weight
bool Model::load_mtp(const std::string& path, const std::string& json) {
    if (path.empty() || json.empty()) return false;
    mtp_rt.load(path, json);
    mtp_rt.upload();

    auto MW = [&](const std::string& n) -> Wq {
        const RT4Tensor* t = mtp_rt.find(n);
        if (!t) { printf("MTP 缺张量 %s\n", n.c_str()); exit(1); }
        if (t->kind != "i4") { printf("MTP 权重 %s 不是 i4（%s）\n", n.c_str(), t->kind.c_str()); exit(1); }
        Wq w;
        w.q = mtp_rt.qptr(t); w.s = mtp_rt.sptr(t);
        w.N = (int)t->N; w.K = (int)t->K; w.group = (int)t->group;
        return w;
    };
    auto MF = [&](const std::string& n) -> Wf {
        const RT4Tensor* t = mtp_rt.find(n);
        if (!t) { printf("MTP 缺张量 %s\n", n.c_str()); exit(1); }
        Wf w;
        if (t->kind == "f16") {
            const uint16_t* src = (const uint16_t*)mtp_rt.hptr(t->q_off);
            const int cnt = (int)(t->nbytes / 2);
            std::vector<float> tmp(cnt);
            for (int i = 0; i < cnt; i++) {
                const uint16_t h = src[i];
                const uint32_t sg = (h >> 15) & 1, ex = (h >> 10) & 0x1F, ma = h & 0x3FF;
                const float f = ex == 0 ? ldexpf(ma / 1024.f, -14)
                                        : ldexpf(1.f + ma / 1024.f, (int)ex - 15);
                tmp[i] = sg ? -f : f;
            }
            float* dp; CK(hipMalloc(&dp, (size_t)cnt * 4));
            CK(hipMemcpy(dp, tmp.data(), (size_t)cnt * 4, hipMemcpyHostToDevice));
            w.p = dp; w.n = cnt;
        } else {
            const int cnt = (int)(t->nbytes / 4);
            float* dp; CK(hipMalloc(&dp, (size_t)cnt * 4));
            CK(hipMemcpy(dp, mtp_rt.hptr(t->q_off), (size_t)cnt * 4, hipMemcpyHostToDevice));
            w.p = dp; w.n = cnt;
        }
        return w;
    };
    auto pgm = [&](Wq& w, const char* n) {
        const RT4Tensor* t = mtp_rt.find(n);
        std::vector<float> gm = make_group_major(mtp_rt, t);
        CK(hipMalloc(&w.s_gm, gm.size() * 4));
        CK(hipMemcpy(w.s_gm, gm.data(), gm.size() * 4, hipMemcpyHostToDevice));
    };

    mtp.fc = MW("mtp.fc.weight");
    mtp.pre_norm_embedding = MF("mtp.pre_fc_norm_embedding.weight");
    mtp.pre_norm_hidden = MF("mtp.pre_fc_norm_hidden.weight");
    mtp.norm = MF("mtp.norm.weight");
    mtp.in_ln = MF("mtp.layers.0.input_layernorm.weight");
    mtp.post_ln = MF("mtp.layers.0.post_attention_layernorm.weight");
    mtp.q_norm = MF("mtp.layers.0.self_attn.q_norm.weight");
    mtp.k_norm = MF("mtp.layers.0.self_attn.k_norm.weight");
    mtp.q_proj = MW("mtp.layers.0.self_attn.q_proj.weight");
    mtp.k_proj = MW("mtp.layers.0.self_attn.k_proj.weight");
    mtp.v_proj = MW("mtp.layers.0.self_attn.v_proj.weight");
    mtp.o_proj = MW("mtp.layers.0.self_attn.o_proj.weight");
    mtp.mlp_gate = MW("mtp.layers.0.mlp.gate_proj.weight");
    mtp.mlp_up = MW("mtp.layers.0.mlp.up_proj.weight");
    mtp.mlp_down = MW("mtp.layers.0.mlp.down_proj.weight");
    pgm(mtp.fc, "mtp.fc.weight");
    pgm(mtp.q_proj, "mtp.layers.0.self_attn.q_proj.weight");
    pgm(mtp.k_proj, "mtp.layers.0.self_attn.k_proj.weight");
    pgm(mtp.v_proj, "mtp.layers.0.self_attn.v_proj.weight");
    pgm(mtp.o_proj, "mtp.layers.0.self_attn.o_proj.weight");
    pgm(mtp.mlp_gate, "mtp.layers.0.mlp.gate_proj.weight");
    pgm(mtp.mlp_up, "mtp.layers.0.mlp.up_proj.weight");
    pgm(mtp.mlp_down, "mtp.layers.0.mlp.down_proj.weight");

    const int KV = cfg.n_kv, D = cfg.head_dim;
    CK(hipMalloc(&mtp_kc, (size_t)KV * max_ctx * (D / 8) * 4));
    CK(hipMalloc(&mtp_ksc, (size_t)KV * 2 * max_ctx * 4));
    CK(hipMalloc(&mtp_vc, (size_t)KV * (max_ctx / 8) * D * 4));
    CK(hipMalloc(&mtp_vsc, (size_t)KV * (max_ctx / BG) * D * 4));
    CK(hipMalloc(&mtp_vstage, (size_t)KV * BG * D * 4));
    CK(hipMalloc(&d.mtp_vstage_snap, (size_t)9 * KV * BG * D * 4));
    mtp.loaded = true;
    mtp_on = true;
    return true;
}

void Model::reset_state() {
    seq_len = 0;
    mtp_len = 0;
    for (int il = 0; il < cfg.n_layer; il++) {
        if (cfg.is_full(il)) {
            // KV 的尺度数组要初始化（未写的 tile 不能是 NaN）
            k_fill(ksc[il], 1.f, (long long)cfg.n_kv * 2 * max_ctx);
            k_fill(vsc[il], 1.f, (long long)cfg.n_kv * (max_ctx / BG) * cfg.head_dim);
        } else {
            k_fill(ssm_state[il], 0.f, (long long)cfg.lv_head * cfg.ldim * cfg.ldim);
            k_fill(conv_state[il], 0.f, (long long)(cfg.conv_k - 1) * 10240);
        }
    }
    if (mtp_on) {
        k_fill(mtp_ksc, 1.f, (long long)cfg.n_kv * 2 * max_ctx);
        k_fill(mtp_vsc, 1.f, (long long)cfg.n_kv * (max_ctx / BG) * cfg.head_dim);
    }
    CK(hipMemset(d.qq, 0, (size_t)24 * (((T_MAX + 63) / 64) * 64) * 32 * 4));
    k_fill(d.qs, 1.f, (long long)24 * 2 * (((T_MAX + 63) / 64) * 64));
    k_fill(d.h_prev, 0.f, (long long)cfg.hidden);
    CK(hipDeviceSynchronize());
}

// 线性层：M 小（≤4）走 W4A8 解码 GEMV，M 大走 W4A8 GEMM（激活 8bit）。
// 权重一律是 int4（RT4）；激活 8bit 是质量与速度的折中：解码是带宽瓶颈，int8 激活
// 不额外花带宽；预填充要多跑一遍 int4 GEMM（约 2 倍算力代价），但每层误差从
// int4 激活的 12~16% 降到 ~1.4%（实测 tools/ref_stages.py + build/qtest.py）。
// RT_ACT4=1 可以强制回退到纯 int4 激活（只用于性能对照）。
void Model::linear_quant(float* y, const Wq& w, const float* x, int M,
                         u32* aq, float* asc, int row_stride, int row_off) {
    static const int act4 = getenv("RT_ACT4") ? atoi(getenv("RT_ACT4")) : 0;
    if (M <= 4) {
        if (act4) k_gemv_w4a4(y, w.q, w.s, x + row_off, M, w.N, w.K);
        else      k_gemv_w4a8(y, w.q, w.s, x + row_off, M, w.N, w.K);
    } else {
        const int Mp = ((M + BM_ALIGN - 1) / BM_ALIGN) * BM_ALIGN;   // GEMM 的 M 必须对齐
        if (act4) {
            k_quant_rows(aq, asc, x + row_off, Mp, w.K, w.group);
            k_gemm_i4(y, w.q, w.s_gm, aq, asc, Mp, w.N, w.K);
        } else {
            k_quant_rows_a8(d.aq_h, d.aq_l, d.asc8, d.asc16, x + row_off, Mp, w.K, w.group);
            k_gemm_i4_a8(y, d.c_tmp, w.q, w.s_gm, d.aq_h, d.aq_l, d.asc8, d.asc16,
                         Mp, w.N, w.K);
        }
    }
}
void Model::linear(float* y, const Wq& w, const float* x, int M) {
    // 激活量化暂存：按最大 K（17408）与 T_MAX 预留
    ProfTick _t(pa(P_LINEAR));
    linear_quant(y, w, x, M, d.aq_big, d.asc_big, 0, 0);
}


// ------------------------------ 注意力层 -----------------------------------
// Qwen3.5 全注意力：q_proj 每头出 [q(256), gate(256)]，q/k 各自 RMSNorm(256)、
// 部分 RoPE(前 64 维)、int4 FlashAttention、再乘 sigmoid(gate)、过 o_proj。
void Model::attention_layer(int il, int n) {
    Layer& L = layers[il];
    const int T = n, H = cfg.n_head, KV = cfg.n_kv, D = cfg.head_dim;
    const int pd = H * D * 2, qd = H * D;
    const int n_kv = seq_len + T;
    const int TP = ((T + 63) / 64) * 64;

    if (getenv("RT_TRACE")) printf("    [trace] attn layer %d T=%d seq_len=%d\n", il, T, seq_len);
    linear(d.qfull, L.q_proj, d.xb, T);          // [T][12288] = 每头 [q(256) gate(256)]
    linear(d.hkk, L.k_proj, d.xb, T);            // [T][1024] = [T][4][256]
    linear(d.hvv, L.v_proj, d.xb, T);
    db(il, 1, d.qfull, (long long)T * pd);
    db(il, 4, d.hkk, (long long)T * KV * D);
    db(il, 5, d.hvv, (long long)T * KV * D);

    {
        ProfTick _t(pa(P_ATTNPRE));
        // q_proj 每个头出 [q(256), gate(256)]，所以头间距是 2D，头内偏移 0 / D
        k_gather_heads(d.hq, d.qfull, T, H, D, pd, 0, 2 * D);      // q 抽成 [T][24][256]
        k_gather_heads(d.hgate, d.qfull, T, H, D, pd, D, 2 * D);   // gate 同上
        db(il, 3, d.hgate, (long long)T * H * D);
        k_rmsnorm(d.hq, d.hq, (const float*)L.q_norm.p, T * H, D, cfg.eps, true);
        k_rmsnorm(d.hkk, d.hkk, (const float*)L.k_norm.p, T * KV, D, cfg.eps, true);

        // 位置直接由 seq_len 推出来（原来每层都 hipMalloc + 阻塞 H2D + hipFree）
        k_rope(d.hq, d.hkk, nullptr, seq_len, T, T, H, KV, D, cfg.rot, cfg.rope_theta);
        db(il, 6, d.hkk, (long long)T * KV * D);      // k：rmsnorm + rope 之后

        // HF 的 attention scaling：scores 乘 1/sqrt(head_dim)，折进 Q 最省事
        k_scale(d.hq, 1.f / sqrtf((float)D), (long long)T * H * D);
        db(il, 2, d.hq, (long long)T * H * D);        // 已含 rmsnorm + rope + scale
    }

    {
        ProfTick _t(pa(P_QK));
        // 注意：Q 的尺度数组布局必须和 FA 内核读的一致（Qscg[g*n_q + row]，n_q=TP 是
        // 补齐后的行数），所以尺度按 TP 行写；按 T 行写会让 T 不是 64 倍数时整段错位。
        // 只量化真实的 T 行（解码时 T=1，原来按 TP=64 行做，白跑 63/64），
        // 补齐行保持 reset_state 的 0/1，FA 逐行独立、输出会被 scatter 丢掉。
        k_attn_q_quant(d.qq, d.qs, d.hq, T, H, D, 128, TP);
        k_kv_append_k(kcache[il], ksc[il], d.hkk, seq_len, T, KV, D, 128, max_ctx);
        if (!snap_mode) {
            ProfTick _t2(pa(P_KV));
            k_kv_append_v(vcache[il], vsc[il], vstage[il], d.hvv, seq_len, T, KV, D, BG,
                          max_ctx);
        }
        }
    if (getenv("RT_DUMP_ATTN") && il == atoi(getenv("RT_DUMP_ATTN"))) {
        // 把 FA 的输入原样落盘（不含补齐的尾部），给 tools/attn_check.py 逐元素核对
        const int nk = std::min(max_ctx, ((n_kv + 63) / 64) * 64);
        const size_t qb = (size_t)H * TP * (D / 8) * 4;
        dump_raw("build/attn_qq.bin", d.qq, qb);
        dump_raw("build/attn_qs.bin", d.qs, (size_t)H * 2 * TP * 4);
        // KV cache 按**分配容量** max_ctx 做行距，所以整块导出（前 nk 个 key 用得上）
        dump_raw("build/attn_k.bin", kcache[il], (size_t)KV * max_ctx * (D / 8) * 4);
        dump_raw("build/attn_ks.bin", ksc[il], (size_t)KV * 2 * max_ctx * 4);
        dump_raw("build/attn_v.bin", vcache[il], (size_t)KV * (max_ctx / 8) * D * 4);
        dump_raw("build/attn_vs.bin", vsc[il], (size_t)KV * (max_ctx / BG) * D * 4);
        printf("  [attn] 层 %d T=%d 的 FA 输入已导出（K 取前 %d 个 key）\n", il, T, nk);
    }
    {
        ProfTick _t(pa(P_FA));
        if (snap_mode && T <= 4) {
            // MTP 验证批：为保证与单 token 解码完全同算术（fp32 P），仍走解码内核。
            // 预填充 FA 的 P 是 4bit，用它校验会让接受/回滚后的轨迹与普通解码漂移。
            // 逐行追加 + 快照 V tile 尺度（回滚时逐行恢复），再一次性算完 T 行：
            // 每行的算术与单行内核逐位相同，但 KV 只读一遍、T 行并行（见 src/k_fa.hip）。
            for (int r = 0; r < T; r++) {
                // V 的 tile 尺度按前缀重算：逐行追加，复现单 token 解码时的量化。
                { ProfTick _t2(pa(P_KV));
                  k_kv_append_v(vcache[il], vsc[il], vstage[il],
                                d.hvv + (size_t)r * KV * D, seq_len + r, 1, KV, D, BG,
                                max_ctx); }
                {
                    const size_t vs = (size_t)KV * BG * D;
                    CK(hipMemcpyAsync(d.vstage_snap +
                                      ((size_t)attn_slot[il] * 4 + r) * vs,
                                      vstage[il], vs * 4, hipMemcpyDeviceToDevice, 0));
                }
            }
            // Vsnap 传每行 append 之后的 fp32 V tile 暂存：内核用它按「本行 prefix」
            // 重算最后一格的 V 量化，才能与逐 token 解码逐位相同。
            const size_t vs = (size_t)KV * BG * D;
            const int seq_mode = getenv("RT_VERIFY_SEQ") ? atoi(getenv("RT_VERIFY_SEQ")) : 0;
            if (seq_mode == 1) {                     // 诊断用：逐行调用（慢，等价性 A/B）
                for (int r = 0; r < T; r++) {
                    k_attention_decode_rows(d.hfa + (size_t)r * D,
                                            d.qq + (size_t)r * (D / 8), d.qs + r,
                                            kcache[il], ksc[il], vcache[il], vsc[il],
                                            d.vstage_snap +
                                                ((size_t)attn_slot[il] * 4 + r) * vs,
                                            0, TP, 1, seq_len + r + 1, H, H / KV, max_ctx,
                                            d.fa_pout, d.fa_pmax, d.fa_psum, 8);
                }
            } else {
                k_attention_decode_rows(d.hfa, d.qq, d.qs, kcache[il], ksc[il], vcache[il],
                                        vsc[il], d.vstage_snap + (size_t)attn_slot[il] * 4 * vs,
                                        vs, TP, T, seq_len + 1, H, H / KV, max_ctx,
                                        d.fa_pout, d.fa_pmax, d.fa_psum, 8);
            }
        } else {
            // FA 内核用 n_q 同时算「头内偏移」，所以要传补齐到 64 的 TP，否则各头输出互相踩
            k_attention(d.hfa, d.qq, d.qs, kcache[il], ksc[il], vcache[il], vsc[il],
                        TP, T, n_kv, seq_len, (n_kv + BG - 1) / BG, H, H / KV, max_ctx,
                        d.fa_pout, d.fa_pmax, d.fa_psum, 8);
        }
    }
    // 注意：输出必须在**注意力跑完之后**导出（以前放在前面，落盘的是上一次的陈旧值）
    if (getenv("RT_DUMP_ATTN") && il == atoi(getenv("RT_DUMP_ATTN")))
        dump_raw("build/attn_out.bin", d.hfa, (size_t)H * TP * D * 4);
    // 诊断用：验证批（T≥2）的注意力输出，用来对比「逐行顺序」与「一次算完」两条路。
    db(il, 7, d.hfa, (long long)H * TP * D);
    {
        ProfTick _t(pa(P_NORM));
        k_scatter_heads(d.hout, d.hfa, T, H, D, qd, 0, TP);
        k_sigmoid_mul(d.hout, d.hout, d.hgate, (long long)T * qd);
    }
    db(il, 8, d.hout, (long long)T * qd);
    linear(d.gz, L.o_proj, d.hout, T);
    db(il, 9, d.gz, (long long)T * cfg.hidden);
    k_add_inplace(d.x, d.gz, (long long)T * cfg.hidden);
}

// --------------------------- 线性注意力（GDN）-------------------------------
void Model::gdn_layer(int il, int n) {
    Layer& L = layers[il];
    const int T = n, Hk = cfg.lk_head, Hv = cfg.lv_head, D = cfg.ldim;
    const int qn = Hk * D, vn = Hv * D;

    linear(d.qkv3, L.in_qkv, d.xb, T);           // [T][10240] = [q(2048) k(2048) v(6144)]
    linear(d.gz, L.in_z, d.xb, T);               // z [T][6144]
    db(il, 1, d.qkv3, (long long)T * (qn * 2 + vn));
    db(il, 2, d.gz, (long long)T * vn);
    {
        ProfTick _t(pa(P_CONV));
        // 卷积要读「更新前」的状态，所以先快照
        CK(hipMemcpyAsync(d.conv_prev, conv_state[il],
                          (size_t)(cfg.conv_k - 1) * (qn * 2 + vn) * 4,
                          hipMemcpyDeviceToDevice, 0));
        k_conv1d_silu(d.conv, d.qkv3, (const float*)L.conv1d.p, d.conv_prev, T, qn * 2 + vn,
                      cfg.conv_k);
        if (snap_mode) {
            const size_t conv_one = (size_t)(cfg.conv_k - 1) * (qn * 2 + vn);
            float* snap = d.conv_snap + (size_t)lin_slot[il] * 4 * conv_one;
            k_conv_state_update_snap(conv_state[il], snap, d.qkv3, d.conv_prev, T,
                                     qn * 2 + vn, cfg.conv_k);
        } else {
            k_conv_state_update(conv_state[il], d.qkv3, d.conv_prev, T, qn * 2 + vn, cfg.conv_k);
        }
    }
    db(il, 3, d.conv, (long long)T * (qn * 2 + vn));
    {
        ProfTick _t(pa(P_GNORM));
        k_split_qkv(d.gq, d.gk, d.gv, d.conv, T, qn, qn, vn);
        k_l2norm(d.gq, T * Hk, D, cfg.eps);
        k_l2norm(d.gk, T * Hk, D, cfg.eps);
    }
    db(il, 4, d.gq, (long long)T * qn);
    db(il, 5, d.gk, (long long)T * qn);
    db(il, 6, d.gv, (long long)T * vn);

    {
        ProfTick _t(pa(P_SSM));
        k_ssm_ab_gate(d.gab, d.gbeta, d.gg, (const float*)L.ssm_alpha.p,
                      (const float*)L.ssm_beta.p, d.xb, (const float*)L.dt_bias.p,
                      (const float*)L.a_log.p, T, Hv, cfg.hidden);
    }
    db(il, 7, d.gab, (long long)2 * T * Hv);     // a | b（b 还是 logits）
    db(il, 8, d.gbeta, (long long)T * Hv);       // beta
    db(il, 9, d.gg, (long long)T * Hv);                     // g
    {
        ProfTick _t(pa(P_GDNA));
        if (snap_mode) {
            const size_t ssm_one = (size_t)Hv * D * D;
            float* snap = d.ssm_snap + (size_t)lin_slot[il] * 4 * ssm_one;
            k_gdn_snap(d.hout, d.gq, d.gk, d.gv, d.gg, d.gbeta, ssm_state[il], snap,
                       T, Hk, Hv, D, Hv / Hk);
        } else {
            k_gdn(d.hout, d.gq, d.gk, d.gv, d.gg, d.gbeta, ssm_state[il],
                  T, Hk, Hv, D, Hv / Hk);
        }
    }
    db(il, 10, d.hout, (long long)T * vn);
    {
        ProfTick _t(pa(P_GNORM));
        k_rmsnorm_gated(d.hout, d.hout, (const float*)L.ssm_norm.p, d.gz, T * Hv, D, cfg.eps);
    }
    db(il, 11, d.hout, (long long)T * vn);
    linear(d.gz, L.out_proj, d.hout, T);
    db(il, 12, d.gz, (long long)T * cfg.hidden);
    k_add_inplace(d.x, d.gz, (long long)T * cfg.hidden);
}

// --------------------------------- MLP -------------------------------------
void Model::mlp_layer(int il, int n) {
    Layer& L = layers[il];
    linear(d.m_gate, L.mlp_gate, d.xb, n);
    linear(d.m_up, L.mlp_up, d.xb, n);
    db(il, 21, d.m_gate, (long long)n * cfg.inter);
    db(il, 22, d.m_up, (long long)n * cfg.inter);
    {
        ProfTick _t(pa(P_NORM));
        k_silu_mul(d.m_gate, d.m_gate, d.m_up, (long long)n * cfg.inter);
    }
    db(il, 23, d.m_gate, (long long)n * cfg.inter);
    linear(d.gz, L.mlp_down, d.m_gate, n);
    db(il, 24, d.gz, (long long)n * cfg.hidden);
    k_add_inplace(d.x, d.gz, (long long)n * cfg.hidden);
}

// -------------------------------- 前向 -------------------------------------
// log_rows=0：只算最后一行（log_last 为真时）；log_rows>0：算前 log_rows 行的 logits
// （MTP 验证批最多 4 行）。extend_mtp=false 用于验证批：不更新 MTP 上下文，交由
// 引擎按接受长度回滚。snap_states=true 时 GDN/卷积保存逐 token 状态快照。
void Model::forward(const int* ids, int n, bool log_last, int log_rows, bool extend_mtp,
                    bool snap_states, const EmbSpan* emb_spans, int n_emb_spans, int base_off) {
    const int base = seq_len;
    // 常驻缓冲 + 异步拷贝：原来每步都 hipMalloc/hipFree + 同步 H2D
    CK(hipMemcpyAsync(d.dids, ids, (size_t)n * 4, hipMemcpyHostToDevice, 0));
    { ProfTick _t(pa(P_EMBED));
      k_embed(d.x, embed.q, (const unsigned short*)embed.s, d.dids, n, cfg.hidden, 128); }
    // 视觉塔 embedding：在 k_embed 之后覆盖对应行（同一条 stream，顺序有保证）。
    for (int si = 0; si < n_emb_spans; si++) {
        const EmbSpan& s = emb_spans[si];
        const int a = std::max(s.start, base_off);
        const int b = std::min(s.start + s.count, base_off + n);
        if (b <= a) continue;
        CK(hipMemcpyAsync(d.x + (size_t)(a - base_off) * cfg.hidden,
                          s.data + (size_t)(a - s.start) * cfg.hidden,
                          (size_t)(b - a) * cfg.hidden * sizeof(float),
                          hipMemcpyHostToDevice, 0));
    }
    if (stats) print_stats(-1, n);
    if (dump_layers && std::find(dump_layers->begin(), dump_layers->end(), -1) != dump_layers->end())
        dump_x(-1, n);
    snap_mode = snap_states;
    if (snap_states && !d.ssm_snap) {
        printf("forward: 需要逐 token 状态快照，但未分配（MTP 未启用）\n");
        exit(1);
    }
    for (int il = 0; il < cfg.n_layer; il++) {
        { ProfTick _t(pa(P_NORM));
          k_rmsnorm(d.xb, d.x, (const float*)layers[il].in_ln.p, n, cfg.hidden, cfg.eps, true); }
        if (layers[il].full) attention_layer(il, n); else gdn_layer(il, n);
        { ProfTick _t(pa(P_NORM));
          k_rmsnorm(d.xb, d.x, (const float*)layers[il].post_ln.p, n, cfg.hidden, cfg.eps, true); }
        mlp_layer(il, n);
        if (stats) print_stats(il, n);
        if (dump_layers && std::find(dump_layers->begin(), dump_layers->end(), il) != dump_layers->end())
            dump_x(il, n);
    }
    snap_mode = false;
    { ProfTick _t(pa(P_HEAD));
      k_rmsnorm(d.h_norm, d.x, (const float*)out_norm.p, n, cfg.hidden, cfg.eps, true);
      static const int act4 = getenv("RT_ACT4") ? atoi(getenv("RT_ACT4")) : 0;
      if (log_rows > 0) {
          if (log_rows > 4) { printf("forward: log_rows=%d > 4\n", log_rows); exit(1); }
          if (act4) k_gemv_w4a4(d.logits_all, lm_head.q, lm_head.s, d.h_norm, log_rows,
                                lm_head.N, lm_head.K);
          else      k_gemv_w4a8(d.logits_all, lm_head.q, lm_head.s, d.h_norm, log_rows,
                                lm_head.N, lm_head.K);
      } else if (log_last) {
          const float* last = d.h_norm + (size_t)(n - 1) * cfg.hidden;
          if (act4) k_gemv_w4a4(d.logits, lm_head.q, lm_head.s, last, 1, lm_head.N, lm_head.K);
          else      k_gemv_w4a8(d.logits, lm_head.q, lm_head.s, last, 1, lm_head.N, lm_head.K);
      } }
    if (mtp_on && extend_mtp) {
        ProfTick _t(pa(P_MTP));
        mtp_extend_context(d.dids, n, base);
        CK(hipMemcpyAsync(d.h_prev, d.h_norm + (size_t)(n - 1) * cfg.hidden,
                          (size_t)cfg.hidden * 4, hipMemcpyDeviceToDevice, 0));
    }
    prof_tokens++;
    seq_len += n;
}

// -------------------------------- MTP --------------------------------
// 跑一层 MTP：输入 rows 个 (token, hidden) 对，hidden 是主模型最终 norm 后的隐藏态
// （第一层 MTP 深度）或上一轮 MTP 输出（链式草稿）。函数会把这些行追加进 MTP 自己的
// KV，并在 rows==1 且 want_logits 时输出 lm_head logits 的 argmax。
void Model::mtp_layer_rows(const int* ids_dev, const float* hin, int rows, int pos0,
                           bool want_logits, int* argmax_out) {
    if (!mtp_on || rows <= 0) return;
    const int T = rows, H = cfg.hidden, HD = cfg.head_dim, KV = cfg.n_kv;
    const int TP = ((T + 63) / 64) * 64;
    const int qd = cfg.n_head * HD;
    const int t0 = mtp_len;

    k_embed(d.mtp_e, embed.q, (const unsigned short*)embed.s, ids_dev, T, H, 128);
    k_rmsnorm(d.mtp_e, d.mtp_e, (const float*)mtp.pre_norm_embedding.p, T, H, cfg.eps, true);
    k_rmsnorm(d.mtp_hn, hin, (const float*)mtp.pre_norm_hidden.p, T, H, cfg.eps, true);
    k_concat2(d.mtp_cat, d.mtp_e, d.mtp_hn, T, H);
    linear_quant(d.mtp_fc, mtp.fc, d.mtp_cat, T, d.aq_big, d.asc_big, 0, 0);

    // ---- 与主模型 full attention 相同的一层 ----
    k_rmsnorm(d.xb, d.mtp_fc, (const float*)mtp.in_ln.p, T, H, cfg.eps, true);
    linear(d.qfull, mtp.q_proj, d.xb, T);
    linear(d.hkk, mtp.k_proj, d.xb, T);
    linear(d.hvv, mtp.v_proj, d.xb, T);
    k_gather_heads(d.hq, d.qfull, T, cfg.n_head, HD, 2 * qd, 0, 2 * HD);
    k_gather_heads(d.hgate, d.qfull, T, cfg.n_head, HD, 2 * qd, HD, 2 * HD);
    k_rmsnorm(d.hq, d.hq, (const float*)mtp.q_norm.p, T * cfg.n_head, HD, cfg.eps, true);
    k_rmsnorm(d.hkk, d.hkk, (const float*)mtp.k_norm.p, T * KV, HD, cfg.eps, true);
    k_rope(d.hq, d.hkk, nullptr, pos0, T, T, cfg.n_head, KV, HD, cfg.rot, cfg.rope_theta);
    k_scale(d.hq, 1.f / sqrtf((float)HD), (long long)T * cfg.n_head * HD);
    k_attn_q_quant(d.qq, d.qs, d.hq, T, cfg.n_head, HD, 128, TP);
    k_kv_append_k(mtp_kc, mtp_ksc, d.hkk, t0, T, KV, HD, 128, max_ctx);
    k_kv_append_v(mtp_vc, mtp_vsc, mtp_vstage, d.hvv, t0, T, KV, HD, BG, max_ctx);
    const int n_kv = t0 + T;
    k_attention(d.hfa, d.qq, d.qs, mtp_kc, mtp_ksc, mtp_vc, mtp_vsc,
                TP, T, n_kv, t0, (n_kv + BG - 1) / BG, cfg.n_head, cfg.n_head / KV,
                max_ctx, d.fa_pout, d.fa_pmax, d.fa_psum, 8);
    k_scatter_heads(d.hout, d.hfa, T, cfg.n_head, HD, qd, 0, TP);
    k_sigmoid_mul(d.hout, d.hout, d.hgate, (long long)T * qd);
    linear(d.mtp_tmp, mtp.o_proj, d.hout, T);
    k_add_inplace(d.mtp_tmp, d.mtp_fc, (long long)T * H);

    // ---- MLP ----
    k_rmsnorm(d.xb, d.mtp_tmp, (const float*)mtp.post_ln.p, T, H, cfg.eps, true);
    linear(d.m_gate, mtp.mlp_gate, d.xb, T);
    linear(d.m_up, mtp.mlp_up, d.xb, T);
    k_silu_mul(d.m_gate, d.m_gate, d.m_up, (long long)T * cfg.inter);
    linear(d.gz, mtp.mlp_down, d.m_gate, T);
    k_add_inplace(d.gz, d.mtp_tmp, (long long)T * H);
    k_rmsnorm(d.mtp_out, d.gz, (const float*)mtp.norm.p, T, H, cfg.eps, true);

    if (want_logits) {
        if (T != 1) { printf("mtp_layer_rows: want_logits 只支持 1 行\n"); exit(1); }
        const int act4 = getenv("RT_ACT4") ? atoi(getenv("RT_ACT4")) : 0;
        if (act4) k_gemv_w4a4(d.mtp_logits, lm_head.q, lm_head.s, d.mtp_out, 1,
                              lm_head.N, lm_head.K);
        else      k_gemv_w4a8(d.mtp_logits, lm_head.q, lm_head.s, d.mtp_out, 1,
                              lm_head.N, lm_head.K);
        if (argmax_out) k_argmax(d.mtp_logits, 248320, argmax_out);
    }
    mtp_len += T;
}

int Model::mtp_draft_one(int token, const float* hin, int snap_slot) {
    CK(hipMemcpyAsync(d.mtp_dids, &token, sizeof(int), hipMemcpyHostToDevice, 0));
    mtp_layer_rows(d.mtp_dids, hin, 1, mtp_len, true, d.argmax);
    if (snap_slot >= 0) {
        const size_t vs = (size_t)cfg.n_kv * BG * cfg.head_dim;
        CK(hipMemcpyAsync(d.mtp_vstage_snap + (size_t)snap_slot * vs, mtp_vstage,
                          vs * 4, hipMemcpyDeviceToDevice, 0));
    }
    int id = -1;
    CK(hipMemcpy(&id, d.argmax, sizeof(int), hipMemcpyDeviceToHost));
    return id;
}

// 把主模型刚算出的最终 norm 隐藏态按「错一位」的方式送进 MTP，建立/续接 MTP KV。
//   abs_off == 0：prompt 第一段，MTP 条目 k=0..n-2（token ids[1..n-1], hidden h[0..n-2]）
//   abs_off  > 0：用上一 token 的 hidden h_prev 和本段全部 token，条目 k=abs_off-1...
void Model::mtp_extend_context(const int* ids_dev, int n, int abs_off) {
    if (!mtp_on || n <= 0) return;
    const int H = cfg.hidden;
    if (abs_off == 0) {
        const int rows = n - 1;
        if (rows <= 0) return;
        CK(hipMemcpyAsync(d.mtp_hin, d.h_norm, (size_t)rows * H * 4,
                          hipMemcpyDeviceToDevice, 0));
        if (mtp_len != 0) { printf("MTP: chunk0 时 mtp_len=%d != 0\n", mtp_len); exit(1); }
        mtp_layer_rows(ids_dev + 1, d.mtp_hin, rows, 0, false, nullptr);
    } else {
        CK(hipMemcpyAsync(d.mtp_hin, d.h_prev, (size_t)H * 4,
                          hipMemcpyDeviceToDevice, 0));
        CK(hipMemcpyAsync(d.mtp_hin + H, d.h_norm, (size_t)(n - 1) * H * 4,
                          hipMemcpyDeviceToDevice, 0));
        if (mtp_len != abs_off - 1) {
            printf("MTP: abs_off=%d 时 mtp_len=%d != %d\n", abs_off, mtp_len, abs_off - 1);
            exit(1);
        }
        mtp_layer_rows(ids_dev, d.mtp_hin, n, abs_off - 1, false, nullptr);
    }
}

// 全接受时的补一步：把最后一个草稿 token 和上一草稿步的 MTP hidden 组成条目 k=mtp_len，
// 使 MTP KV 与新的主模型 seq_len-1 对齐。
void Model::mtp_extra_entry(int token, bool want_logits, int* argmax_out, int snap_slot) {
    if (!mtp_on) return;
    CK(hipMemcpyAsync(d.mtp_dids, &token, sizeof(int), hipMemcpyHostToDevice, 0));
    mtp_layer_rows(d.mtp_dids, d.mtp_out, 1, mtp_len, want_logits, argmax_out);
    if (snap_slot >= 0) {
        const size_t vs = (size_t)cfg.n_kv * BG * cfg.head_dim;
        CK(hipMemcpyAsync(d.mtp_vstage_snap + (size_t)snap_slot * vs, mtp_vstage,
                          vs * 4, hipMemcpyDeviceToDevice, 0));
    }
}

void Model::mtp_restore_stage(int slot) {
    if (!mtp_on || slot < 0) return;
    const size_t vs = (size_t)cfg.n_kv * BG * cfg.head_dim;
    CK(hipMemcpyAsync(mtp_vstage, d.mtp_vstage_snap + (size_t)slot * vs,
                      vs * 4, hipMemcpyDeviceToDevice, 0));
}

// 验证批之后按保留的候选数回滚：keep 是「接受的候选 token 数」（含 bonus token），
// 取第 keep-1 行状态快照恢复 GDN/卷积；主模型 final norm 还在 d.h_norm 里，
// 顺手取最后接受行的 hidden 作为下一轮 MTP 的 h_prev。
void Model::rollback_state(int keep) {
    const int H = cfg.hidden;
    if (keep <= 0) return;
    CK(hipMemcpyAsync(d.h_prev, d.h_norm + (size_t)(keep - 1) * H,
                      (size_t)H * 4, hipMemcpyDeviceToDevice, 0));
    const size_t ssm_one = (size_t)cfg.lv_head * cfg.ldim * cfg.ldim;
    const size_t conv_one = (size_t)(cfg.conv_k - 1) * 10240;
    for (int il = 0; il < cfg.n_layer; il++) {
        const int slot = lin_slot[il];
        if (slot < 0) {
            const int as = attn_slot[il];
            const size_t vs = (size_t)cfg.n_kv * BG * cfg.head_dim;
            CK(hipMemcpyAsync(vstage[il],
                              d.vstage_snap + ((size_t)as * 4 + (keep - 1)) * vs,
                              vs * 4, hipMemcpyDeviceToDevice, 0));
            continue;
        }
        const float* ss = d.ssm_snap + ((size_t)slot * 4 + (keep - 1)) * ssm_one;
        const float* cs = d.conv_snap + ((size_t)slot * 4 + (keep - 1)) * conv_one;
        CK(hipMemcpyAsync(ssm_state[il], ss, ssm_one * 4, hipMemcpyDeviceToDevice, 0));
        CK(hipMemcpyAsync(conv_state[il], cs, conv_one * 4, hipMemcpyDeviceToDevice, 0));
    }
}

void Model::print_buf(const char* tag, const float* p, long long n) {
    std::vector<float> buf(n);
    CK(hipMemcpy(buf.data(), p, (size_t)n * 4, hipMemcpyDeviceToHost));
    double s = 0, mx = 0; int nan = 0;
    for (float v : buf) { if (std::isnan(v)) nan++; else { s += fabs(v); mx = std::max(mx, (double)fabs(v)); } }
    printf("    [%s] mean|.|=%.5f max|.|=%.4f nan=%d / %lld\n", tag,
           s / (double)(n - nan), mx, nan, n);
    if (nan) {
        int shown = 0;
        for (long long i = 0; i < n && shown < 4; i++)
            if (std::isnan(buf[i])) { printf("        nan@%lld (行 %lld)", i, i % (n / 4)); shown++; }
        printf("\n");
    }
}

// 分阶段计时汇总（RT_PROF=1）。按「每次 forward」折算成毫秒。
// 注意：走 stderr，避免污染 --engine 的逐行协议（stdout）。
void Model::prof_print(const char* tag) {
    if (!prof) return;
    prof_fold(prof_acc);
    const double d = prof_tokens > 0 ? (double)prof_tokens : 1.0;
    double tot = 0;
    for (int i = 0; i < P_NCAT; i++) tot += prof_acc[i];
    fprintf(stderr, "PROF %s: %.2f ms/步（n=%lld 步，合计 %.1f ms）\n", tag, tot / d, prof_tokens,
            tot);
    for (int i = 0; i < P_NCAT; i++)
        fprintf(stderr, "PROF   %-12s %8.2f ms  %5.1f%%\n", P_NAME[i], prof_acc[i] / d,
                100.0 * prof_acc[i] / (tot > 0 ? tot : 1.0));
    fflush(stderr);
}

void Model::print_stats(int il, int n) {
    std::vector<float> buf((size_t)n * cfg.hidden);
    CK(hipMemcpy(buf.data(), d.x, buf.size() * 4, hipMemcpyDeviceToHost));
    double s = 0, mx = 0; int nan = 0;
    for (float v : buf) { if (std::isnan(v)) nan++; else { s += fabs(v); mx = std::max(mx, (double)fabs(v)); } }
    printf("  [x] layer %3d: mean|.|=%.4f max|.|=%.3f nan=%d\n", il, s / (double)(buf.size() - nan), mx, nan);
    if (nan) {
        for (size_t i = 0; i < buf.size(); i++)
            if (std::isnan(buf[i])) { printf("     首个 NaN: idx=%zu (token %zu, dim %zu)\n", i, i / cfg.hidden, i % cfg.hidden); break; }
    }
}

void Model::dump_buf(int tag, const float* p, long long cnt) {
    std::vector<float> buf(cnt);
    CK(hipMemcpy(buf.data(), p, (size_t)cnt * 4, hipMemcpyDeviceToHost));
    FILE* f = fopen(dump_file.c_str(), "ab");
    const int hdr[2] = { tag, (int)cnt };
    fwrite(hdr, sizeof(int), 2, f);
    fwrite(buf.data(), 4, buf.size(), f);
    fclose(f);
    printf("  [dump] tag %d → %s (%lld)\n", tag, dump_file.c_str(), cnt);
}

void Model::dump_x(int il, int n) {
    std::vector<float> buf((size_t)n * cfg.hidden);
    CK(hipMemcpy(buf.data(), d.x, buf.size() * 4, hipMemcpyDeviceToHost));
    FILE* f = fopen(dump_file.c_str(), "ab");
    const int hdr[2] = { il, n * cfg.hidden };
    fwrite(hdr, sizeof(int), 2, f);
    fwrite(buf.data(), 4, buf.size(), f);
    fclose(f);
    printf("  [dump] layer %d → %s (%d×%d)\n", il, dump_file.c_str(), n, cfg.hidden);
}

// 中间量 dump：tag = 1000 + 层号*100 + 阶段号（见 tools/ref_hf.py 的对照表）。
// 只有 RT_DUMP_BUF=1 且给了 --dump 时才写，正常路径零开销。
void Model::db(int il, int stage, const float* p, long long cnt) {
    static const int on = getenv("RT_DUMP_BUF") ? atoi(getenv("RT_DUMP_BUF")) : 0;
    if (!on || dump_file.empty()) return;
    dump_buf(1000 + il * 100 + stage, p, cnt);
}

void Model::dump_raw(const std::string& path, const void* p, size_t bytes) {
    std::vector<uint8_t> buf(bytes);
    CK(hipMemcpy(buf.data(), p, bytes, hipMemcpyDeviceToHost));
    FILE* f = fopen(path.c_str(), "wb");
    fwrite(buf.data(), 1, bytes, f);
    fclose(f);
    printf("  [dump] %s (%zu B)\n", path.c_str(), bytes);
}

// ============================== 采样器 =====================================
struct Sampler {
    float temp = 1.f, top_p = 1.f;
    int top_k = 0;
    uint64_t st = 88172645463325252ull;
    std::vector<std::pair<float, int>> cand;   // (概率, id)
    std::vector<int> idx;
    std::vector<float> ex;                     // 无截断路径的 exp 缓存（省掉第二遍 expf）

    uint32_t rnd() { st ^= st << 13; st ^= st >> 7; st ^= st << 17; return (uint32_t)(st >> 32); }
    float uni() { return (float)(rnd() >> 8) * (1.f / (float)(1u << 24)); }

    // 采样器。原来 top_k=0/top_p=1 时会对全部 248320 个 logit 做一次 std::sort，
    // 墙钟实测比贪心慢 ~20 ms/token（见 RESUME 的「解码提速」第 1 条）。现在：
    //   * 无截断 → 完全不用排序，两遍 O(V) 精确采样；
    //   * 有截断 → nth_element 选出前 K 个（K=top_k 或 2048），只对 K 个排序。
    int pick(const float* lg, int V) {
        if (temp <= 0.f) {                       // 贪心
            int best = 0; float bv = lg[0];
            for (int i = 1; i < V; i++) if (lg[i] > bv) { bv = lg[i]; best = i; }
            return best;
        }
        float mx = lg[0];
        for (int i = 1; i < V; i++) mx = std::max(mx, lg[i]);
        const float inv_t = 1.f / temp;

        if (top_k <= 0 && top_p >= 1.f) {        // 无截断：O(V)，不排序
            ex.resize(V);
            double sum = 0;
            for (int i = 0; i < V; i++) {
                const float e = expf((lg[i] - mx) * inv_t);
                ex[i] = e;
                sum += (double)e;
            }
            const double r = (double)uni() * sum;
            double c = 0;
            for (int i = 0; i < V; i++) {
                c += (double)ex[i];
                if (r <= c) return i;
            }
            return V - 1;
        }

        // 有截断：先选候选集（top_p 用「前 K 大」近似，K 个以外的质量可忽略）
        const int K = std::min(V, top_k > 0 ? top_k : 2048);
        idx.resize(V);
        for (int i = 0; i < V; i++) idx[i] = i;
        std::nth_element(idx.begin(), idx.begin() + K, idx.end(),
                         [&](int a, int b) { return lg[a] > lg[b]; });
        idx.resize(K);
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
        cand.resize(K);
        double sum = 0;
        for (int i = 0; i < K; i++) {
            const float p = expf((lg[idx[i]] - mx) * inv_t);
            cand[i] = {p, idx[i]};
            sum += p;
        }
        int keep = K;
        if (top_p < 1.f) {                       // top-p：留累积概率到 top_p 为止（至少 1 个）
            double acc = 0;
            for (int i = 0; i < K; i++) {
                acc += cand[i].first;
                if (acc >= (double)top_p * sum) { keep = i + 1; break; }
            }
        }
        double acc = 0;
        for (int i = 0; i < keep; i++) acc += cand[i].first;
        const float r = uni() * (float)acc;
        float c = 0;
        for (int i = 0; i < keep; i++) {
            c += cand[i].first;
            if (r <= c) return cand[i].second;
        }
        return cand[keep - 1].second;
    }
};

static void topk_of(const std::vector<float>& lg, int k, std::vector<int>& ids,
                    std::vector<float>& vals) {
    std::vector<int> idx(lg.size());
    for (size_t i = 0; i < idx.size(); i++) idx[i] = (int)i;
    const int kk = std::min<int>(k, (int)idx.size());
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(),
                      [&](int a, int b) { return lg[a] > lg[b]; });
    ids.assign(idx.begin(), idx.begin() + kk);
    vals.clear();
    for (int i = 0; i < kk; i++) vals.push_back(lg[idx[i]]);
}

static std::vector<int> parse_ids(const std::string& s) {
    std::vector<int> v;
    for (size_t p = 0; p < s.size();) {
        size_t q = s.find(',', p);
        if (q == std::string::npos) q = s.size();
        std::string tok = s.substr(p, q - p);
        if (!tok.empty()) v.push_back(atoi(tok.c_str()));
        p = q + 1;
    }
    return v;
}

// 引擎模式：stdin 逐行命令、stdout 逐行结果（scripts/serve.py 当作模型进程）。
//   RESET                                    状态清零
//   PREFILL <id,id,...>                      前向（按 CHUNK 自动分块），回 top-5
//   PREFILL_EMB <ids> <emb_file> <s:c,...>   同 PREFILL，但用视觉塔 embedding 覆盖
//                                            [s,s+c) 行的词嵌入（emb_file 为 f32）
//   IMG_EMB <patch_file> <out_file> <gh> <gw>
//                                            用 RT4 视觉塔编码图片 patch，输出 f32
//                                            [n_tokens,5120]，回 "OK image <n_tokens>"
//   MTP <n>                                  设置 MTP 草稿数（0=普通逐 token 解码）
//   GEN <n> <temp> <top_p> <top_k> <seed> <stop_csv>
//                                            从当前 logits 采样，每个 token 一行 "TOK <id>"，
//                                            收尾 "END <n> <ms>"
//   ROLLTEST / ROLLTEST2                     状态回滚等价性诊断（见各自分支注释）
//   QUIT
static void engine_loop(Model& m) {
    const int CHUNK = 512;
    std::vector<float> lg(248320);
    Sampler smp;
    bool have_logits = false;
    std::string line;

    // GEN 期间非阻塞看一眼 stdin：Web 端「停止」按钮会发 STOP，收到就在本轮
    // （MTP 一轮 ~75ms）结束后中断生成。GEN 期间的其它命令一律丢弃 —— 服务端
    // 在发出 STOP 后会等本次 GEN 收尾才释放请求锁，所以不会有人在这时插命令。
    auto stop_pending = []() -> bool {
        struct pollfd pfd;
        pfd.fd = STDIN_FILENO;
        pfd.events = POLLIN;
        pfd.revents = 0;
        if (std::cin.rdbuf()->in_avail() <= 0 && poll(&pfd, 1, 0) <= 0) return false;
        std::string s;
        if (!std::getline(std::cin, s)) return true;       // stdin 关了，等同于停止
        return s.rfind("STOP", 0) == 0 || s.rfind("QUIT", 0) == 0;
    };

    auto run_prefill = [&](const std::vector<int>& ids, const EmbSpan* spans, int nspans) {
        m.reset_state();
        m.prof_reset();
        auto t0 = std::chrono::steady_clock::now();
        for (size_t off = 0; off < ids.size(); off += CHUNK) {
            const int n = (int)std::min<size_t>(CHUNK, ids.size() - off);
            m.forward(ids.data() + off, n, off + n == ids.size(), 0, true, false,
                      spans, nspans, (int)off);
        }
        CK(hipDeviceSynchronize());
        auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        { ProfTick _t(m.pa(P_COPY));
          CK(hipMemcpy(lg.data(), m.d.logits, lg.size() * 4, hipMemcpyDeviceToHost)); }
        have_logits = true;
        std::vector<int> ti; std::vector<float> tv;
        topk_of(lg, 5, ti, tv);
        printf("OK prefill %zu tokens in %.1f ms (%.1f t/s)", ids.size(), ms,
               ids.size() / (ms / 1000.0));
        for (size_t i = 0; i < ti.size(); i++) printf(" %d:%.3f", ti[i], tv[i]);
        printf("\n");
        fflush(stdout);
        m.prof_print("prefill");
    };

    while (std::getline(std::cin, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        if (line.empty()) continue;
        const size_t sp = line.find(' ');
        const std::string op = sp == std::string::npos ? line : line.substr(0, sp);
        const std::string arg = sp == std::string::npos ? "" : line.substr(sp + 1);
        if (op == "QUIT") break;
        if (op == "RESET") {
            m.reset_state();
            have_logits = false;
            printf("OK reset\n");
            fflush(stdout);
        } else if (op == "MTP") {
            m.mtp_n = std::max(0, std::min(3, atoi(arg.c_str())));
            printf("OK mtp %d%s\n", m.mtp_n, m.mtp_on ? "" : " (no weights)");
            fflush(stdout);
        } else if (op == "PREFILL") {
            const std::vector<int> ids = parse_ids(arg);
            run_prefill(ids, nullptr, 0);
        } else if (op == "PREFILL_EMB") {
            const size_t p1 = arg.find(' ');
            const size_t p2 = p1 == std::string::npos ? std::string::npos
                                                      : arg.find(' ', p1 + 1);
            if (p1 == std::string::npos || p2 == std::string::npos) {
                printf("ERR usage PREFILL_EMB <ids> <emb_file> <start:count,...>\n");
                fflush(stdout);
                continue;
            }
            const std::vector<int> ids = parse_ids(arg.substr(0, p1));
            const std::string path = arg.substr(p1 + 1, p2 - p1 - 1);
            const std::string spec = arg.substr(p2 + 1);
            std::vector<std::pair<int, int>> ranges;
            long long total_tokens = 0;
            size_t p = 0;
            while (p <= spec.size()) {
                const size_t q = spec.find(',', p);
                const std::string one = spec.substr(p, q == std::string::npos
                                                       ? std::string::npos : q - p);
                if (!one.empty()) {
                    const size_t colon = one.find(':');
                    if (colon == std::string::npos) {
                        printf("ERR bad emb span %s\n", one.c_str());
                        fflush(stdout);
                        ranges.clear();
                        break;
                    }
                    const int st = atoi(one.substr(0, colon).c_str());
                    const int ct = atoi(one.substr(colon + 1).c_str());
                    if (ct <= 0) {
                        printf("ERR bad emb count %s\n", one.c_str());
                        fflush(stdout);
                        ranges.clear();
                        break;
                    }
                    ranges.push_back({st, ct});
                    total_tokens += ct;
                }
                if (q == std::string::npos) break;
                p = q + 1;
            }
            if (ranges.empty() && !spec.empty()) continue;
            const size_t H = (size_t)m.cfg.hidden;
            std::vector<float> embs((size_t)total_tokens * H);
            std::ifstream f(path, std::ios::binary);
            if (!f) {
                printf("ERR open emb file %s\n", path.c_str());
                fflush(stdout);
                continue;
            }
            f.seekg(0, std::ios::end);
            const size_t bytes = (size_t)f.tellg();
            f.seekg(0, std::ios::beg);
            if (bytes != embs.size() * sizeof(float)) {
                printf("ERR emb size got %zu expect %zu\n", bytes,
                       embs.size() * sizeof(float));
                fflush(stdout);
                continue;
            }
            if (!embs.empty()) f.read((char*)embs.data(), bytes);
            std::vector<EmbSpan> spans;
            long long off = 0;
            for (const auto& r : ranges) {
                spans.push_back({embs.data() + (size_t)off * H, r.first, r.second});
                off += r.second;
            }
            run_prefill(ids, spans.data(), (int)spans.size());
        } else if (op == "IMG_EMB") {
            if (!m.vision_on) {
                printf("ERR vision not loaded\n");
                fflush(stdout);
                continue;
            }
            std::vector<std::string> f;
            { size_t p = 0; while (p <= arg.size()) { size_t q = arg.find(' ', p);
                if (q == std::string::npos) q = arg.size();
                f.push_back(arg.substr(p, q - p)); p = q + 1; } }
            if (f.size() < 4) {
                printf("ERR usage IMG_EMB <patch_file> <out_file> <gh> <gw>\n");
                fflush(stdout);
                continue;
            }
            int n_tokens = 0;
            const int gh = atoi(f[2].c_str()), gw = atoi(f[3].c_str());
            auto ti0 = std::chrono::steady_clock::now();
            if (m.vm.encode_file(f[0], f[1], n_tokens, gh, gw)) {
                auto ti1 = std::chrono::steady_clock::now();
                const double ms = std::chrono::duration<double, std::milli>(ti1 - ti0).count();
                printf("OK image %d %.1f ms\n", n_tokens, ms);
            } else {
                printf("ERR vision encode failed\n");
            }
            fflush(stdout);
        } else if (op == "GEN") {
            std::vector<std::string> f;
            { size_t p = 0; while (p <= arg.size()) { size_t q = arg.find(' ', p);
                if (q == std::string::npos) q = arg.size();
                f.push_back(arg.substr(p, q - p)); p = q + 1; } }
            const int n = !f.empty() ? atoi(f[0].c_str()) : 1;
            smp.temp = f.size() > 1 ? (float)atof(f[1].c_str()) : 1.f;
            smp.top_p = f.size() > 2 ? (float)atof(f[2].c_str()) : 1.f;
            smp.top_k = f.size() > 3 ? atoi(f[3].c_str()) : 0;
            if (f.size() > 4 && !f[4].empty())
                smp.st = (uint64_t)atoll(f[4].c_str()) ^ 0x9E3779B97F4A7C15ull;
            const std::vector<int> stops = f.size() > 5 ? parse_ids(f[5]) : std::vector<int>{};
            if (!have_logits) { printf("ERR no logits\n"); fflush(stdout); continue; }
            int produced = 0;
            long long draft_try = 0, draft_ok = 0, mtp_rounds = 0;
            bool interrupted = false;              // 客户端发来 STOP
            std::vector<float> tlog;
            m.prof_reset();
            auto t0 = std::chrono::steady_clock::now();
            const size_t V = 248320;
            if (m.mtp_on && smp.temp <= 0.f && m.mtp_n > 0) {
                // 贪心 MTP 投机：每轮草拟 K 个、主模型一次前向校验，接受前缀。
                while (produced < n) {
                    if (stop_pending()) { interrupted = true; break; }
                    int id;
                    { ProfTick _t(m.pa(P_SAMPLE)); id = smp.pick(lg.data(), V); }
                    printf("TOK %d\n", id);
                    fflush(stdout);
                    produced++;
                    if (std::find(stops.begin(), stops.end(), id) != stops.end()) break;
                    if (produced >= n) break;

                    const int K = std::min(std::min(m.mtp_n, n - produced), 3);
                    mtp_rounds++;
                    int drafts[8];
                    const float* hin = m.d.h_prev;
                    for (int i = 0; i < K; i++) {
                        const int feed = (i == 0) ? id : drafts[i - 1];
                        drafts[i] = m.mtp_draft_one(feed, hin, i);
                        hin = m.d.mtp_out;               // 链式：上一 MTP 隐藏态
                    }

                    int cand[8];
                    cand[0] = id;
                    for (int i = 0; i < K; i++) cand[i + 1] = drafts[i];
                    const int base = m.seq_len;
                    {
                        ProfTick _t(m.pa(P_MTP));
                        m.forward(cand, K + 1, false, K + 1, false, true);
                        CK(hipDeviceSynchronize());
                    }
                    // 校验批的 argmax 在主机做：拷回 (K+1)×1MB logits 再扫一遍全词表（0.82ms/轮）。
                    // 试过搬去设备侧（单 block k_argmax × (K+1)），反而 2.30ms/轮——单 block
                    // 读 1MB 是延迟受限的，要赢得多 block 两级归约，留给以后连同小 batch
                    // 注意力一起做（见 docs/MTP.md §7）。
                    tlog.resize((size_t)(K + 1) * V);
                    { ProfTick _t(m.pa(P_COPY));
                      CK(hipMemcpy(tlog.data(), m.d.logits_all, tlog.size() * 4,
                                   hipMemcpyDeviceToHost)); }

                    int acc = 0;
                    for (int i = 0; i < K; i++) {
                        const float* row = tlog.data() + (size_t)i * V;
                        int bi = 0; float bv = row[0];
                        for (int v = 1; v < (int)V; v++)
                            if (row[v] > bv) { bv = row[v]; bi = v; }
                        if (bi == drafts[i]) acc++; else break;
                    }
                    // 调试开关：强制全部拒绝，用来单独验证回滚路径与普通解码等价
                    if (getenv("RT_MTP_FORCE_REJECT")) acc = 0;
                    if (getenv("RT_MTP_TRACE")) {
                        fprintf(stderr, "MTPROUND base=%d K=%d acc=%d drafts=", base, K, acc);
                        for (int i = 0; i < K; i++) fprintf(stderr, "%d,", drafts[i]);
                        fprintf(stderr, "\n");
                    }
                    draft_try += K;
                    draft_ok += acc;

                    // 输出接受的前缀；遇到 stop 或到达 n 就停在这里。
                    int a_use = 0;
                    bool hit_stop = false, hit_limit = false;
                    for (int i = 0; i < acc; i++) {
                        const int tok = drafts[i];
                        printf("TOK %d\n", tok);
                        fflush(stdout);
                        produced++;
                        a_use = i + 1;
                        if (std::find(stops.begin(), stops.end(), tok) != stops.end()) {
                            hit_stop = true;
                            break;
                        }
                        if (produced >= n) { hit_limit = true; break; }
                    }
                    if (!hit_stop && !hit_limit) a_use = acc;

                    const int keep = 1 + a_use;
                    m.rollback_state(keep);
                    m.seq_len = base + keep;
                    if (a_use < K) {
                        m.mtp_len = base + a_use;
                        m.mtp_restore_stage(a_use);
                    } else {
                        // 全接受：最后一个 token 的 MTP 条目还没写，补一步。
                        m.mtp_len = base + K - 1;
                        m.mtp_extra_entry(drafts[K - 1], false, nullptr, K);
                    }

                    if (hit_stop || produced >= n) break;
                    {
                        ProfTick _t(m.pa(P_COPY));
                        memcpy(lg.data(), tlog.data() + (size_t)a_use * V, V * 4);
                    }
                }
            } else {
                for (int i = 0; i < n; i++) {
                    if (stop_pending()) { interrupted = true; break; }
                    int id;
                    { ProfTick _t(m.pa(P_SAMPLE)); id = smp.pick(lg.data(), V); }
                    printf("TOK %d\n", id);
                    fflush(stdout);
                    produced++;
                    if (std::find(stops.begin(), stops.end(), id) != stops.end()) break;
                    if (i + 1 < n) {
                        m.forward(&id, 1, true);
                        { ProfTick _t(m.pa(P_COPY));
                          CK(hipMemcpy(lg.data(), m.d.logits, lg.size() * 4,
                                       hipMemcpyDeviceToHost)); }
                    }
                }
            }
            auto t1 = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            printf("END %d %.1f%s\n", produced, ms, interrupted ? " stopped" : "");
            fflush(stdout);
            if (draft_try > 0)
                fprintf(stderr, "MTP 统计：轮数 %lld，草稿 %lld，接受 %lld（%.3f token/轮，%.1f%%）\n",
                        mtp_rounds, draft_try, draft_ok, (double)draft_ok /
                        std::max(1.0, (double)mtp_rounds),
                        100.0 * (double)draft_ok / (double)draft_try);
            m.prof_print("decode");
        } else if (op == "ROLLTEST") {
            // 调试：<prompt_csv> <t0> <t1> <draft_csv>
            // 比较 (t0,t1) 逐 token 前向，与 [t0,d1,d2,d3] 验证批回滚到 t0 后再前向 t1
            // 的 logits。两者应在 1e-5 量级内一致。
            std::vector<std::string> f;
            { size_t p = 0; while (p <= arg.size()) { size_t q = arg.find(' ', p);
                if (q == std::string::npos) q = arg.size();
                f.push_back(arg.substr(p, q - p)); p = q + 1; } }
            if (f.size() < 4) { printf("ERR usage ROLLTEST <prompt_csv> <t0> <t1> <draft_csv>\n");
                                fflush(stdout); continue; }
            const std::vector<int> prompt = parse_ids(f[0]);
            const int t0 = atoi(f[1].c_str()), t1 = atoi(f[2].c_str());
            std::vector<int> dr = parse_ids(f[3]);
            while ((int)dr.size() < 3) dr.push_back(0);
            const size_t V = 248320;
            std::vector<float> la0(V), la1(V), lb0(V), lb1(V);
            auto run_prefill = [&]() {
                m.reset_state();
                for (size_t off = 0; off < prompt.size(); off += CHUNK) {
                    const int nn = (int)std::min<size_t>(CHUNK, prompt.size() - off);
                    m.forward(prompt.data() + off, nn, off + nn == prompt.size());
                }
                CK(hipDeviceSynchronize());
            };
            auto copy_last = [&](std::vector<float>& v) {
                CK(hipMemcpy(v.data(), m.d.logits, V * 4, hipMemcpyDeviceToHost));
            };
            run_prefill();
            m.forward(&t0, 1, true); copy_last(la0);
            m.forward(&t1, 1, true); copy_last(la1);
            run_prefill();
            const int base = m.seq_len;
            const int cand[4] = {t0, dr[0], dr[1], dr[2]};
            m.forward(cand, 4, false, 4, false, true);
            CK(hipDeviceSynchronize());
            CK(hipMemcpy(lb0.data(), m.d.logits_all, V * 4, hipMemcpyDeviceToHost));
            m.rollback_state(1);
            m.seq_len = base + 1;
            m.mtp_len = base;
            m.forward(&t1, 1, true); copy_last(lb1);
            auto diff = [&](const std::vector<float>& a, const std::vector<float>& b) {
                double mx = 0, s = 0;
                for (size_t i = 0; i < V; i++) { mx = std::max(mx, (double)fabs(a[i] - b[i]));
                                                s += fabs(a[i] - b[i]); }
                return std::pair<double, double>{mx, s / V};
            };
            auto d0 = diff(la0, lb0), d1 = diff(la1, lb1);
            printf("ROLLTEST t0 max=%.6g mean=%.6g   t1 max=%.6g mean=%.6g\n",
                   d0.first, d0.second, d1.first, d1.second);
            fflush(stdout);
        } else if (op == "ROLLTEST2") {
            // 调试：<prompt_csv> <4个候选token_csv> <probe>
            // 对 keep=1..4 分别比较：普通逐 token 前向 vs 验证批 + 回滚后再前向 probe。
            std::vector<std::string> f;
            { size_t p = 0; while (p <= arg.size()) { size_t q = arg.find(' ', p);
                if (q == std::string::npos) q = arg.size();
                f.push_back(arg.substr(p, q - p)); p = q + 1; } }
            if (f.size() < 3) { printf("ERR usage ROLLTEST2 <prompt_csv> <tok4_csv> <probe>\n");
                                fflush(stdout); continue; }
            const std::vector<int> prompt = parse_ids(f[0]);
            std::vector<int> toks = parse_ids(f[1]);
            while ((int)toks.size() < 4) toks.push_back(0);
            const int probe = atoi(f[2].c_str());
            const size_t V = 248320;
            std::vector<float> la(V), lb(V);
            auto run_prefill = [&]() {
                m.reset_state();
                for (size_t off = 0; off < prompt.size(); off += CHUNK) {
                    const int nn = (int)std::min<size_t>(CHUNK, prompt.size() - off);
                    m.forward(prompt.data() + off, nn, off + nn == prompt.size());
                }
                CK(hipDeviceSynchronize());
            };
            auto diff = [&](const std::vector<float>& a, const std::vector<float>& b) {
                double mx = 0, s = 0;
                for (size_t i = 0; i < V; i++) { mx = std::max(mx, (double)fabs(a[i] - b[i]));
                                                s += fabs(a[i] - b[i]); }
                return std::pair<double, double>{mx, s / V};
            };
            for (int keep = 1; keep <= 4; keep++) {
                run_prefill();
                for (int j = 0; j < keep; j++) m.forward(&toks[j], 1, true);
                m.forward(&probe, 1, true);
                CK(hipMemcpy(la.data(), m.d.logits, V * 4, hipMemcpyDeviceToHost));

                run_prefill();
                const int base = m.seq_len;
                m.forward(toks.data(), 4, false, 4, false, true);
                CK(hipDeviceSynchronize());
                m.rollback_state(keep);
                m.seq_len = base + keep;
                m.mtp_len = base + keep - 1;
                m.forward(&probe, 1, true);
                CK(hipMemcpy(lb.data(), m.d.logits, V * 4, hipMemcpyDeviceToHost));
                auto d = diff(la, lb);
                printf("ROLLTEST2 keep=%d max=%.6g mean=%.6g\n", keep, d.first, d.second);
            }
            fflush(stdout);
        } else {
            printf("ERR unknown op %s\n", op.c_str());
            fflush(stdout);
        }
    }
}

// -------------------------------- main -------------------------------------
int main(int argc, char** argv) {
    std::string model, json, dump, mtp_path, mtp_json, vision_path, vision_json;
    std::vector<int> ids;
    bool stats_flag = false;
    int dbg_layer = 1;
    int topk = 10, ctx = 32768;
    int mtp_n = getenv("RT_MTP_N") ? atoi(getenv("RT_MTP_N")) : 3;
    bool no_mtp = getenv("RT_NO_MTP") ? atoi(getenv("RT_NO_MTP")) != 0 : false;
    bool engine = false;
    std::vector<int> dumplayers;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto nx = [&]() { return std::string(argv[++i]); };
        if (a == "--model") model = nx();
        else if (a == "--json") json = nx();
        else if (a == "--mtp") mtp_path = nx();
        else if (a == "--mtp-json") mtp_json = nx();
        else if (a == "--vision") vision_path = nx();
        else if (a == "--vision-json") vision_json = nx();
        else if (a == "--mtp-n") mtp_n = atoi(nx().c_str());
        else if (a == "--no-mtp") no_mtp = true;
        else if (a == "--ctx") ctx = atoi(nx().c_str());
        else if (a == "--topk") topk = atoi(nx().c_str());
        else if (a == "--dump") dump = nx();
        else if (a == "--engine") engine = true;
        else if (a == "--stats") stats_flag = true;
        else if (a == "--dbg-layer") dbg_layer = atoi(nx().c_str());
        else if (a == "--ids") {
            std::string s = nx();
            for (size_t p = 0; p < s.size();) {
                size_t q = s.find(',', p);
                if (q == std::string::npos) q = s.size();
                ids.push_back(atoi(s.substr(p, q - p).c_str()));
                p = q + 1;
            }
        } else if (a == "--dump-layers") {
            std::string s = nx();
            for (size_t p = 0; p < s.size();) {
                size_t q = s.find(',', p);
                if (q == std::string::npos) q = s.size();
                dumplayers.push_back(atoi(s.substr(p, q - p).c_str()));
                p = q + 1;
            }
        } else { printf("未知参数 %s\n", a.c_str()); return 1; }
    }
    if (model.empty() || (ids.empty() && !engine)) {
        printf("用法: rt --model <rt4> --json <manifest> --ids 1,2,3 [--ctx 32768] [--topk 10] "
               "[--mtp mtp.rt4 --mtp-json mtp.rt4.json --mtp-n 3] "
               "[--dump f.bin --dump-layers 0,1,3]  或  rt --engine\n");
        return 1;
    }
    if (no_mtp) { mtp_path.clear(); mtp_json.clear(); }
    // 未显式给 MTP 时：先看 RT_MTP 环境变量，再找权重同目录的 qwen38_27b_mtp.rt4。
    if (!no_mtp && mtp_path.empty()) {
        const char* env = getenv("RT_MTP");
        if (env && *env) mtp_path = env;
    }
    if (!no_mtp && mtp_path.empty() && !model.empty()) {
        const size_t slash = model.find_last_of('/');
        const std::string dir = slash == std::string::npos ? "." : model.substr(0, slash);
        mtp_path = dir + "/qwen38_27b_mtp.rt4";
    }
    if (!mtp_path.empty() && mtp_json.empty()) mtp_json = mtp_path + ".json";
    if (!mtp_path.empty() && access(mtp_path.c_str(), R_OK) != 0) mtp_path.clear();
    if (mtp_path.empty()) mtp_json.clear();
    if (vision_path.empty()) {
        const char* env = getenv("RT_VISION_RT4");
        if (env && *env) vision_path = env;
    }
    if (vision_path.empty() && !model.empty()) {
        const size_t slash = model.find_last_of('/');
        const std::string dir = slash == std::string::npos ? "." : model.substr(0, slash);
        vision_path = dir + "/qwen38_27b_vision.rt4";
    }
    if (!vision_path.empty() && vision_json.empty()) vision_json = vision_path + ".json";
    if (!vision_path.empty() && access(vision_path.c_str(), R_OK) != 0) {
        vision_path.clear();
        vision_json.clear();
    }
    if (!dump.empty()) remove(dump.c_str());      // 每次运行重建 dump 文件
    Model m;
    m.stats = stats_flag;
    m.prof = getenv("RT_PROF") ? atoi(getenv("RT_PROF")) : 0;
    m.dbg_layer = dbg_layer;
    m.max_ctx = ctx;
    m.mtp_path = mtp_path;
    m.mtp_json = mtp_json;
    m.mtp_n = std::max(0, std::min(3, mtp_n));
    m.dump_file = dump;
    m.dump_layers = dumplayers.empty() ? nullptr : &dumplayers;
    m.T_MAX = engine ? 512 : (int)ids.size();
    auto t0 = std::chrono::steady_clock::now();
    m.init(model, json);
    if (!vision_path.empty()) {
        m.vm.init(vision_path, vision_json);
        m.vision_on = true;
    }
    auto t1 = std::chrono::steady_clock::now();
    m.reset_state();
    if (engine) {
        printf("READY %.1f\n", std::chrono::duration<double>(t1 - t0).count());
        fflush(stdout);
        engine_loop(m);
        return 0;
    }
    m.forward(ids.data(), (int)ids.size(), true);
    CK(hipDeviceSynchronize());
    auto t2 = std::chrono::steady_clock::now();
    std::vector<float> logits(248320);
    CK(hipMemcpy(logits.data(), m.d.logits, logits.size() * 4, hipMemcpyDeviceToHost));
    // top-k
    std::vector<int> idx(logits.size());
    for (size_t i = 0; i < idx.size(); i++) idx[i] = (int)i;
    std::partial_sort(idx.begin(), idx.begin() + topk, idx.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });
    printf("前向完成：%zu token，加载 %.1fs，前向 %.2fs\n",
           ids.size(),
           std::chrono::duration<double>(t1 - t0).count(),
           std::chrono::duration<double>(t2 - t1).count());
    printf("top-%d logits：", topk);
    for (int i = 0; i < topk; i++) printf(" %d(%.3f)", idx[i], logits[idx[i]]);
    printf("\n");
    return 0;
}
