// 运行时（src/）用到的新算子声明。已有的三个现役内核走 kernels/*_core.h：
//   gemv_w4a4_core.h（W4A4 解码 GEMV）、flash_attn_core.h（int4 FlashAttention）
// 这里只放这个文件里新写的算子 + 两个复用内核的薄封装。
#pragma once
#include <cstdint>

typedef uint32_t u32;

// ---------------- 基础元素级 ----------------
void k_fill(float* y, float v, long long n);
void k_add_inplace(float* y, const float* x, long long n);              // y += x
void k_concat2(float* y, const float* a, const float* b, int rows, int D); // y[r] = [a[r] | b[r]]
void k_rmsnorm(float* y, const float* x, const float* w, int rows, int D, float eps, bool zero_centered);
void k_silu_mul(float* y, const float* gate, const float* up, long long n);   // y = silu(gate)*up
void k_sigmoid_mul(float* y, const float* x, const float* g, long long n);    // y = x*sigmoid(g)
void k_l2norm(float* x, int rows, int S, float eps);                    // 每 S 维一组，原地
// 全量 argmax（n ≤ ~2^31）：给 MTP 草稿的贪心采样用，省掉每步 1MB 的 logits D2H 拷贝
void k_argmax(const float* x, long long n, int* out);
// pos 为 nullptr 时用 pos0 + row 当位置（解码路径，省掉每层一次的临时分配与拷贝）
void k_rope(float* q, float* k, const int* pos, int pos0, int n_q, int n_kv, int nq_head,
            int nkv_head, int D, int rot, float theta);

// ---------------- 投影与量化 ----------------
// int4 权重 GEMV（W4A4）：x [rows][K] f32 → y [rows][N]
void k_gemv_w4a4(float* y, const u32* wq, const float* wsc, const float* x, int rows, int N, int K);
// int4 权重 GEMV（W4A8，dot4）：y [rows][N] = W[N][K] · x[rows][K]
void k_gemv_w4a8(float* y, const u32* wq, const float* wsc, const float* x, int rows, int N, int K);
// int4 权重 GEMM（W4A4，预填充）：a 已量化（组优先尺度）
void k_gemm_i4(float* c, const u32* wq, const float* wsc_gm, const u32* aq, const float* asc_gm,
               int M, int N, int K);
// int4 权重 GEMM（W4A8，预填充）：激活是 8bit，拆成两个 int4 数字各跑一遍再相加
void k_gemm_i4_a8(float* c, float* c2, const u32* wq, const float* wsc_gm, const u32* ah,
                  const u32* al, const float* sc, const float* sc16, int M, int N, int K);
// 激活量化：f32 [rows][K]（行间距 in_stride、行内偏移 in_off）→ int4 打包 dword
//（低半字节 = 偶数）+ 每 G 个一组尺度 [ng][rows]
void k_quant_rows(u32* q, float* sc, const float* x, int rows, int K, int G,
                  int in_stride = 0, int in_off = 0);
// 注意力的 Q 量化（一次做完所有头，只写 row < T 的行）：x [T][H*D] → qq [H][TP][D/8]
// + qs [H][D/G][TP]（布局与 k_quant_rows 逐个头的调用完全一致）
void k_attn_q_quant(u32* qq, float* qs, const float* x, int T, int H, int D, int G, int TP);
// 8bit 激活量化，拆成两个 int4 数字：q8 = 16*h + l（h,l ∈ [-8,7]，精确）
// 顺便输出两套尺度（sc 给 l 用，sc16 = 16*sc 给 h 用），GEMM 跑两遍相加即可。
void k_quant_rows_a8(u32* qh, u32* ql, float* sc, float* sc16, const float* x,
                     int rows, int K, int G, int in_stride = 0, int in_off = 0);
// 词嵌入：int4 表 [V][D] + 每行每 G 一组尺度 → f32 [n][D]
void k_embed(float* y, const u32* tbl, const unsigned short* sc, const int* ids, int n, int D, int G);

// ---------------- KV cache（格式见 kernels/flash_attn_core.h 顶部）----------------
// K：f32 [T][H][D] → 打包 [H][kv][D/8] + 每 (行,128 维组) 尺度 [H][2][kv]
void k_kv_append_k(u32* kc, float* ksc, const float* kf, int t0, int T, int H, int D, int G,
                   int kv_total);
// V：f32 [T][H][D] → 暂存 f32 stage[H][BG][D]，再把整块重量化成 [H][kv/8][D] + 尺度 [H][tile][D]
void k_kv_append_v(u32* vc, float* vsc, float* stage, const float* vf,
                   int t0, int T, int H, int D, int BG, int kv_total);

// ---------------- gated delta net ----------------
// 深度因果卷积（核 K=4）+ SiLU：in/out [T][C]，w [K][C]，state [K-1][C]（跨块续接）
void k_conv1d_silu(float* out, const float* in, const float* w, float* state, int T, int C, int K);
void k_conv_state_update(float* state, const float* in, const float* prev, int T, int C, int K);
// 与上一行相同，但把「每处理一个 token 之后的 state」另外写进 snap；
// snap 布局 [T][K-1][C]。MTP 验证批回滚时按接受长度选用其中一行。
void k_conv_state_update_snap(float* state, float* snap, const float* in, const float* prev,
                              int T, int C, int K);
// 上面两步 + a/b 两个 f32 GEMV 合成一个内核（解码热点，见 src/k_new.hip）
void k_ssm_ab_gate(float* gab, float* beta_out, float* gg, const float* wa, const float* wb,
                   const float* x, const float* dt, const float* alog, int T, int Hv, int K);
// 递推：q/k [T][Hk][D]、v [T][Hv][D]、g/beta [T][Hv]、state [Hv][D][D]
void k_gdn(float* out, const float* q, const float* k, const float* v, const float* g,
           const float* beta, float* state, int T, int Hk, int Hv, int D, int rep);
// 同上，但把「每处理一个 token 之后的递推状态」写进 snap（可为 nullptr）。
// snap 布局 [T][Hv][D][D]，与 state 的 [h][i][col] 布局一致。
void k_gdn_snap(float* out, const float* q, const float* k, const float* v, const float* g,
                const float* beta, float* state, float* snap, int T, int Hk, int Hv, int D,
                int rep);

// ---------------- 注意力（复用 kernels/flash_attn_core.h）----------------
// 传进来的 Q 已按 (行, 128 维组) 量化；KV 已在打包格式里；out [n_q][HD] f32
// n_q 是按 64 补齐后的行数（决定 Q/out 的头内偏移），n_row 是真实 query 行数；
// n_row == 1 时走单行解码内核（src/k_fa.hip 的 fa_decode_k），否则走预填充内核。
// pout/pmax/psum 是解码 split-KV 的暂存（[n_heads][max_split] 与 [..][HD]）。
void k_attention(float* out, const u32* qq, const float* qsc,
                 const u32* kc, const float* ksc, const u32* vc, const float* vsc,
                 int n_q, int n_row, int n_kv, int off, int n_tile, int n_heads, int ratio,
                 int kv_cap, float* pout, float* pmax, float* psum, int max_split);

// 验证批（MTP）多行解码注意力：T≤4 行 query 一次算完，每行看到的 KV 长度是
// n_kv0 + row（因果）。逐行算术与单行解码内核完全一致（同一 device 函数），
// 所以验证批与逐 token 解码逐位相同。pout/pmax/psum 需要按 **4 行**分配：
// pout 每行 n_heads*max_split*HD，pmax/psum 每行 n_heads*max_split。
// vstage_snap 是每行 append 之后的 fp32 V tile 暂存（布局 [row][KV][BG][D]），
// 用来复现「每行按自己的 prefix 量化最后一格 V」——这是与单行内核逐位相同的关键。
void k_attention_decode_rows(float* out, const u32* qq, const float* qsc,
                             const u32* kc, const float* ksc, const u32* vc,
                             const float* vsc, const float* vstage_snap,
                             size_t vsnap_row_stride, int n_q, int n_rows, int n_kv0,
                             int n_heads, int ratio, int kv_cap, float* pout, float* pmax,
                             float* psum, int max_split);

// ---------------- 其它补充算子 ----------------
// 抽头：src 每行是 n_head 个头，头间距 head_stride（不一定等于 D：
// 注意注意注意 q_proj 每个头是 [q(D), gate(D)]，头间距是 2D，头内偏移 0 或 D）
void k_gather_heads(float* dst, const float* src, int T, int n_head, int D,
                    int tok_stride, int head_off, int head_stride);
// 注意力输出是 [n_head][src_rows][D]（src_rows 是补齐到 64 的 query 行数），
// 塞回 [T][tok_stride] 的紧凑布局时行距必须用 src_rows，不能用 T。
void k_scatter_heads(float* dst, const float* src, int T, int n_head, int D,
                     int tok_stride, int head_off, int src_rows);
void k_scale(float* x, float s, long long n);
void k_rmsnorm_gated(float* y, const float* x, const float* w, const float* g, int rows, int D,
                     float eps);
void k_split_qkv(float* q, float* k, float* v, const float* c, int T, int qn, int kn, int vn);
