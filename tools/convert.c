/* convert.c — NVFP4 / FP8 (compressed-tensors) safetensors -> RT4 runtime format
 *
 * 源：unsloth/Qwen3.8-27B-NVFP4 的 model.safetensors（单文件、mmap 流式处理，
 *     常驻内存只有几十 MB——本机只有 7GB 物理内存）。
 *
 * 目标格式 RT4（见 rt/FORMAT.md）：
 *   每个线性层权重 [N, K] 存成
 *     q: N * (K/2) 字节   int4 有符号、沿 K 每 2 个一组打包（低半字节在前）
 *     s: N * (K/128) * 2 字节  f16 分组尺度（每 128 个 K 一个）
 *   —— 也就是「统一 int4 + 每 128 组 f16 尺度」，因为它能走本卡
 *      v_dot8_i32_i4（75.5 TMAC/s）；源里的 NVFP4(E2M1)+FP8 只能走
 *      v_dot4_i32_i8（37.9 TMAC/s），到不了 1000 t/s 预填充。
 *
 * 反量化公式（compressed-tensors 语义，已用 BF16 原模型的 rms 逐张量核对）：
 *   NVFP4: w = e2m1(code) * fp8_e4m3(block_scale) / weight_global_scale
 *          （块尺度在存储前已乘过 global_scale，反量化时要除回来；
 *            乘 vs 除的差别是 1e8 量级，用 BF16 原模型 rms 一测便知）
 *   FP8  : w = fp8_e4m3(code) * bf16(channel_scale)
 *
 * 编译: gcc -O2 -o convert convert.c -lm
 * 用法: convert <model.safetensors> <out.rt4> [--int8=lm_head,embed] [--skip-visual]
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>

#define MAX_TENSORS 8192
#define GRP 128            /* K 方向的分组大小 */

typedef struct {
    char     name[192];
    char     dtype[16];
    int      ndim;
    int64_t  shape[4];
    int64_t  off;          /* 数据段内偏移 */
    int64_t  bytes;
    int      used;         /* 已被转换/已处理 */
} tensor_t;

static tensor_t  g_t[MAX_TENSORS];
static int       g_n;
static const uint8_t *g_data;

/* ---------------- fp8 / e2m1 量化 LUT ---------------- */
static float f8_lut[256];
static float e2m1_lut[16];

static float bf16_to_f32(uint16_t h) {
    uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

/* f16 -> f32（含非规格化） */
static float half_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp = (h >> 10) & 0x1F, mant = h & 0x3FF, f;
    if (exp == 0) {
        if (mant == 0) f = sign;
        else {
            int e = -1;
            uint32_t m = mant;
            while (!(m & 0x400)) { m <<= 1; e++; }
            f = sign | ((uint32_t)(127 - 15 - e) << 23) | ((m & 0x3FF) << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &f, 4);
    return out;
}

static void build_luts(void) {
    for (int i = 0; i < 256; i++) {
        int s = (i >> 7) & 1, e = (i >> 3) & 0xF, m = i & 7;
        float v;
        if (e == 0)      v = ldexpf((float)m / 8.0f, -6);
        else if (e == 15 && m == 7) v = NAN;
        else             v = ldexpf(1.0f + (float)m / 8.0f, e - 7);
        f8_lut[i] = s ? -v : v;
    }
    for (int c = 0; c < 16; c++) {
        int s = (c >> 3) & 1, e = (c >> 1) & 3, m = c & 1;
        float v = (e == 0) ? (m ? 0.5f : 0.0f) : ((1.0f + 0.5f * m) * (float)(1 << (e - 1)));
        e2m1_lut[c] = s ? -v : v;
    }
}

/* fp32 -> f16（就近偶数舍入），避免依赖编译器的 __fp16 支持 */
static uint16_t f32_to_f16(float x) {
    uint32_t u; memcpy(&u, &x, 4);
    uint32_t s = (u >> 16) & 0x8000;
    int32_t  e = (int32_t)((u >> 23) & 0xFF) - 127 + 15;
    uint32_t m = u & 0x7FFFFF;
    if (((u >> 23) & 0xFF) == 0xFF) return (uint16_t)(s | 0x7C00 | (m ? 0x200 : 0));
    if (e >= 0x1F) return (uint16_t)(s | 0x7C00);
    if (e <= 0) {
        if (e < -10) return (uint16_t)s;
        m |= 0x800000;
        int shift = 14 - e;
        uint32_t half = 1u << (shift - 1);
        uint32_t r = (m + half - 1 + ((m >> shift) & 1)) >> shift;
        return (uint16_t)(s | r);
    }
    uint32_t r = m + 0x1000 + ((m >> 13) & 1);
    if (r & 0x800000) { r = 0; e++; if (e >= 0x1F) return (uint16_t)(s | 0x7C00); }
    return (uint16_t)(s | ((uint32_t)e << 10) | (r >> 13));
}

/* ---------------- safetensors 头部扫描 ---------------- */
static const char *skip_ws(const char *p) {
    while (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r') p++;
    return p;
}

static const char *parse_string(const char *p, char *out, size_t cap) {
    p = skip_ws(p);
    if (*p != '"') return NULL;
    p++;
    size_t i = 0;
    while (*p && *p != '"') {
        if (*p == '\\' && p[1]) p++;
        if (i + 1 < cap) out[i++] = *p;
        p++;
    }
    if (*p != '"') return NULL;
    out[i] = 0;
    return p + 1;
}

static const char *parse_ll(const char *p, int64_t *out) {
    p = skip_ws(p);
    char *end;
    *out = strtoll(p, &end, 10);
    return end == p ? NULL : end;
}

static int parse_header(const char *json, int64_t json_len) {
    const char *p = json, *end = json + json_len;
    g_n = 0;
    while (p < end) {
        p = memchr(p, '"', end - p);
        if (!p) break;
        char name[192];
        const char *q = parse_string(p, name, sizeof(name));
        if (!q) break;
        q = skip_ws(q);
        if (*q != ':') { p = q; continue; }
        q = skip_ws(q + 1);
        if (*q != '{') { p = q; continue; }        /* __metadata__ 等 */
        if (g_n >= MAX_TENSORS) { fprintf(stderr, "too many tensors\n"); exit(1); }
        tensor_t *t = &g_t[g_n];
        memset(t, 0, sizeof(*t));
        snprintf(t->name, sizeof(t->name), "%s", name);
        const char *r = q + 1;
        int ok = 0;
        while (r < end && *r != '}') {
            r = skip_ws(r);
            if (*r == ',') { r++; continue; }
            char key[64];
            const char *r2 = parse_string(r, key, sizeof(key));
            if (!r2) break;
            r = skip_ws(r2);
            if (*r != ':') break;
            r = skip_ws(r + 1);
            if (!strcmp(key, "dtype")) {
                char v[16];
                r = parse_string(r, v, sizeof(v));
                if (!r) break;
                snprintf(t->dtype, sizeof(t->dtype), "%s", v);
            } else if (!strcmp(key, "shape")) {
                if (*r != '[') break;
                r++;
                t->ndim = 0;
                while (*r != ']' && t->ndim < 4) {
                    int64_t v;
                    r = parse_ll(r, &v);
                    if (!r) break;
                    t->shape[t->ndim++] = v;
                    r = skip_ws(r);
                    if (*r == ',') r++;
                }
                if (*r == ']') r++;
            } else if (!strcmp(key, "data_offsets")) {
                if (*r != '[') break;
                r++;
                int64_t a, b;
                r = parse_ll(r, &a);
                r = skip_ws(r);
                if (*r == ',') r++;
                r = parse_ll(r, &b);
                t->off = a;
                t->bytes = b - a;
                ok = 1;
                while (r < end && *r != ']') r++;
                if (*r == ']') r++;
            } else {
                /* 跳过未知字段 */
                int depth = 0, instr = 0;
                while (r < end) {
                    char c = *r;
                    if (instr) { if (c == '\\') r++; else if (c == '"') instr = 0; }
                    else if (c == '"') instr = 1;
                    else if (c == '[' || c == '{') depth++;
                    else if (c == ']' || c == '}') { if (depth == 0) break; depth--; }
                    else if (depth == 0 && c == ',') break;
                    r++;
                }
            }
        }
        if (ok) {
            t->used = 0;
            g_n++;
        }
        p = r;
    }
    return g_n;
}

static int find_tensor(const char *name) {
    for (int i = 0; i < g_n; i++)
        if (!strcmp(g_t[i].name, name)) return i;
    return -1;
}

/* ---------------- 输出与统计 ---------------- */
static FILE *g_out;
static int64_t g_write_pos;
static int g_int8_lm_head = 0;
static int g_int8_embed = 0;
static int g_skip_visual = 1;
static double g_err_sum;
static long long g_err_n;

typedef struct {
    char name[192];
    char kind[8];          /* i4 / i8 / f16 / f32 */
    int  ndim;
    int64_t shape[4];
    int64_t grp;
    int64_t q_off, s_off, nbytes;
    double  rms_err;       /* 反量化重构相对误差 */
    double  src_rms;       /* 源权重的 RMS（用于跟 BF16 原模型对照，验证反量化公式） */
} manifest_t;

static manifest_t g_man[MAX_TENSORS];
static int g_man_n;

static void man_add(const char *name, const char *kind, const tensor_t *t,
                    int64_t grp, int64_t q_off, int64_t s_off, double err, double src_rms) {
    manifest_t *m = &g_man[g_man_n++];
    memset(m, 0, sizeof(*m));
    snprintf(m->name, sizeof(m->name), "%s", name);
    snprintf(m->kind, sizeof(m->kind), "%s", kind);
    m->ndim = t->ndim;
    for (int i = 0; i < t->ndim; i++) m->shape[i] = t->shape[i];
    m->grp = grp;
    m->q_off = q_off;
    m->s_off = s_off;
    m->nbytes = (int64_t)g_write_pos - q_off;
    m->rms_err = err;
    m->src_rms = src_rms;
}

/* 张量级转换：源权重 -> f32 行块 -> RT4/RT8 */
static void convert_linear(const char *name, const tensor_t *wt, int bits,
                           const tensor_t *bs, const tensor_t *gs, const tensor_t *cs) {
    int64_t N = wt->shape[0], K = wt->shape[1];
    int is_fp8 = !strcmp(wt->dtype, "F8_E4M3");
    int is_pack = !strcmp(wt->dtype, "U8");
    int is_bf16 = !strcmp(wt->dtype, "BF16");
    if (is_pack) K *= 2;               /* U8 里一行是 K/2 字节（每字节 2 个 int4） */
    tensor_t logical = *wt;
    logical.shape[1] = K;

    int64_t rows_per_block = 512;
    float *buf = malloc(sizeof(float) * rows_per_block * K);
    if (!buf) { fprintf(stderr, "OOM for %s (%lld x %lld)\n", name, (long long)rows_per_block, (long long)K); exit(1); }

    double err = 0;
    int64_t q_off = 0, s_off = 0;
    int64_t nb = (bits == 4) ? (K / 2) : K;
    float qmax = (bits == 4) ? 7.0f : 127.0f;
    double sumsq = 0, ref = 0;

    uint8_t *qout = malloc(nb * N);
    uint16_t *sout = malloc(sizeof(uint16_t) * N * (K / GRP));
    if (!qout || !sout) { fprintf(stderr, "OOM qout %s\n", name); exit(1); }

    for (int64_t n0 = 0; n0 < N; n0 += rows_per_block) {
        int64_t nr = N - n0 < rows_per_block ? N - n0 : rows_per_block;
        /* 反量化到 buf */
        for (int64_t r = 0; r < nr; r++) {
            int64_t n = n0 + r;
            float *dst = buf + r * K;
            if (is_pack) {
                const uint8_t *codes = g_data + wt->off + n * (K / 2);
                const uint8_t *bsrow = g_data + bs->off + n * (K / 16);
                float gscale = gs ? *(const float *)(g_data + gs->off) : 1.0f;
                for (int64_t i = 0; i < K; i++) {
                    int c = (i & 1) ? (codes[i >> 1] >> 4) : (codes[i >> 1] & 0xF);
                    float blk = f8_lut[bsrow[i >> 4]];
                    dst[i] = e2m1_lut[c] * blk / gscale;
                }
            } else if (is_fp8) {
                const uint8_t *codes = g_data + wt->off + n * K;
                float cscale = cs ? bf16_to_f32(*(const uint16_t *)(g_data + cs->off + n * 2)) : 1.0f;
                for (int64_t i = 0; i < K; i++) dst[i] = f8_lut[codes[i]] * cscale;
            } else if (is_bf16) {
                const uint16_t *codes = (const uint16_t *)(g_data + wt->off + n * K * 2);
                for (int64_t i = 0; i < K; i++) dst[i] = bf16_to_f32(codes[i]);
            } else {
                fprintf(stderr, "unsupported dtype %s for %s\n", wt->dtype, name);
                exit(1);
            }
        }
        /* 量化 */
        int64_t rnb = (bits == 4) ? (K / 2) : K;
        if (getenv("RT_DEBUG") && n0 == 0) {
            fprintf(stderr, "[debug] %s off=%lld bs_off=%lld gs_off=%lld K=%lld N=%lld dtype=%s  src[0..7]=",
                    name, (long long)wt->off, (long long)(bs ? bs->off : -1),
                    (long long)(gs ? gs->off : -1), (long long)K, (long long)N, wt->dtype);
            for (int i = 0; i < 8; i++) fprintf(stderr, "%.5f ", buf[i]);
            fprintf(stderr, "\n");
        }
        for (int64_t r = 0; r < nr; r++) {
            const float *src = buf + r * K;
            uint8_t *qrow = qout + (n0 + r) * rnb;
            uint16_t *srow = sout + (n0 + r) * (K / GRP);
            for (int64_t g = 0; g < K / GRP; g++) {
                const float *blk = src + g * GRP;
                float amax = 0;
                for (int i = 0; i < GRP; i++) { float a = fabsf(blk[i]); if (a > amax) amax = a; }
                float s = amax > 0 ? amax / qmax : 1.0f;
                srow[g] = f32_to_f16(s);
                float inv = 1.0f / s;
                for (int i = 0; i < GRP; i++) {
                    float q = roundf(blk[i] * inv);
                    if (q >  qmax) q =  qmax;
                    if (q < -qmax) q = -qmax;
                    int qi = (int)q;
                    double e = blk[i] - (double)qi * s;
                    sumsq += e * e; ref += (double)blk[i] * blk[i];
                    int64_t idx = g * GRP + i;      /* 行内绝对下标（不是组内下标） */
                    if (bits == 4) {
                        if (idx & 1) qrow[idx >> 1] |= (uint8_t)((qi & 0xF) << 4);
                        else         qrow[idx >> 1]  = (uint8_t)(qi & 0xF);
                    } else {
                        qrow[idx] = (uint8_t)(int8_t)qi;
                    }
                }
            }
        }
    }
    err = ref > 0 ? sqrt(sumsq / ref) : 0.0;
    /* 顺序写盘 */
    fflush(g_out);
    q_off = g_write_pos;
    fwrite(qout, 1, nb * N, g_out);
    g_write_pos += nb * N;
    int64_t spad = (256 - (g_write_pos & 255)) & 255;
    if (spad) { static const char z[256] = {0}; fwrite(z, 1, spad, g_out); g_write_pos += spad; }
    s_off = g_write_pos;
    fwrite(sout, 1, sizeof(uint16_t) * N * (K / GRP), g_out);
    g_write_pos += sizeof(uint16_t) * N * (K / GRP);
    fflush(g_out);

    double src_rms = sqrt(ref / (double)(N * K));
    man_add(name, bits == 4 ? "i4" : "i8", &logical, GRP, q_off, s_off, err, src_rms);
    fprintf(stderr, "  %-64s [%6lld,%6lld] %s  relerr=%.4f  src_rms=%.5f\n", name,
            (long long)N, (long long)K, bits == 4 ? "i4" : "i8", err, src_rms);
    g_err_sum += sumsq; g_err_n++;
    free(qout); free(sout); free(buf);
}

/* 小张量原样搬运（f16/f32）。force_f32 用于 norm / A_log / dt_bias / conv1d —— 
 * 这些直接参与 exp、除法等非线性运算，半精度会给 SSM 带来可见误差。 */
static void convert_plain(const char *name, const tensor_t *t, int force_f32) {
    int64_t n = 1;
    for (int i = 0; i < t->ndim; i++) n *= t->shape[i];
    int to_f16 = !force_f32 && strcmp(t->dtype, "F32") != 0;
    fflush(g_out);
    int64_t off = g_write_pos;
    if (to_f16) {
        uint16_t *out = malloc(2 * n);
        if (!strcmp(t->dtype, "BF16")) {
            const uint16_t *src = (const uint16_t *)(g_data + t->off);
            /* bf16 -> f32 -> f16 */
            for (int64_t i = 0; i < n; i++) out[i] = f32_to_f16(bf16_to_f32(src[i]));
        } else if (!strcmp(t->dtype, "F16")) {
            memcpy(out, g_data + t->off, 2 * n);
        } else if (!strcmp(t->dtype, "F8_E4M3")) {
            for (int64_t i = 0; i < n; i++) out[i] = f32_to_f16(f8_lut[g_data[t->off + i]]);
        } else {
            fprintf(stderr, "plain: unsupported %s (%s)\n", t->dtype, name);
            free(out); exit(1);
        }
        fwrite(out, 2, n, g_out);
        g_write_pos += 2 * n;
        free(out);
        man_add(name, "f16", t, 0, off, 0, 0, 0);
    } else {
        /* 目标格式是 f32，但源可能是 BF16/F16/F8 —— **必须真的转换**。
         * 之前的版本直接 `fwrite(g_data + off, 4, n, ...)` 按 4 字节原样搬，
         * 结果 norm / A_log / dt_bias / conv1d / ssm_norm 这 ~350 张量全废：
         * 前半等于把相邻两个 bf16 当成一个 f32（低 16 位尾数错），后半直接读到
         * 下一张量的数据。逐张量与源对照 relerr 高达 0.5~5，模型因此完全跑偏
         * （残差流被放大 10 倍）。见 docs/CONVERT-VERIFY.md 的复查记录。 */
        if (!strcmp(t->dtype, "F32")) {
            fwrite(g_data + t->off, 4, n, g_out);
        } else {
            float *out = malloc(4 * n);
            if (!strcmp(t->dtype, "BF16")) {
                const uint16_t *src = (const uint16_t *)(g_data + t->off);
                for (int64_t i = 0; i < n; i++) out[i] = bf16_to_f32(src[i]);
            } else if (!strcmp(t->dtype, "F16")) {
                const uint16_t *src = (const uint16_t *)(g_data + t->off);
                for (int64_t i = 0; i < n; i++) out[i] = half_to_f32(src[i]);
            } else if (!strcmp(t->dtype, "F8_E4M3")) {
                for (int64_t i = 0; i < n; i++) out[i] = f8_lut[g_data[t->off + i]];
            } else {
                fprintf(stderr, "plain(f32): unsupported %s (%s)\n", t->dtype, name);
                free(out); exit(1);
            }
            fwrite(out, 4, n, g_out);
            free(out);
        }
        g_write_pos += 4 * n;
        man_add(name, "f32", t, 0, off, 0, 0, 0);
    }
    fflush(g_out);
}

static int starts_with(const char *s, const char *p) { return !strncmp(s, p, strlen(p)); }
static int ends_with(const char *s, const char *p) {
    size_t a = strlen(s), b = strlen(p);
    return a >= b && !strcmp(s + a - b, p);
}

/* 必须保持 f32 的小张量：RMSNorm 权重、SSM 的 A_log / dt_bias / conv1d 卷积核 */
static int keep_f32(const char *name) {
    return ends_with(name, "norm.weight") || ends_with(name, ".A_log") ||
           ends_with(name, ".dt_bias")    || ends_with(name, ".conv1d.weight") ||
           ends_with(name, ".bias");
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <in.safetensors> <out.rt4> [--all]\n", argv[0]); return 1; }
    const char *in_path = argv[1], *out_path = argv[2];
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--all")) g_skip_visual = 0;
        else if (!strcmp(argv[i], "--int8-lmhead")) g_int8_lm_head = 1;
        else if (!strcmp(argv[i], "--int8-embed")) g_int8_embed = 1;
    }
    build_luts();

    int fd = open(in_path, O_RDONLY);
    if (fd < 0) { perror(in_path); return 1; }
    struct stat st;
    fstat(fd, &st);
    const uint8_t *map = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) { perror("mmap"); return 1; }
    uint64_t hlen;
    memcpy(&hlen, map, 8);
    const char *json = (const char *)(map + 8);
    g_data = map + 8 + hlen;
    fprintf(stderr, "输入 %s  %.2f GB, 头部 %llu B\n", in_path, st.st_size / 1e9, (unsigned long long)hlen);

    int n = parse_header(json, hlen);
    fprintf(stderr, "解析到 %d 个张量\n", n);

    g_out = fopen(out_path, "wb+");
    if (!g_out) { perror(out_path); return 1; }

    /* 按文件顺序遍历（顺序读，NAS/磁盘友好） */
    int64_t last_off = -1;
    for (int pass = 0; pass < 2; pass++) {
        for (int i = 0; i < n; i++) {
            tensor_t *t = &g_t[i];
            if (t->used) continue;
            if (last_off >= 0 && pass == 0 && t->off < last_off) continue;
            if (g_skip_visual && starts_with(t->name, "model.visual.")) { t->used = 1; continue; }
            /* 伴侣张量（尺度/全局尺度）不单独输出，随父张量处理 */
            if (ends_with(t->name, ".weight_scale") || ends_with(t->name, "_global_scale") ||
                ends_with(t->name, ".k_scale") || ends_with(t->name, ".v_scale")) continue;

            char par[256];
            if (ends_with(t->name, ".weight_packed")) {
                /* NVFP4 MLP：找 weight_scale / weight_global_scale */
                /* 注意：截掉结尾的 "weight_packed"（保留前面那个点），不能用 %.*s 取前 N 个字符 */
                int keep = (int)(strlen(t->name) - strlen("weight_packed"));
                snprintf(par, sizeof(par), "%.*sweight_scale", keep, t->name);
                int si = find_tensor(par);
                snprintf(par, sizeof(par), "%.*sweight_global_scale", keep, t->name);
                int gi = find_tensor(par);
                if (si < 0 || gi < 0) { fprintf(stderr, "缺少伴侣: %s (si=%d gi=%d)\n", t->name, si, gi); continue; }
                /* 输出名去掉 .weight_packed 后缀，统一成 ...weight */
                char final[192];
                snprintf(final, sizeof(final), "%.*sweight", keep, t->name);
                convert_linear(final, t, 4, &g_t[si], &g_t[gi], NULL);
                t->used = 1; g_t[si].used = 1;
            } else if (!strcmp(t->dtype, "F8_E4M3") && t->ndim == 2) {
                snprintf(par, sizeof(par), "%s_scale", t->name);
                int si = find_tensor(par);
                int bits = 4;
                if (starts_with(t->name, "lm_head")) bits = g_int8_lm_head ? 8 : 4;
                convert_linear(t->name, t, bits, NULL, NULL, si >= 0 ? &g_t[si] : NULL);
                if (si >= 0) g_t[si].used = 1;
                t->used = 1;
            } else if (!strcmp(t->dtype, "BF16") && t->ndim == 2 &&
                       (ends_with(t->name, "embed_tokens.weight") || ends_with(t->name, "in_proj_a.weight") ||
                        ends_with(t->name, "in_proj_b.weight"))) {
                if (ends_with(t->name, "embed_tokens.weight")) {
                    convert_linear(t->name, t, g_int8_embed ? 8 : 4, NULL, NULL, NULL);
                } else {
                    convert_plain(t->name, t, 0);
                }
                t->used = 1;
            } else if (!strcmp(t->dtype, "BF16") && t->ndim == 2 &&
                       t->shape[0] * t->shape[1] >= 1000000) {
                /* 未量化的 BF16 大矩阵（例如 MTP 头的 mtp.fc / mtp.layers.*）也走 int4，
                 * 与主模型保持一致；小矩阵（in_proj_a/b 之类）留给 f16 分支。 */
                convert_linear(t->name, t, starts_with(t->name, "lm_head") && g_int8_lm_head ? 8 : 4,
                               NULL, NULL, NULL);
                t->used = 1;
            } else {
                convert_plain(t->name, t, keep_f32(t->name));
                t->used = 1;
            }
        }
        last_off = 0;
    }

    /* manifest */
    char mpath[512];
    snprintf(mpath, sizeof(mpath), "%s.json", out_path);
    FILE *mf = fopen(mpath, "w");
    fprintf(mf, "{\n \"format\": \"RT4-v1\",\n \"group\": %d,\n \"weight_file\": \"%s\",\n \"tensors\": [\n", GRP, out_path);
    for (int i = 0; i < g_man_n; i++) {
        manifest_t *m = &g_man[i];
        fprintf(mf, "  {\"name\": \"%s\", \"kind\": \"%s\", \"shape\": [", m->name, m->kind);
        for (int d = 0; d < m->ndim; d++) fprintf(mf, "%s%lld", d ? ", " : "", (long long)m->shape[d]);
        fprintf(mf, "], \"group\": %lld, \"q_off\": %lld, \"s_off\": %lld, \"nbytes\": %lld,"
                    " \"relerr\": %.5f, \"src_rms\": %.6g}%s\n",
                (long long)m->grp, (long long)m->q_off, (long long)m->s_off, (long long)m->nbytes,
                m->rms_err, m->src_rms, i + 1 < g_man_n ? "," : "");
    }
    fprintf(mf, " ]\n}\n");
    fclose(mf);
    fprintf(stderr, "\n写出 %s  %.2f GB, %d 个张量, manifest=%s\n",
            out_path, g_write_pos / 1e9, g_man_n, mpath);
    fclose(g_out);
    return 0;
}
