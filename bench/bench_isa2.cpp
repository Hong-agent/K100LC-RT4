// K100_LC (gfx926) arithmetic-ceiling probe #2.
//
// Purpose: determine the true MMA-less compute peak of this chip by measuring
// every usable packed-dot instruction at matched ILP, plus a read-bandwidth
// kernel with proper memory-level parallelism.
//
//   v_dot4_i32_i8   : 4 int8  MACs per instruction
//   v_dot8_i32_i4   : 8 int4  MACs per instruction
//   v_dot2_i32_i16  : 2 int16 MACs per instruction
//   v_dot2_f32_f16  : 2 f16   MACs per instruction
//   v_pk_fma_f16    : 2 f16   FMAs per instruction (reference)
//
// build: hipcc -O3 --offload-arch=gfx926 -o bench_isa2 bench_isa2.cpp
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    printf("HIP ERR %s @%d: %s\n", #x, __LINE__, hipGetErrorString(e_)); exit(1);} } while (0)

// ---------------- dot8_i32_i4 ----------------
__global__ void k_dot8_i4(int* __restrict__ out, int iters) {
    int a[16], b[2];
#pragma unroll
    for (int i = 0; i < 16; i++) a[i] = threadIdx.x + i;
    b[0] = 0x11111111 * (1 + (threadIdx.x & 1));
    b[1] = 0x22222222 * (1 + (threadIdx.x & 3));
    for (int k = 0; k < iters; k++) {
#pragma unroll
        for (int i = 0; i < 16; i++)
            asm volatile("v_dot8_i32_i4 %0, %1, %2, %0\n" : "+v"(a[i]) : "v"(b[0]), "v"(b[1]));
    }
    int s = 0;
#pragma unroll
    for (int i = 0; i < 16; i++) s += a[i];
    if (s == 0x7ABCDEF1) out[0] = s;
}

// ---------------- dot4_i32_i8 ----------------
__global__ void k_dot4_i8(int* __restrict__ out, int iters) {
    int a[16], b[2];
#pragma unroll
    for (int i = 0; i < 16; i++) a[i] = threadIdx.x + i;
    b[0] = 0x01010101 * (1 + (threadIdx.x & 1));
    b[1] = 0x01010101 * (2 + (threadIdx.x & 1));
    for (int k = 0; k < iters; k++) {
#pragma unroll
        for (int i = 0; i < 16; i++)
            asm volatile("v_dot4_i32_i8 %0, %1, %2, %0\n" : "+v"(a[i]) : "v"(b[0]), "v"(b[1]));
    }
    int s = 0;
#pragma unroll
    for (int i = 0; i < 16; i++) s += a[i];
    if (s == 0x7ABCDEF1) out[0] = s;
}

// ---------------- dot2_i32_i16 ----------------
__global__ void k_dot2_i16(int* __restrict__ out, int iters) {
    int a[16], b[2];
#pragma unroll
    for (int i = 0; i < 16; i++) a[i] = threadIdx.x + i;
    b[0] = 0x00010001 * (1 + (threadIdx.x & 1));
    b[1] = 0x00020002 * (1 + (threadIdx.x & 1));
    for (int k = 0; k < iters; k++) {
#pragma unroll
        for (int i = 0; i < 16; i++)
            asm volatile("v_dot2_i32_i16 %0, %1, %2, %0\n" : "+v"(a[i]) : "v"(b[0]), "v"(b[1]));
    }
    int s = 0;
#pragma unroll
    for (int i = 0; i < 16; i++) s += a[i];
    if (s == 0x7ABCDEF1) out[0] = s;
}

// ---------------- dot2_f32_f16 ----------------
// dst is a single f32 accumulator; each instruction folds a 2x f16 dot into it.
__global__ void k_dot2_f16(float* __restrict__ out, unsigned b0, unsigned b1, int iters) {
    float a[16];
#pragma unroll
    for (int i = 0; i < 16; i++) a[i] = threadIdx.x * 1e-6f + i * 1e-5f;
    for (int k = 0; k < iters; k++) {
#pragma unroll
        for (int i = 0; i < 16; i++)
            asm volatile("v_dot2_f32_f16 %0, %1, %2, %0\n" : "+v"(a[i]) : "v"(b0), "v"(b1));
    }
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < 16; i++) s += a[i];
    if (s == 1234567.891f) out[0] = s;
}

// ---------------- bandwidth: N independent loads in flight, coalesced ----------------
// Thread t touches in[base + j*stride + k*stride*ILP]; within one instruction
// consecutive threads read consecutive float4, so every access is coalesced.
#ifndef RD_ILP
#define RD_ILP 8
#endif
__global__ void k_read_ilp8(const float4* __restrict__ in, float* __restrict__ out, size_t n4) {
    size_t base = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    float4 acc[RD_ILP];
#pragma unroll
    for (int j = 0; j < RD_ILP; j++) acc[j] = make_float4(0.f, 0.f, 0.f, 0.f);
    size_t i = base;
    for (; i + (RD_ILP - 1) * stride < n4; i += (size_t)RD_ILP * stride) {
#pragma unroll
        for (int j = 0; j < RD_ILP; j++) {
            float4 v = in[i + (size_t)j * stride];
            acc[j].x += v.x; acc[j].y += v.y; acc[j].z += v.z; acc[j].w += v.w;
        }
    }
    for (; i < n4; i += stride) {
        float4 v = in[i];
        acc[0].x += v.x; acc[0].y += v.y; acc[0].z += v.z; acc[0].w += v.w;
    }
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < RD_ILP; j++) s += acc[j].x + acc[j].y + acc[j].z + acc[j].w;
    if (s == 1234567.891f) out[0] = s;
}

// ---------------- v_dot8_i32_i4 semantics ----------------
// Resolve what the instruction actually computes: operand width, accumulator
// layout, and whether products are signed. Needed before any kernel relies on it.
__global__ void k_dot8_semantics(int* __restrict__ out) {
    // src0 nibbles 0..7 (low nibble first), src1 all ones, dst starts at 0.
    int s0 = 0x76543210;
    int s1 = 0x11111111;
    int acc = 0;
    asm volatile("v_dot8_i32_i4 %0, %1, %2, %0\n" : "+v"(acc) : "v"(s0), "v"(s1));
    if (threadIdx.x == 0) {
        out[0] = acc;                       // as one int32
        out[1] = (int)((unsigned)acc);      // same, raw
    }

    // signed check: nibble 0xF should be -1 if signed
    int t0 = 0xFFFFFFFF;
    int t1 = 0x11111111;
    int acc2 = 0;
    asm volatile("v_dot8_i32_i4 %0, %1, %2, %0\n" : "+v"(acc2) : "v"(t0), "v"(t1));
    if (threadIdx.x == 1) out[2] = acc2;
}

static void props() {
    hipDeviceProp_t p; CK(hipGetDeviceProperties(&p, 0));
    printf("name=%s  gcnArchName=%s  warpSize=%d  CU=%d\n",
           p.name, p.gcnArchName, p.warpSize, p.multiProcessorCount);
    printf("clockRate=%.0f MHz  memClock=%.0f MHz  busWidth=%d bit  L2=%d KB  sharedPerBlock=%zu KB\n",
           p.clockRate / 1000.0, p.memoryClockRate / 1000.0, p.memoryBusWidth,
           p.l2CacheSize / 1024, p.sharedMemPerBlock / 1024);
    printf("theory BW = %.0f GB/s\n",
           p.memoryClockRate * 1000.0 * 2.0 * (p.memoryBusWidth / 8.0) / 1e9);
}

template <typename K>
static double time_kernel(K launch, int reps) {
    hipEvent_t s, e; CK(hipEventCreate(&s)); CK(hipEventCreate(&e));
    launch(); CK(hipDeviceSynchronize());
    CK(hipEventRecord(s));
    for (int i = 0; i < reps; i++) launch();
    CK(hipEventRecord(e)); CK(hipEventSynchronize(e));
    float ms = 0; CK(hipEventElapsedTime(&ms, s, e));
    hipEventDestroy(s); hipEventDestroy(e);
    return ms / reps;
}

int main() {
    props();
    int cu = 0;
    { hipDeviceProp_t p; CK(hipGetDeviceProperties(&p, 0)); cu = p.multiProcessorCount; }

    const int iters = 20000;
    int* d_out; CK(hipMalloc(&d_out, 1 << 20));
    float* d_outf; CK(hipMalloc(&d_outf, 1 << 20));

    // ---- instruction-rate probes: identical shape, only the opcode differs ----
    // Try several occupancy configurations so a stalled config can't be
    // mistaken for a hardware limit.
    struct Cfg { int waves_per_cu; int block; const char* tag; };
    const Cfg cfgs[] = { {32, 256, "32w/256t"}, {16, 512, "16w/512t"}, {8, 1024, "8w/1024t"} };

    for (const Cfg& c : cfgs) {
        int grid = cu * c.waves_per_cu;
        size_t threads = (size_t)grid * c.block;
        printf("\n=== config %s (grid=%d block=%d) ===\n", c.tag, grid, c.block);

        double ms = time_kernel([&] { k_dot8_i4<<<grid, c.block>>>(d_out, iters); }, 3);
        printf("  v_dot8_i32_i4  : %7.1f TMACs/s  (%8.1f TOPS-equiv)\n",
               threads * 16.0 * 8.0 * iters / (ms * 1e-3) / 1e12,
               threads * 16.0 * 16.0 * iters / (ms * 1e-3) / 1e12);

        ms = time_kernel([&] { k_dot4_i8<<<grid, c.block>>>(d_out, iters); }, 3);
        printf("  v_dot4_i32_i8  : %7.1f TMACs/s  (%8.1f TOPS-equiv)\n",
               threads * 16.0 * 4.0 * iters / (ms * 1e-3) / 1e12,
               threads * 16.0 * 8.0 * iters / (ms * 1e-3) / 1e12);

        ms = time_kernel([&] { k_dot2_i16<<<grid, c.block>>>(d_out, iters); }, 3);
        printf("  v_dot2_i32_i16 : %7.1f TMACs/s  (%8.1f TOPS-equiv)\n",
               threads * 16.0 * 2.0 * iters / (ms * 1e-3) / 1e12,
               threads * 16.0 * 4.0 * iters / (ms * 1e-3) / 1e12);

        ms = time_kernel([&] { k_dot2_f16<<<grid, c.block>>>(d_outf, 0x3c003c00u,
                                                             0x3c003c00u, iters); }, 3);
        printf("  v_dot2_f32_f16 : %7.1f TMACs/s  (%8.1f TFLOPS-equiv)\n",
               threads * 16.0 * 2.0 * iters / (ms * 1e-3) / 1e12,
               threads * 16.0 * 4.0 * iters / (ms * 1e-3) / 1e12);
    }
    CK(hipFree(d_out)); CK(hipFree(d_outf));

    // ---- dot8 semantics ----
    {
        int* d_sem; int h_sem[4] = { -999, -999, -999, -999 };
        CK(hipMalloc(&d_sem, 64));
        CK(hipMemset(d_sem, 0, 64));
        k_dot8_semantics<<<1, 64>>>(d_sem);
        CK(hipDeviceSynchronize());
        CK(hipMemcpy(h_sem, d_sem, sizeof(h_sem), hipMemcpyDeviceToHost));
        printf("\n=== v_dot8_i32_i4 semantics ===\n");
        printf("  src0=0x76543210 (nibbles 0..7) src1=0x11111111 -> dst=%d (0x%08x)\n",
               h_sem[0], (unsigned)h_sem[0]);
        printf("  src0=0xFFFFFFFF src1=0x11111111 -> dst=%d\n", h_sem[2]);
        printf("  (expect 28 for unsigned sum of 0..7; -8 if nibbles are signed)\n");
        CK(hipFree(d_sem));
    }

    // ---- read bandwidth with real MLP ----
    {
        size_t bytes = 8ull << 30;
        float *dA, *dB;
        CK(hipMalloc(&dA, bytes)); CK(hipMalloc(&dB, 4096));
        CK(hipMemset(dA, 1, bytes));
        size_t n4 = bytes / sizeof(float4);
        printf("\n=== HBM read (MLP-tuned) ===\n");
        for (const Cfg& c : cfgs) {
            int grid = cu * c.waves_per_cu;
            double ms = time_kernel([&] { k_read_ilp8<<<grid, c.block>>>(
                (const float4*)dA, dB, n4); }, 5);
            printf("  %s : %7.1f GB/s\n", c.tag, bytes / (ms * 1e-3) / 1e9);
        }
        CK(hipFree(dA)); CK(hipFree(dB));
    }
    return 0;
}
