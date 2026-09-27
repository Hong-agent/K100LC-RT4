// Ground-truth HBM bandwidth probe for K100_LC (gfx926).
//
// The generic read kernel tops out near 450-470 GB/s, which is only ~52% of the
// 896 GB/s the driver's clock/bus numbers imply. Before accepting that as the
// hardware limit, cross-check against the driver's own copy path and against
// several distinct access shapes.
//
// build: hipcc -O3 --offload-arch=gfx926 -o bench_bw bench_bw.cpp
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    printf("HIP ERR %s @%d: %s\n", #x, __LINE__, hipGetErrorString(e_)); exit(1);} } while (0)

// plain grid-stride read, 128-bit
__global__ void k_read128(const float4* __restrict__ in, float* __restrict__ out, size_t n4) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (; i < n4; i += stride) {
        float4 v = in[i];
        acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
    float s = acc.x + acc.y + acc.z + acc.w;
    if (s == 1234567.891f) out[0] = s;
}

// 256-bit via two independent float4 streams (unaligned pair per thread)
__global__ void k_read256(const float4* __restrict__ in, float* __restrict__ out, size_t n4) {
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    float4 a = make_float4(0.f,0.f,0.f,0.f), b = a;
    size_t i = t * 2;
    for (; i + 1 < n4; i += stride * 2) {
        float4 v0 = in[i], v1 = in[i + 1];
        a.x += v0.x; a.y += v0.y; a.z += v0.z; a.w += v0.w;
        b.x += v1.x; b.y += v1.y; b.z += v1.z; b.w += v1.w;
    }
    float s = a.x+a.y+a.z+a.w + b.x+b.y+b.z+b.w;
    if (s == 1234567.891f) out[0] = s;
}

// byte-granular read (no vectorization) - exposes raw request throughput
__global__ void k_read8(const unsigned char* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    unsigned acc = 0;
    for (; i < n; i += stride) acc += in[i];
    if (acc == 0xDEADBEEFu) out[0] = 1.f;
}

__global__ void k_copy128(const float4* __restrict__ in, float4* __restrict__ out, size_t n4) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n4; i += stride) out[i] = in[i];
}

static double timeit(void (*fn)(void*), void* arg, int reps) {
    hipEvent_t s, e; CK(hipEventCreate(&s)); CK(hipEventCreate(&e));
    fn(arg); CK(hipDeviceSynchronize());
    CK(hipEventRecord(s));
    for (int i = 0; i < reps; i++) fn(arg);
    CK(hipEventRecord(e)); CK(hipEventSynchronize(e));
    float ms = 0; CK(hipEventElapsedTime(&ms, s, e));
    hipEventDestroy(s); hipEventDestroy(e);
    return ms / reps;
}

struct ReadArgs { const void* in; float* out; size_t n; int grid; int block; int variant; };

static void launch_read(void* p) {
    ReadArgs* a = (ReadArgs*)p;
    switch (a->variant) {
        case 0: k_read128<<<a->grid, a->block>>>((const float4*)a->in, a->out, a->n / 16); break;
        case 1: k_read256<<<a->grid, a->block>>>((const float4*)a->in, a->out, a->n / 16); break;
        case 2: k_read8<<<a->grid, a->block>>>((const unsigned char*)a->in, a->out, a->n); break;
        case 3: k_copy128<<<a->grid, a->block>>>((const float4*)a->in, (float4*)a->out, a->n / 16); break;
    }
}

int main() {
    hipDeviceProp_t p; CK(hipGetDeviceProperties(&p, 0));
    int cu = p.multiProcessorCount;
    printf("CU=%d  clock=%.0f MHz  memClock=%.0f MHz  bus=%d bit  theory=%.0f GB/s\n",
           cu, p.clockRate / 1000.0, p.memoryClockRate / 1000.0, p.memoryBusWidth,
           p.memoryClockRate * 1000.0 * 2.0 * (p.memoryBusWidth / 8.0) / 1e9);

    const size_t bytes = 8ull << 30;
    void *dA, *dB;
    CK(hipMalloc(&dA, bytes));
    CK(hipMalloc(&dB, bytes));
    CK(hipMemset(dA, 0x5A, bytes));
    CK(hipMemset(dB, 0x00, bytes));
    float* dOut; CK(hipMalloc(&dOut, 4096));

    // sanity: is the data real?
    unsigned char probe;
    CK(hipMemcpy(&probe, dA, 1, hipMemcpyDeviceToHost));
    printf("first byte of A = 0x%02x\n", probe);

    printf("\n--- driver copy (reference) ---\n");
    {
        hipEvent_t s, e; CK(hipEventCreate(&s)); CK(hipEventCreate(&e));
        CK(hipMemcpy(dB, dA, bytes, hipMemcpyDeviceToDevice)); CK(hipDeviceSynchronize());
        CK(hipEventRecord(s));
        for (int i = 0; i < 3; i++) CK(hipMemcpy(dB, dA, bytes, hipMemcpyDeviceToDevice));
        CK(hipEventRecord(e)); CK(hipEventSynchronize(e));
        float ms = 0; CK(hipEventElapsedTime(&ms, s, e)); ms /= 3;
        printf("  hipMemcpy D2D     : %7.1f GB/s moved (%.1f GB/s r+w)\n",
               bytes / (ms * 1e-3) / 1e9, 2.0 * bytes / (ms * 1e-3) / 1e9);
        hipEventDestroy(s); hipEventDestroy(e);
    }

    printf("\n--- kernel read variants (8 GiB, 5 reps) ---\n");
    const int blocks[] = { 256, 512, 1024 };
    const int grids[] = { 8, 16, 32, 64 };
    for (int variant = 0; variant <= 2; variant++) {
        const char* nm = variant == 0 ? "read128" : variant == 1 ? "read256" : "read8  ";
        double best = 0; int bg = 0, bb = 0;
        for (int gi = 0; gi < 4; gi++) for (int bi = 0; bi < 3; bi++) {
            ReadArgs a{ dA, dOut, bytes, cu * grids[gi], blocks[bi], variant };
            double ms = timeit(launch_read, &a, 5);
            double gbs = bytes / (ms * 1e-3) / 1e9;
            if (gbs > best) { best = gbs; bg = cu * grids[gi]; bb = blocks[bi]; }
        }
        printf("  %s : best %7.1f GB/s  (grid=%d block=%d)\n", nm, best, bg, bb);
    }

    printf("\n--- copy kernel (8 GiB, 3 reps) ---\n");
    {
        double best = 0; int bg = 0, bb = 0;
        for (int gi = 0; gi < 4; gi++) for (int bi = 0; bi < 3; bi++) {
            ReadArgs a{ dA, (float*)dB, bytes, cu * grids[gi], blocks[bi], 3 };
            double ms = timeit(launch_read, &a, 3);
            double gbs = bytes / (ms * 1e-3) / 1e9;
            if (gbs > best) { best = gbs; bg = cu * grids[gi]; bb = blocks[bi]; }
        }
        printf("  copy128: best %7.1f GB/s moved (%.1f GB/s r+w) (grid=%d block=%d)\n",
               best, 2 * best, bg, bb);
    }

    printf("\n--- fine sweep: read128, reps=10 ---\n");
    {
        const int blks[] = { 128, 192, 256, 320, 384, 512 };
        const int mult[] = { 2, 3, 4, 6, 8, 12, 16 };
        double best = 0; int bg = 0, bb = 0;
        for (int mi = 0; mi < 7; mi++) {
            printf("  %3dw/CU :", mult[mi]);
            for (int bi = 0; bi < 6; bi++) {
                ReadArgs a{ dA, dOut, bytes, cu * mult[mi], blks[bi], 0 };
                double ms = timeit(launch_read, &a, 10);
                double gbs = bytes / (ms * 1e-3) / 1e9;
                printf(" %4db=%6.1f", blks[bi], gbs);
                if (gbs > best) { best = gbs; bg = cu * mult[mi]; bb = blks[bi]; }
            }
            printf("\n");
        }
        printf("  BEST read: %.1f GB/s (grid=%d block=%d)  = %.0f%% of theory\n",
               best, bg, bb, 100.0 * best / (p.memoryClockRate * 1000.0 * 2.0 *
               (p.memoryBusWidth / 8.0) / 1e9));
    }
    return 0;
}
