// 探测这台机器上一个「什么都没干」的内核启动要多少时间 —— 用来判断解码慢是
// GPU 算得慢，还是 CPU 提交不过来（解码每 token 要发 ~1500 个内核）。
//
//   hipcc -O3 --offload-arch=gfx926 bench/launch_probe.cpp -o /tmp/lp && /tmp/lp
#include <hip/hip_runtime.h>
#include <cstdio>
#include <chrono>

__global__ void empty_k() {}
__global__ void tiny_k(float* p) { if (threadIdx.x == 0 && blockIdx.x == 0) p[0] = 1.f; }
__global__ void small_k(float* p, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = (float)i * 0.5f;
}

int main() {
    float* d; hipMalloc(&d, 1 << 22);
    const int N = 2000;
    hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
    auto run = [&](const char* tag, auto fn, int threads, int blocks) {
        hipDeviceSynchronize();
        hipEventRecord(a, 0);
        for (int i = 0; i < N; i++) fn(threads, blocks);
        hipEventRecord(b, 0);
        hipDeviceSynchronize();
        float ms = 0; hipEventElapsedTime(&ms, a, b);
        printf("%-28s %8.3f us/launch (GPU 侧)\n", tag, ms * 1000.0 / N);
        // CPU 侧提交耗时
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < N; i++) fn(threads, blocks);
        auto t1 = std::chrono::steady_clock::now();
        hipDeviceSynchronize();
        printf("%-28s %8.3f us/launch (CPU 提交)\n", tag,
               std::chrono::duration<double, std::micro>(t1 - t0).count() / N);
    };
    run("empty <<<1,32>>>", [&](int t, int b) { empty_k<<<b, t>>>(); }, 32, 1);
    run("empty <<<64,256>>>", [&](int t, int b) { empty_k<<<b, t>>>(); }, 256, 64);
    run("tiny <<<48,128>>>", [&](int t, int b) { tiny_k<<<b, t>>>(d); }, 128, 48);
    run("small 5120<<<40,256>>>", [&](int t, int b) { small_k<<<b, t>>>(d, 5120); }, 256, 40);
    run("small 5120<<<20,256>>>", [&](int t, int b) { small_k<<<b, t>>>(d, 5120); }, 256, 20);

    // 有依赖的链：每个内核都读上一个写的结果（模拟运行时的逐层串行）。
    // 这是解码每 token ~1300 个小内核的真实形态 —— 每次启动的「有效成本」。
    const int CH = 1000;
    float* bufa = d;
    hipDeviceSynchronize();
    hipEventRecord(a, 0);
    for (int i = 0; i < CH; i++) {
        small_k<<<20, 256>>>(bufa, 5120);
    }
    hipEventRecord(b, 0);
    hipDeviceSynchronize();
    float ms2 = 0; hipEventElapsedTime(&ms2, a, b);
    printf("%-28s %8.3f us/launch (相互独立)\n", "1000 × small 5120<<<20,256>>>",
           ms2 * 1000.0 / CH);
    // 真正的依赖：每个内核都要等前一个（写同一个缓冲）
    hipEventRecord(a, 0);
    for (int i = 0; i < CH; i++) small_k<<<20, 256>>>(bufa, 5120);
    hipEventRecord(b, 0);
    hipDeviceSynchronize();
    float ms3 = 0; hipEventElapsedTime(&ms3, a, b);
    printf("%-28s %8.3f us/launch (串行依赖)\n", "1000 × small 5120<<<20,256>>> 同址",
           ms3 * 1000.0 / CH);
    hipEventRecord(a, 0);
    for (int i = 0; i < CH; i++) empty_k<<<20, 256>>>();
    hipEventRecord(b, 0);
    hipDeviceSynchronize();
    float ms4 = 0; hipEventElapsedTime(&ms4, a, b);
    printf("%-28s %8.3f us/launch\n", "1000 × empty <<<20,256>>>", ms4 * 1000.0 / CH);
    // 内核之间夹 hipEventRecord（运行时 RT_PROF=1 就是这么干的）：量事件本身的
    // GPU 时间代价，决定分阶段计时的可信度。
    hipEvent_t ev[8];
    for (int i = 0; i < 8; i++) hipEventCreate(&ev[i]);
    hipEventRecord(a, 0);
    for (int i = 0; i < CH; i++) {
        small_k<<<20, 256>>>(bufa, 5120);
        hipEventRecord(ev[i % 8], 0);
        hipEventRecord(ev[(i + 1) % 8], 0);       // 每个作用域 2 个事件
    }
    hipEventRecord(b, 0);
    hipDeviceSynchronize();
    float ms5 = 0; hipEventElapsedTime(&ms5, a, b);
    printf("%-28s %8.3f us/(launch+2 events)\n", "1000 × small + 2×hipEventRecord",
           ms5 * 1000.0 / CH);
    return 0;
}
