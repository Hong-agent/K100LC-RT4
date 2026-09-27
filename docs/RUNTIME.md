# 自研运行时：全模型前向（进行中）

> **2026-09-28 更新：已经能对话了。** 端到端跑通，`scripts/serve.py` 起 OpenAI 兼容接口，
> 中文/英文生成都通顺（预填充 273 t/s、解码 12 t/s）。这一轮修掉 7 个 bug（其中一个致命：
> 转换器把 449 张非量化张量写错位，模型因此完全跑偏），完整排错记录见
> **[QUALITY-FIX.md](QUALITY-FIX.md)**，那份文档比本文件新，先看它。
> 本文件下面关于「logits 太平、NaN」的内容都是修复前的历史记录。

## 修复后的跑法（推荐）

```bash
bash scripts/serve.sh                                  # OpenAI 兼容服务 :8080
bash scripts/dsh.sh 'python3 scripts/chat.py --prompt "你好" --n 128'   # 命令行对话
bash scripts/dsh.sh 'python3 scripts/bench_rt.py'      # 预填充/解码速度
```

目标（用户指定）：**先做到「真的能对话、能接 API」，性能优化放第二步。**

代码：`src/`

| 文件 | 作用 |
|---|---|
| `src/model.cpp` | RT4 加载（mmap + 分块 H2D）、层表、全模型图、CLI |
| `src/kernels.h` | 运行时算子声明 |
| `src/k_new.hip` | 新写的算子（RMSNorm / RoPE / 量化 / 词嵌入 / KV 写入 / 卷积 / gated delta net …） |
| `src/k_gemm.hip` | 预填充 int4 GEMM 封装（复用 `kernels/gemm_core.h`） |
| `src/k_gemv.hip` | 解码 W4A4 GEMV 封装（复用 `kernels/gemv_w4a4_core.h`） |
| `src/k_fa.hip` | 注意力封装（复用 `kernels/flash_attn_core.h`） |
| `tools/ref_layer.py` | **单层 torch 参考实现**（权重从同一份 RT4 反量化，可模拟 int4 激活量化） |
| `tools/gguf_probe.py` / `tools/hf_vs_gguf.py` | 张量表探测与 HF↔GGUF 逐行比对 |

## 跑法

```bash
bash scripts/build_rt.sh                    # 编译 → build/rt
# 前向（直接给 token id），打印 top-k logits
docker run ... ./build/rt --model models/Qwen3.8-27B-NVFP4/rt4/qwen38_27b.rt4 \
  --json models/Qwen3.8-27B-NVFP4/rt4/qwen38_27b.rt4.json --ids 248044,9707,11,1879 --topk 5

# 逐层正确性对照（dump 中间层 → torch 参考逐元素比）
./build/rt ... --ids ... --dump /rt/build/d.bin --dump-layers -1,0,1,2,3
python3 tools/ref_layer.py /rt/build/d.bin <rt4> <json> 0
REF_QUANT=1 REF_G=128 python3 tools/ref_layer.py ... 0     # 参考端也模拟 int4 激活量化
```

其它开关：`--stats`（每层 x 的统计）、`--dbg-layer N`（该层的中间量统计）、`--ctx N`、
`--dump-layers` 里的 `900` 表示 dump 注意力层的 q_proj 输出。

## 架构（与 `transformers/models/qwen3_5/modeling_qwen3_5.py` 逐行对齐）

64 层，`linear_attention`（gated delta net）每 4 层插一个 `full_attention`：

```
每层：x += mixer(rmsnorm(x, input_layernorm, **零中心** (1+w)))
      x += mlp(rmsnorm(x, post_attention_layernorm)) , mlp = down(silu(gate)*up)
full_attention：q_proj→每头[q(256),gate(256)] → q/k 各 RMSNorm(256) → 部分 RoPE(前 64 维)
                → 注意力(scale 1/sqrt(256)) → ×sigmoid(gate) → o_proj
linear_attention：in_proj_qkv(10240) → 因果卷积(核 4)+SiLU → 切 q(16×128)/k(16×128)/v(48×128)
                → q/k 各头 L2 归一 → **gated delta net 递推**（q 先乘 1/sqrt(128)）
                → 门控 RMSNorm(×silu(z)) → out_proj
```

要点（都踩过坑）：

* 普通 RMSNorm 的权重是**零中心**的，前向要乘 `(1+w)`；门控 RMSNorm（SSM 那个）不是。
* GDN 的 q/k 头数 16、v 头数 48，配对方式是 HF 的 `repeat_interleave` ⇒ **v 头 j 用 k 头 j//3**
  （llama.cpp 用的是 `j%16`，与 HF 不一致；我用 HF/vllm 语义，已用 `tools/hf_vs_gguf.py`
  确认 GGUF 未做头重排）。
* GDN 递推：`s = s*exp(g); kv = Σ_i s[i][j]k[i]; d = β(v−kv); s += k⊗d; o = Σ_i s[i][j]q[i]`，
  状态按列切开放在寄存器里（一线程一列，列间独立）。
* V 的 KV cache 按「沿 key 转置打包」存（`[kv/8][256]`），预填充与解码共用。

## 已确认修好的 bug（都留了注释）

1. 词嵌入、以及所有权重的**尺度是 f16**——按 f32 读会把结果搞成 0/NaN。
2. `k_gemv_f32`（in_proj_a/b 用）漏了 `blockIdx.y`，只算了第 0 个 token。
3. GDN 漏了 q 的 `1/sqrt(head_dim)` 缩放（输出大 11 倍）。
4. 注意力漏了 `1/sqrt(head_dim)` 的 score 缩放（折进 Q）。
5. FlashAttention 内核用 `n_q` 算头内偏移，传真实行数会让各头输出互相覆盖 ⇒ 传补齐到 64 的行数。
6. 解码 GEMV 在 K 不是步长整数倍时越界读激活尺度（`0*NaN=NaN`）——已夹住。

## 当前状态（2026-09-27，未完成）

**能跑通**：加载 13.91GB RT4（mmap+分块 H2D，宿主内存 <1GB）、64 层前向、打印 top-k logits；
T=8 走 GEMM 路径、T≤4 走 GEMV 路径都不崩、无 NaN（层内插值统计正常）。

**但生成质量还不达标**：logits 太平（top-1 只有 ~7.8，正常应该 15+），说明还有实现错误。
已有的证据：

* 单层对照（`tools/ref_layer.py`，层 0/1/3）：内核 vs 参考的相对 RMS 在
  「参考端也模拟 int4 激活量化」时约 **5%~8%**，其中
  * 逐阶段统计（qkv/z/conv/a/beta/g/core/norm/out_proj）都对得上（1~3%），
  * 剩下 ~5% 的来源还没定位（含注意力层 ~50% 的对不上，说明注意力路径仍有问题）。
* T≤4（GEMV 路径）在隐藏态最后一维（dim 5119）出现 NaN，来源未定位
  （已排除：权重/尺度 NaN、设备副本与文件不一致、哨兵没被写）。

## 下一步（按优先级）

1. **定位注意力路径**：用 `--dump-layers ...900` 把 q_proj 输出 dump 出来，
   与 `tools/ref_layer.py` 的逐步中间量对齐（q_norm / rope / attn 输出 / gate）；
   目前已知 gate 半段与参考差 ~2.5 倍，先查这里。
2. **定位 GEMV 的最后一维 NaN**：写一个最小复现（对 in_proj_z 用全 1 输入跑
   `k_gemv_w4a4`，看是否复现），再二分（权重/尺度/激活量化/归约）。
3. 单层对齐到 <1e-3 之后，再上：
   * 分词器（`tokenizer.json` 的 BPE）+ chat 模板（`chat_template.jinja`）；
   * 采样器 + KV/GDN 状态的增量解码（`seq_len` 已经在跑，需补解码路径的验收）；
   * OpenAI 兼容 HTTP 服务（接口与常见客户端一致，见 `scripts/serve.py`）。
4. 与 **vllm**（DTK 镜像自带，且是厂商为本模型做的实现）同 prompt 对 logits，
   而不是只对 llama.cpp——因为 llama.cpp 的 GDN 头映射与 HF 不同。
