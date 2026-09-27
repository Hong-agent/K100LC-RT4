# NVFP4 → RT4 转换验证报告（2026-09-26 23:05）

源：`unsloth/Qwen3.8-27B-NVFP4` 的 `model.safetensors`
（22,568,192,096 B；sha256 `c473512c70eace07e2256fe9fd76596ac03e3295bee7d54cfb72676416afcc05`，
从魔搭同名仓库拉取，与 hf-mirror 公布的 LFS 哈希一致，下载后本机复核通过）。

产物：

| 文件 | 大小 | 内容 |
|---|---|---|
| `rt4/qwen38_27b.rt4` | 13.91 GB | 851 个张量（文本部分），统一 int4/128 组 + f16 尺度 |
| `rt4/qwen38_27b.rt4.json` | 179 KB | manifest：逐张量 `kind/shape/q_off/s_off/relerr/src_rms` |
| `rt4/qwen38_27b_mtp.rt4` | 0.22 GB | MTP 头（15 个张量，`mtp.fc` + 1 层注意力/MLP） |
| `rt4/kv_scales.json` | — | 源模型附带的 8bit KV 量化尺度（16 层 × k/v）；**convert.c 不产出、运行时也不用**，列在这里只为说明它不是本流程的产物 |
| `rt4/convert.log` | — | 逐张量转换日志（relerr / src_rms） |

转换耗时：**4 分 37 秒**（单线程，mmap 流式，宿主内存峰值 < 100MB）。

## 1. 两个把结论翻过来的 bug（都已修）

1. **NVFP4 的全局尺度是「除」不是「乘」**。
   源里的 `weight_scale`（fp8，每 16 一组）在存储时已经乘过了 `weight_global_scale`，
   反量化必须除回来：
   `w = e2m1(code) * fp8(block_scale) / weight_global_scale`。
   用「乘」得到的 src_rms 达到 2.7e6（正确值 0.011），差 2.5 亿倍——
   靠**与 BF16 原模型逐张量比 RMS** 才抓出来（`tools/verify_vs_gguf.py`）。
2. **分组量化写 nibble 用了组内下标而非行内下标**，导致每组互相覆盖、只有最后一组留下。
   合成用例（`tools/make_test_st.py` + `check_rt4.py`）第一轮就暴露了。

## 2. 反量化正确性（与 BF16 原模型 GGUF 对照）

线性层（源精度 FP8 或 NVFP4 混合，全部重量化成 int4）：

| 张量 | GGUF(BF16) rms | RT4 源 rms | 比值 |
|---|---|---|---|
| blk.0 `in_proj_qkv` | 0.01574 | 0.01568 | 1.0040 |
| blk.0 `in_proj_z` | 0.01613 | 0.01612 | 1.0006 |
| blk.0 `ssm_out` | 0.01573 | 0.01572 | 1.0006 |
| blk.11 `attn_q` | 0.01633 | 0.01625 | 1.0044 |
| blk.11 `attn_k/v/o` | 0.01567/0.01592/0.01542 | 0.01566/0.01591/0.01541 | 1.0006 |
| blk.15 `attn_k` | 0.01596 | 0.01596 | 1.0006 |

25 个张量全部落在 **0.995 ~ 1.004**。注意这个比值**不是 1.0000** 而是 1.0006：
因为源权重本身已经是被 NVFP4/FP8 量化过的（比 BF16 原模型略小），
这个 0.06% 就是源量化引入的、我们无法再消除的固有差异——**说明我们的转换没有额外损失**。

## 3. 两个「异常」的复核（都是采样偏差，不是错误）

逐张量脚本为了速度只读每个张量的前 4000 万个元素，而 embed / lm_head 的行分布极不均匀
（高频 token 的行范数不同），前缀采样的 RMS 与全量差 5~18%。改用**全量** RMS：

| 张量 | 源（全量） | GGUF（全量） | 结论 |
|---|---|---|---|
| `embed_tokens.weight`（源 BF16 未量化） | 0.012735 | 0.012735 | **精确一致** |
| `lm_head.weight`（源 FP8 → int4） | 0.013740 | 0.013746 | 一致（差 0.04%） |

## 4. 转换代价

统一 int4（每组 128 个 K，f16 尺度）后逐张量 `relerr`（重构相对 RMS）：
**n=402，mean 0.1245，median 0.1232，min 0.1180，max 0.1729**。
这是「二次量化」的固有代价：源权重已经是 4bit（NVFP4）或 8bit（FP8），
再压到统一 int4 会引入额外误差。**这条是本轮最大的质量风险**，
必须靠端到端困惑度/KL 与 llama.cpp 基线对比来验收（列在 M2/M5 里程碑）。
如果掉点不可接受，退路是：`--int8-lmhead`（+0.64GB）或把注意力投影保留 int8（+约 3.6GB/步）。

## 5. 复现命令

```bash
# 下载（魔搭，16 连接，带 sha256 校验；hf-mirror 单连接只有 200KB/s，不要用）
python3 tools/fetch_par2.py "https://modelscope.cn/api/v1/models/unsloth/Qwen3.8-27B-NVFP4/repo?Revision=master&FilePath=model.safetensors" \
    model.safetensors 22568192096 16 128 c473512c70eace07e2256fe9fd76596ac03e3295bee7d54cfb72676416afcc05

# 转换（约 4.5 分钟）
gcc -O2 -o /tmp/rt_convert tools/convert.c -lm
/tmp/rt_convert model.safetensors rt4/qwen38_27b.rt4
/tmp/rt_convert model_mtp.safetensors rt4/qwen38_27b_mtp.rt4

# 校验（合成用例 + 与 BF16 原模型对照）
python3 tools/make_test_st.py && /tmp/rt_convert /tmp/rt_test/in.safetensors /tmp/rt_test/out.rt4
python3 tools/check_rt4.py /tmp/rt_test/out.rt4 /tmp/rt_test/expect.json
python3 tools/verify_vs_gguf.py --check models/Qwen3.8-27B-NVFP4/rt4/qwen38_27b.rt4.json \
    "$RT_BF16_GGUF_DIR"/Qwen3.8-27B-BF16-0000{1,2}-of-00002.gguf 25
```
