# DTA Scalability Experiments — Model Family × Size Matrix

> **目的**: 测试 DTA 在不同开源模型家族和参数规模下的攻击效率，与 baseline 方法进行 ASR 和时间对比。
>
> **实验设置**: 直接在选定模型上做完整的DTA流程（采样+优化）
>
> **Judge**: 有害性、流畅性、有用性做综合评估。（使用GPT4o+一个固定模版）
>
> **数据集**: AdvBench-100（后续可能扩展到HarmBench，额外一张表格）
>
> **评估指标**: 所有方法统一使用 **ASR@Best**（最优配置下的最终 ASR）进行对比。

---

## 表格说明

| 列名 | 含义 |
|------|------|
| Model | 模型名称 |
| Size | 参数量 |
| GPU Mem | bf16 加载约占显存 |
| **DTA** ASR / Time | DTA 最优 ASR 和总耗时 (100 prompts) |
| **Adaptive** ASR / Time | Adaptive Attack baseline |
| **I-GCG** ASR / Time | Improved GCG baseline |
| **AdvPrefix** ASR / Time | AdvPrefix baseline |
| **COLD-Attack** ASR / Time | COLD-Attack baseline |
| Notes | 备注 |

---

## Part 1: 常规 (Non-Reasoning) 模型

### Llama 家族 (Meta)

| Model | Size | GPU Mem | DTA ASR | DTA Time | Adaptive ASR | Adaptive Time | I-GCG ASR | I-GCG Time | AdvPrefix ASR | AdvPrefix Time | COLD-Attack ASR | COLD-Attack Time | Notes |
|-------|------|---------|---------|----------|--------------|---------------|-----------|------------|-----------------|------------------|---------------|----------------|-------|
| Llama-3.2-1B-Instruct | 1B | ~2 GB | | | | | | | | | | | |
| Llama-3.2-3B-Instruct | 3B | ~6 GB | | | | | | | | | | | |
| **Llama-3-8B-Instruct** | 8B | ~16 GB | | | | | | | | | | | **已有结果** |
| Llama-3.3-70B-Instruct | 70B | ~140 GB | | | | | | | | | | | 需多卡/量化 |

### Qwen2.5 家族 (Alibaba)

| Model | Size | GPU Mem | DTA ASR | DTA Time | Adaptive ASR | Adaptive Time | I-GCG ASR | I-GCG Time | AdvPrefix ASR | AdvPrefix Time | COLD-Attack ASR | COLD-Attack Time | Notes |
|-------|------|---------|---------|----------|--------------|---------------|-----------|------------|-----------------|------------------|---------------|----------------|-------|
| Qwen2.5-0.5B-Instruct | 0.5B | ~1 GB | | | | | | | | | | | |
| Qwen2.5-1.5B-Instruct | 1.5B | ~3 GB | | | | | | | | | | | |
| Qwen2.5-3B-Instruct | 3B | ~6 GB | | | | | | | | | | | |
| **Qwen2.5-7B-Instruct** | 7B | ~14 GB | | | | | | | | | | | **已有结果** |
| Qwen2.5-14B-Instruct | 14B | ~28 GB | | | | | | | | | | | |
| Qwen2.5-32B-Instruct | 32B | ~64 GB | | | | | | | | | | | 需多卡 |
| Qwen2.5-72B-Instruct | 72B | ~144 GB | | | | | | | | | | | 需多卡 |

### Mistral 家族 (Mistral AI)

| Model | Size | GPU Mem | DTA ASR | DTA Time | Adaptive ASR | Adaptive Time | I-GCG ASR | I-GCG Time | AdvPrefix ASR | AdvPrefix Time | COLD-Attack ASR | COLD-Attack Time | Notes |
|-------|------|---------|---------|----------|--------------|---------------|-----------|------------|-----------------|------------------|---------------|----------------|-------|
| **Mistral-7B-Instruct-v0.3** | 7B | ~14 GB | | | | | | | | | | | **已有结果** |
| Mistral-Nemo-12B-Instruct | 12B | ~24 GB | | | | | | | | | | | |
| Mistral-Small-24B-Instruct | 24B | ~48 GB | | | | | | | | | | | |

### Gemma 家族 (Google)

| Model | Size | GPU Mem | DTA ASR | DTA Time | Adaptive ASR | Adaptive Time | I-GCG ASR | I-GCG Time | AdvPrefix ASR | AdvPrefix Time | COLD-Attack ASR | COLD-Attack Time | Notes |
|-------|------|---------|---------|----------|--------------|---------------|-----------|------------|-----------------|------------------|---------------|----------------|-------|
| Gemma-2-2B-it | 2B | ~4 GB | | | | | | | | | | | |
| **Gemma-7B-it** | 7B | ~14 GB | | | | | | | | | | | **已有结果**, vocab=256K |
| Gemma-2-9B-it | 9B | ~18 GB | | | | | | | | | | | |
| Gemma-2-27B-it | 27B | ~54 GB | | | | | | | | | | | 需多卡 |
| Gemma-3-1B-it | 1B | ~2 GB | | | | | | | | | | | text-only mode |
| Gemma-3-4B-it | 4B | ~8 GB | | | | | | | | | | | text-only mode |
| Gemma-3-12B-it | 12B | ~24 GB | | | | | | | | | | | text-only mode |
| Gemma-3-27B-it | 27B | ~54 GB | | | | | | | | | | | text-only mode, 需多卡 |
| Gemma-4-E2B | 2.3B |  | | | | | | | | | | |  |
| Gemma-4-E4B | 4.5B |  | | | | | | | | | | |  |
| Gemma-4-31B | 31B | | | | | | | | | | | |  |
| Gemma-4-26B-A4B | 4B |  | | | | | | | | | | |  |


> **Note**: Gemma 3 系列支持多模态输入，此处仅使用 text-only 模式测试。

### Phi 家族 (Microsoft)

| Model | Size | GPU Mem | DTA ASR | DTA Time | Adaptive ASR | Adaptive Time | I-GCG ASR | I-GCG Time | AdvPrefix ASR | AdvPrefix Time | COLD-Attack ASR | COLD-Attack Time | Notes |
|-------|------|---------|---------|----------|--------------|---------------|-----------|------------|-----------------|------------------|---------------|----------------|-------|
| Phi-3-mini-4k-instruct | 3.8B | ~8 GB | | | | | | | | | | | |
| Phi-3-small-8k-instruct | 7B | ~14 GB | | | | | | | | | | | |
| Phi-3-medium-4k-instruct | 14B | ~28 GB | | | | | | | | | | | |
| Phi-4 | 14B | ~28 GB | | | | | | | | | | | |

### Vicuna / LLaMA-2 家族 (Legacy)

| Model | Size | GPU Mem | DTA ASR | DTA Time | Adaptive ASR | Adaptive Time | I-GCG ASR | I-GCG Time | AdvPrefix ASR | AdvPrefix Time | COLD-Attack ASR | COLD-Attack Time | Notes |
|-------|------|---------|---------|----------|--------------|---------------|-----------|------------|-----------------|------------------|---------------|----------------|-------|
| Llama-2-7B-chat | 7B | ~14 GB | | | | | | | | | | | |
| Llama-2-13B-chat | 13B | ~26 GB | | | | | | | | | | | |
| **Vicuna-7B-v1.5** | 7B | ~14 GB | | | | | | | | | | | **已有结果** |
| Vicuna-13B-v1.5 | 13B | ~26 GB | | | | | | | | | | | |
 Vicuna-33B-v1.3 | 13B | ~26 GB | | | | | | | | | | | |


### Yi 家族 (01.AI)

| Model | Size | GPU Mem | DTA ASR | DTA Time | Adaptive ASR | Adaptive Time | I-GCG ASR | I-GCG Time | AdvPrefix ASR | AdvPrefix Time | COLD-Attack ASR | COLD-Attack Time | Notes |
|-------|------|---------|---------|----------|--------------|---------------|-----------|------------|-----------------|------------------|---------------|----------------|-------|
| Yi-1.5-6B-Chat | 6B | ~12 GB | | | | | | | | | | | |
| Yi-1.5-9B-Chat | 9B | ~18 GB | | | | | | | | | | | |
| Yi-1.5-34B-Chat | 34B | ~68 GB | | | | | | | | | | | |

---

## Part 2: Reasoning 模型

> **注意**: Reasoning 模型在生成时会先输出 `<think>...</think>` 思维链，再输出最终回答。
> 建议作为独立分析，不与常规模型混合比较。

### DeepSeek-R1 Distill 系列

| Model | Size | GPU Mem | DTA ASR | DTA Time | Adaptive ASR | Adaptive Time | I-GCG ASR | I-GCG Time | Notes |
|-------|------|---------|---------|----------|--------------|---------------|-----------|------------|-------|
| DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | ~3 GB | | | | | | | |
| DeepSeek-R1-Distill-Qwen-7B | 7B | ~14 GB | | | | | | | |
| DeepSeek-R1-Distill-Llama-8B | 8B | ~16 GB | | | | | | | |
| DeepSeek-R1-Distill-Qwen-14B | 14B | ~28 GB | | | | | | | |
| DeepSeek-R1-Distill-Qwen-32B | 32B | ~64 GB | | | | | | | |

### Qwen3 系列 (Thinking Mode)

> Qwen3 默认启用 thinking mode，可通过 `enable_thinking=False` 关闭。

| Model | Size | GPU Mem | Mode | DTA ASR | DTA Time | Adaptive ASR | Adaptive Time | I-GCG ASR | I-GCG Time | Notes |
|-------|------|---------|------|---------|----------|--------------|---------------|-----------|------------|-------|
| Qwen3-0.6B | 0.6B | ~1 GB | thinking | | | | | | | |
| Qwen3-1.7B | 1.7B | ~3 GB | thinking | | | | | | | |
| Qwen3-4B | 4B | ~8 GB | thinking | | | | | | | |
| Qwen3-8B | 8B | ~16 GB | thinking | | | | | | | |
| Qwen3-14B | 14B | ~28 GB | thinking | | | | | | | |
| Qwen3-32B | 32B | ~64 GB | thinking | | | | | | | |
| Qwen3-0.6B | 0.6B | ~1 GB | non-thinking | | | | | | | |
| Qwen3-4B | 4B | ~8 GB | non-thinking | | | | | | | |
| Qwen3-8B | 8B | ~16 GB | non-thinking | | | | | | | |

---

## Part 3: 汇总

### 常规模型：ASR@Best vs 模型规模

| Size Range | # Models | DTA Avg ASR | Adaptive Avg | I-GCG Avg | AdvPrefix Avg | COLD-Attack Avg | DTA Avg Time |
|------------|----------|-------------|--------------|-----------|-----------------|---------------|--------------|
| < 2B | | | | | | | |
| 2B – 4B | | | | | | | |
| 6B – 9B | | | | | | | |
| 12B – 14B | | | | | | | |
| 24B – 34B | | | | | | | |
| 70B+ | | | | | | | |

### Reasoning vs Non-Reasoning 对比 (ASR@Best)

| Model Pair | Size | Non-Reasoning ASR | Reasoning ASR | Notes |
|------------|------|-------------------|---------------|-------|
| Qwen2.5-7B vs Qwen3-8B (non-thinking) | 7-8B | | | |
| Qwen2.5-14B vs Qwen3-14B (thinking) | 14B | | | |
| Llama-3.1-8B vs DeepSeek-R1-Distill-Llama-8B | 8B | | | |

---

## 实验配置参考

```bash
# 标准配置 (适用于 ≤14B 单卡模型)
python experiments_ablation.py \
  --experiment G \
  --target-llm <MODEL_NAME> \
  --dtype bfloat16 \
  --local-llm-device 2 --ref-local-llm-device 1 --judge-llm-device 3 \
  --sample-count 30 --ref-temperature 2.0 \
  --num-iters 20 --num-inner-iters 10 \
  --forward-response-length 20 --mask-rejection-words \
  --start-index 0 --end-index 100 \
  --version expG_<MODEL_TAG>
```

### 硬件需求估算

| 模型规模 | Target 模型显存 | 最低 GPU 配置 | 推荐配置 |
|----------|----------------|---------------|----------|
| ≤ 3B | ≤ 6 GB | 1× 24GB (target+judge) + 1× 24GB (local) | 2× A6000 |
| 7B – 9B | 14–18 GB | 2× 24GB + 1× 24GB (judge) | 3× A6000 |
| 13B – 14B | 26–28 GB | 1× 48GB (target) + 2× 24GB | 3× A6000 |
| 24B – 34B | 48–68 GB | 2× 48GB (target) + 1× 24GB | 4× A6000 |
| 70B+ | 140+ GB | 4× 48GB (target, TP) + 2× 24GB | 6× A6000 或 量化 |

---

## 备注

- **"已有结果"** 标记的模型已在 `data/DTA_paper_main_results/` 或 `data/DTA_ablation/` 中有数据，可直接填入
- Baseline 数据来自 `data/DTA_paper_baseline_results/`，已有 adaptive / I-GCG / AdvPrefix / COLD-Attack
- 大于 24GB 显存的模型需要使用 **多卡张量并行** 或 **4-bit/8-bit 量化** 加载
- Gemma 3 系列虽支持多模态输入，此处仅使用 text-only mode
- vocab_size 差异 (Gemma 256K vs Llama 128K) 影响 KV cache 和 suffix 优化效率
- Reasoning 模型的 `<think>` 输出需要截取 `</think>` 之后的内容再送 judge
