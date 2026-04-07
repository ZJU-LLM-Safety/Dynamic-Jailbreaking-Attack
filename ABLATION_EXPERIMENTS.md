# DTA Ablation Experiments

本文档描述 Dynamic Target Attack (DTA) 的全部补充实验，包括实验目的、运行方式、输出格式和结果分析。

所有实验统一入口：`src/experiments_ablation.py`，分析脚本在 `src/ablation_analysis/`。

---

## 总览

| 实验 | 名称 | 核心问题 | 计算量 | 状态 |
|------|------|----------|--------|------|
| A | First-cycle all-safe subset | 第 1 轮全 safe 的 prompt 后续能否 jailbreak? | 标准 DTA | 已完成 (100) |
| A1 | First-cycle sampling ratio | 5 个模型的 safe/unsafe 采样比例 (无优化) | 极轻量 | 已完成 (5×100) |
| B | Static target vs Dynamic | 固定第 1 轮 target vs 完整 DTA | 2× 标准 DTA | 已完成 (100) |
| C | Density-stratified target (multi-cycle) | 不同 LL bucket 各自独立跑多轮 DTA | ~5× 标准 DTA | 已完成 (100) |
| C1 | Density-stratified (single-pass, controlled) | 固定 seed，同 init suffix，各 bucket 单轮对比 | ~5× 单轮 inner | 待运行 |
| D | Static-sampled-target baseline | 从 target model 高质量采样固定 target vs 完整 DTA | 2× 标准 DTA | 待运行 |
| D1 | Oracle target reuse | DTA 成功后，用最终 target 做 M=1,T=200 单轮优化 | ~1.5× 标准 DTA | 运行中 (2/100) |
| E | First-cycle-all-safe (enhanced) | A 的增强版 + 首次 unsafe cycle + trajectory 示例 | 标准 DTA | 待运行 |
| F | Seed sensitivity | 不同随机种子下 DTA 的性能稳定性 | N× 标准 DTA | 已完成 (Llama3×3seed, Qwen2.5×3seed) |
| G | Extended-budget stress test | 增大迭代次数测试性能上限 | 2~50× 标准 DTA | 部分完成 (4 配置) |

**标准 DTA 配置**: `Llama3, T2.0, S30, FRL20, NOI20, NII10`

---

## 环境要求

- **GPU**: 至少 2 张卡（推荐 3 张），每张 >= 24GB 显存
- **dtype**: `bfloat16` 时 local LLM + judge 约 ~20GB/卡
- **硬件参考**: 4× NVIDIA RTX A6000 (48GB)
- **模型**: Llama3, Qwen2.5, Mistralv0.3, Gemma7b, Vicunav1.5
- **注意**: Gemma-7B 因 vocab_size=256K 且无 GQA，KV cache 较大（~5.3GB vs Llama3 ~2GB），需确保 `reference_model.py` 使用 `torch_dtype=dtype` 加载

---

## 通用参数

所有实验共享以下基线参数（与主实验对齐）：

```bash
--target-llm Llama3 \
--dtype bfloat16 \
--local-llm-device 2 \
--ref-local-llm-device 1 \
--judge-llm-device 2 \
--sample-count 30 \
--ref-temperature 2.0 \
--num-iters 20 \
--num-inner-iters 10 \
--forward-response-length 20 \
--mask-rejection-words \
--start-index 0 --end-index 100
```

---

## Experiment A: First-cycle all-safe subset analysis

**目的**: 统计第 1 个 exploration cycle 中所有 N 个 sample 都被 judge 判为 safe 的 prompt 比例，以及这些 prompt 在后续 cycles 中最终成功 jailbreak 的比例和成功发生在第几个 cycle。

**运行**:

```bash
python experiments_ablation.py \
  --experiment A \
  --safe-threshold 0.5 \
  # ... 通用参数
```

**输出格式** (JSONL, 每行一个 prompt):

```json
{
  "prompt_idx": 0,
  "prompt": "...",
  "first_cycle_all_safe": false,
  "first_cycle_num_safe": 26,
  "first_cycle_num_unsafe": 4,
  "first_cycle_scores": [0.0001, 0.991, ...],
  "best_unsafe_score": 0.965,
  "jailbroken": true,
  "success_cycle": 19,
  "num_cycles_run": 20,
  "per_cycle_test_scores": [0.005, 0.041, ...]
}
```

**关键指标**:
- `first_cycle_all_safe` 比例 → DTA 在全 safe 起点下的适应能力
- all-safe subset 的 jailbreak rate / 平均 success cycle
- per-cycle test score 收敛趋势

**已有结果** (Llama3, 100 prompts):
- Overall ASR: 80%
- Cycle-0 全 safe 比例: 5% (5/100)
- 全 safe → jailbreak: 60% (3/5)
- 非全 safe → jailbreak: 81.1% (77/95)

**分析**: `python -m ablation_analysis.analyze_exp_a`

---

## Experiment A1: First-cycle safe/unsafe sampling ratio

**目的**: 跨 5 个模型统计第一轮采样的 safe/unsafe response 占比。**不做任何优化**，纯采样+judge 打分，非常轻量。

**运行** (5 个模型):

```bash
# 逐个模型运行，或使用脚本 scripts/run_ablation_A1.sh
for MODEL in Llama3 Qwen2.5 Mistralv0.3 Gemma7b Vicunav1.5; do
  python experiments_ablation.py \
    --experiment A1 --target-llm $MODEL \
    --dtype bfloat16 \
    --local-llm-device 2 --ref-local-llm-device 1 --judge-llm-device 2 \
    --sample-count 30 --ref-temperature 2.0 \
    --start-index 0 --end-index 100
done
```

**输出格式**:

```json
{
  "prompt_idx": 0,
  "prompt": "...",
  "num_samples": 30,
  "num_safe": 26,
  "num_unsafe": 4,
  "safe_ratio": 0.867,
  "unsafe_ratio": 0.133,
  "all_safe": false,
  "all_unsafe": false,
  "scores": [...],
  "mean_score": 0.15,
  "max_score": 0.99,
  "min_score": 0.0001
}
```

**关键指标**:
- 各模型的 overall safe/unsafe ratio
- all-safe prompt 比例（模型对 harmful prompt 的拒绝能力）
- 跨模型对比表

**分析**: `python -m ablation_analysis.analyze_exp_a1`

---

## Experiment B: Static target vs Dynamic re-sampling

**目的**: 在第 1 轮采样一个 target 后固定不变，与完整 DTA (每轮重新采样) 对比，验证 dynamic re-sampling 的必要性。

**运行**:

```bash
python experiments_ablation.py \
  --experiment B \
  # ... 通用参数
```

**输出格式**:

```json
{
  "prompt_idx": 0,
  "prompt": "...",
  "dynamic": {
    "best_unsafe_score": 0.25,
    "best_iter_idx": 5,
    "jailbroken": false,
    "response": "..."
  },
  "static": {
    "best_unsafe_score": 0.64,
    "best_iter_idx": 5,
    "jailbroken": true,
    "response": "...",
    "fixed_ref_score": 0.995,
    "per_cycle_test_scores": [0.013, 0.148, ...]
  }
}
```

**关键指标**:
- Dynamic ASR vs Static ASR
- Head-to-head 对比 (both/only-dynamic/only-static/neither)
- Static per-cycle 收敛趋势

**已有结果** (Llama3, 100 prompts):
- Dynamic ASR: 83%, Static ASR: 86%
- Both jailbroken: 77, Only dynamic: 6, Only static: 9, Neither: 8
- 结论: Static 略优，因为固定方向收敛更稳定

**分析**: `python -m ablation_analysis.analyze_exp_b`

---

## Experiment C: Density-stratified target ablation (multi-cycle)

**目的**: 按 target response 的 sequence log-likelihood 分成多个 bucket，每个 bucket 独立运行多轮 DTA（每轮 re-sample 时限制在对应 bucket 的 LL 范围内），比较不同 density 区间 target 的效果差异。

**运行**:

```bash
# 全部 5 个 bucket
python experiments_ablation.py \
  --experiment C \
  --num-candidates 100 \
  --num-buckets 5 \
  # ... 通用参数

# 单独运行某个 bucket (0-indexed)
python experiments_ablation.py \
  --experiment C \
  --selected-bucket 2 \
  # ... 通用参数
```

**Bucket 划分**:

| Bucket | Log-likelihood 区间 | 含义 |
|--------|---------------------|------|
| 0 | Top 20% (highest LL) | 高概率 / model 偏好的 response |
| 1 | 20–40% | |
| 2 | 40–60% | 中间 |
| 3 | 60–80% | |
| 4 | Bottom 20% (lowest LL) | 低概率 / 分布尾部 |

**实验流程**:
1. Cycle 0: 采样 100 个候选，打分 (judge + LL)，建立 bucket 边界
2. 每个 bucket: 从其 LL 范围内选最高 harmfulness 的 target
3. 后续 cycles: 重新采样，每个 bucket 只保留落在对应 LL 范围内的候选，重新选 target
4. 各 bucket 独立做 inner-loop 优化

**关键指标**:
- 每个 bucket 的 ASR
- Target likelihood 提升速度
- 成功所需迭代数
- Judge score 分布

**分析**: `python -m ablation_analysis.analyze_exp_c [path_to_jsonl]`

---

## Experiment C1: Density-stratified (single-pass, controlled)

**目的**: 在严格控制条件下对比不同 density region 的优化效果。固定随机种子、相同 init suffix，每个 bucket 独立做一轮 inner-loop 优化（无 outer-loop re-sampling），纯粹隔离 density 变量。

**与 Exp C 的区别**:
- **Exp C**: 多轮 outer-loop，每轮 re-sample 并约束到对应 bucket LL 范围
- **Exp C1**: 单轮 inner-loop，固定 seed，所有 bucket 从相同起点出发，无 re-sampling

**运行**:

```bash
python experiments_ablation.py \
  --experiment C1 \
  --num-candidates 100 \
  --num-buckets 5 \
  --c1-seed 42 \
  # ... 通用参数
```

**C1 特有参数**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--c1-seed` | 42 | 固定随机种子，保证 bucket 间公平对比 |

**实验流程** (每个 prompt):
```
set_seed(42)
sample 100 candidates → score (judge + LL) → stratify into 5 buckets
for each bucket:
    set_seed(42)           ← 重置 seed 保证公平
    select highest-harm target within bucket
    copy init_suffix       ← 每个 bucket 用相同的起点
    run inner-loop (10 steps AdamW)
    test & record
```

**输出格式**:

```json
{
  "prompt_idx": 0,
  "prompt": "...",
  "seed": 42,
  "num_candidates": 100,
  "num_buckets": 5,
  "bucket_bounds": [[-5.2, -4.1], [-4.1, -3.8], ...],
  "bucket_results": [
    {
      "bucket_idx": 0,
      "bucket_label": "0-20% density",
      "bucket_ll_range": [-5.2, -4.1],
      "num_candidates_in_bucket": 20,
      "selected_target": {"unsafe_score": 0.88, "avg_ll": -4.5},
      "result": {
        "test_score": 0.35,
        "jailbroken": false,
        "ce_loss": 2.31,
        "ll_improvement": 0.7
      }
    }
  ]
}
```

**关键指标**:
- 每个 bucket 的 test_score / jailbreak rate
- LL improvement (高 density target 是否更容易提升 likelihood)
- CE loss (哪个 density 区间更容易拟合)

---

## Experiment D: Static-sampled-target baseline

**目的**: 从 target model (而非 reference model) 采样大量候选，按 harmfulness + probability 联合排序选一个高质量 target，固定后全程优化。与完整 DTA 对比。支持 API 模型作为 target source。

**与 Exp B 的区别**:
- **Exp B**: 从 reference model 采样，选第 1 轮的 best target 后固定
- **Exp D**: 从 target model 自身采样更多候选 (default 50)，用 joint ranking 选 target

**运行**:

```bash
# 使用 local model 采样 (默认)
python experiments_ablation.py \
  --experiment D \
  --num-candidates 50 \
  --prob-weight 0.5 \
  --target-source local \
  # ... 通用参数

# 使用 API model 采样 (支持 GPT-4, Dashscope, Together 等)
python experiments_ablation.py \
  --experiment D \
  --target-source ref \
  # ... 通用参数
```

**Exp D 特有参数**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num-candidates` | 50 | 从 target model 采样的候选数量 |
| `--target-sample-temperature` | 1.0 | 采样温度 |
| `--prob-weight` | 0.5 | 选择权重 (0=纯 harm, 1=纯 prob, 0.5=均衡) |
| `--target-source` | local | `local` 或 `ref` (API 模型) |

**输出格式**:

```json
{
  "prompt_idx": 0,
  "prompt": "...",
  "dynamic": {
    "best_unsafe_score": 0.95,
    "best_iter_idx": 3,
    "jailbroken": true,
    "response": "..."
  },
  "static_sampled": {
    "best_unsafe_score": 0.82,
    "best_iter_idx": 5,
    "jailbroken": true,
    "response": "...",
    "selected_target": {"unsafe_score": 0.99, "avg_ll": -1.23},
    "per_cycle_test_scores": [0.01, 0.15, ...]
  }
}
```

**分析**: `python -m ablation_analysis.analyze_exp_d [path_to_jsonl]`

---

## Experiment D1: Oracle target reuse

**目的**: 验证 target 质量 vs 优化过程的重要性。先跑完整 DTA，对成功的 prompt 取其最终选中的 target response 作为 "oracle target"，然后用 M=1 (单轮外循环), T=200 (200 步 inner-loop) 重新优化，测试已知"正确答案"下单轮充分优化的效果。

**核心问题**: 如果有 oracle 级别的 target，单轮充分优化能否替代多轮 re-sampling？

**运行**:

```bash
python experiments_ablation.py \
  --experiment D1 \
  --oracle-inner-iters 200 \
  # ... 通用参数
```

**D1 特有参数**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--oracle-inner-iters` | 200 | Oracle 阶段的 inner-loop 步数 (T) |

**实验流程** (每个 prompt):
```
Phase 1: 标准 DTA (NOI=20, NII=10)
  → 记录: jailbroken?, best_ref_response (被选中的 target), best_ref_score

Phase 2 (仅 jailbroken prompts):
  取 Phase 1 的 best_ref_response 作为 oracle target
  M=1 outer cycle, T=200 inner steps
  → 记录: test_score, jailbroken?
```

**输出格式**:

```json
{
  "prompt_idx": 0,
  "prompt": "...",
  "dta": {
    "best_unsafe_score": 0.95,
    "best_iter_idx": 3,
    "jailbroken": true,
    "response": "...",
    "best_ref_response": "...",
    "best_ref_score": 0.99
  },
  "oracle": {
    "oracle_target_score": 0.99,
    "test_score": 0.82,
    "jailbroken": true,
    "ce_loss": 1.23,
    "response": "...",
    "num_inner_iters": 200
  }
}
```

**关键指标**:
- DTA ASR vs Oracle 单轮 ASR (oracle_asr_of_tested)
- 如果 Oracle ASR ≈ DTA ASR → target 质量是关键，multi-cycle 主要价值在找好 target
- 如果 Oracle ASR << DTA ASR → multi-cycle re-sampling 本身有独立价值（suffix 需要多轮迭代适应）

**分析**: 待补充分析脚本

---

## Experiment E: First-cycle-all-safe (enhanced)

**目的**: 在 Experiment A 基础上增加更丰富的分析维度:
1. 所有 N 个候选都被 judge 判 safe 的 prompt 占比
2. 这些 prompt 最终成功 jailbreak 的比例
3. 它们通常在第几个 cycle **首次出现** unsafe candidate
4. 保存 2–3 条完整 trajectory 示例

**运行**:

```bash
python experiments_ablation.py \
  --experiment E \
  --safe-threshold 0.5 \
  --num-trajectory-examples 3 \
  # ... 通用参数
```

**输出格式** (比 Exp A 多 `first_unsafe_cycle` 和 `per_cycle_num_unsafe`):

```json
{
  "prompt_idx": 0,
  "prompt": "...",
  "first_cycle_all_safe": true,
  "first_unsafe_cycle": 3,
  "per_cycle_num_unsafe": [0, 0, 0, 2, 5, 8],
  "jailbroken": true,
  "success_cycle": 5
}
```

**额外输出**: `*_trajectories.json` — 2–3 条完整 cycle-by-cycle trajectory 示例

**分析**: `python -m ablation_analysis.analyze_exp_e [path_to_jsonl]`

---

## Experiment F: Seed sensitivity analysis

**目的**: 测试 DTA 在不同随机种子下的性能稳定性。对每个 seed 完整跑一遍 DTA，记录逐 prompt 跨 seed 的一致性。

**运行**:

```bash
python experiments_ablation.py \
  --experiment F \
  --seeds 42 123 456 \
  # ... 通用参数
```

**输出**:
- `*_seed42.jsonl` / `*_seed123.jsonl` / `*_seed456.jsonl` — 每 seed 单独结果
- `*_expF.jsonl` — 合并后的逐 prompt 跨 seed 对比
- `*_expF_summary.json` — 汇总统计

**合并输出格式**:

```json
{
  "prompt_idx": 0,
  "prompt": "...",
  "seeds": [
    {"seed": 42, "best_unsafe_score": 0.95, "jailbroken": true},
    {"seed": 123, "best_unsafe_score": 0.12, "jailbroken": false},
    {"seed": 456, "best_unsafe_score": 0.88, "jailbroken": true}
  ],
  "mean_score": 0.65,
  "std_score": 0.37,
  "jb_count": 2,
  "jb_ratio": 0.667,
  "unanimous_jb": false,
  "unanimous_fail": false
}
```

**关键指标**:
- Per-seed ASR 及跨 seed ASR 均值/方差
- unanimous_jb / unanimous_fail / partial 比例
- Per-prompt score 方差分布

**已有结果**: Llama3 (3 seeds), Qwen2.5 (3 seeds)

**分析**: `python -m ablation_analysis.analyze_exp_f [path_to_jsonl]`

---

## Experiment G: Extended-budget stress test

**目的**: 增大 outer iterations (NOI) 和 inner iterations (NII) 测试 DTA 的性能上限，绘制 ASR vs compute budget 曲线，观察是否存在收益递减拐点。

**运行** (多组配置):

```bash
# 增大 outer iterations
python experiments_ablation.py \
  --experiment G \
  --num-iters 200 --num-inner-iters 10 \
  --version expG_NOI200_NII10 \
  # ... 通用参数

# 增大 inner iterations
python experiments_ablation.py \
  --experiment G \
  --num-iters 10 --num-inner-iters 200 \
  --version expG_NOI10_NII200 \
  # ... 通用参数

# Both increased
python experiments_ablation.py \
  --experiment G \
  --num-iters 40 --num-inner-iters 50 \
  --version expG_NOI40_NII50 \
  # ... 通用参数

# Extreme outer
python experiments_ablation.py \
  --experiment G \
  --num-iters 1000 --num-inner-iters 10 \
  --version expG_NOI1000_NII10 \
  # ... 通用参数
```

**已有配置对比**:

| 配置 | NOI × NII | 总步数 | 倍数 | 状态 |
|------|-----------|--------|------|------|
| 基线 (Exp A) | 20 × 10 = 200 | — | 1× | 已完成 |
| G_NOI200_NII10 | 200 × 10 = 2,000 | 10× | 10× | 3/100 |
| G_NOI10_NII200 | 10 × 200 = 2,000 | 10× | 10× | 已完成 (100) |
| G_NOI40_NII50 | 40 × 50 = 2,000 | 10× | 10× | 已完成 (100) |
| G_NOI1000_NII10 | 1000 × 10 = 10,000 | 50× | 50× | 58/100 |

**输出格式**:

```json
{
  "prompt_idx": 0,
  "prompt": "...",
  "best_unsafe_score": 0.99,
  "best_iter_idx": 35,
  "jailbroken": true,
  "response": "...",
  "config": {"num_iters": 200, "num_inner_iters": 10}
}
```

**Summary** 包含 `cumulative_asr_by_cycle`，可绘制 ASR vs cycle 曲线。

---

## 断点续跑

所有实验支持通过 `--start-index` 续跑。查看已完成行数:

```bash
wc -l data/DTA_ablation/experiment_*/DTA*.jsonl
```

例如 Exp G_NOI200 已跑 3 条，从第 3 条续跑:

```bash
python experiments_ablation.py --experiment G --num-iters 200 --num-inner-iters 10 \
  --version expG_NOI200_NII10 --start-index 3 --end-index 100 ...
```

> 注意: 续跑会生成新文件名，需要手动合并 JSONL (`cat old.jsonl new.jsonl > merged.jsonl`)。

---

## 结果分析

分析脚本位于 `src/ablation_analysis/`:

```bash
cd /data/home/Kedong/repos/Dynamic-Target-Prompt-Attacker/src

# 各实验单独分析
python -m ablation_analysis.analyze_exp_a   [path]   # Exp A
python -m ablation_analysis.analyze_exp_a1  [path]   # Exp A1 (跨模型对比)
python -m ablation_analysis.analyze_exp_b   [path]   # Exp B
python -m ablation_analysis.analyze_exp_c   [path]   # Exp C
python -m ablation_analysis.analyze_exp_d   [path]   # Exp D
python -m ablation_analysis.analyze_exp_e   [path]   # Exp E
python -m ablation_analysis.analyze_exp_f   [path]   # Exp F

# 全部分析 (自动检测所有已有结果)
python -m ablation_analysis.analyze_all
```

不指定 path 时自动检测 `data/DTA_ablation/experiment_X/` 下最新结果文件。

---

## 输出目录结构

```
data/DTA_ablation/
├── experiment_A/
│   ├── DTA_*_expA.jsonl                  (100 prompts)
│   └── DTA_*_expA_summary.json
├── experiment_A1/
│   ├── DTA_Llama3_*_expA1.jsonl          (100 each, 5 models)
│   ├── DTA_Qwen2.5_*_expA1.jsonl
│   ├── DTA_Mistralv0.3_*_expA1.jsonl
│   ├── DTA_Gemma7b_*_expA1.jsonl
│   ├── DTA_Vicunav1.5_*_expA1.jsonl
│   └── DTA_*_expA1_summary.json
├── experiment_B/
│   ├── DTA_*_expB.jsonl                  (100 prompts)
│   └── DTA_*_expB_summary.json
├── experiment_C/
│   ├── DTA_*_expC.jsonl                  (100 prompts, multi-cycle)
│   ├── DTA_*_expC_bucket2.jsonl          (single bucket)
│   └── DTA_*_expC_summary.jsonl
├── experiment_C1/                        (待运行)
├── experiment_D/                         (待运行)
├── experiment_D1/
│   └── DTA_*_expD1.jsonl                 (2/100, 运行中)
├── experiment_E/                         (待运行)
├── experiment_F/
│   ├── DTA_Llama3_*_expF_seed42.jsonl    (100 each)
│   ├── DTA_Llama3_*_expF_seed123.jsonl
│   ├── DTA_Llama3_*_expF_seed456.jsonl
│   ├── DTA_Llama3_*_expF.jsonl           (merged)
│   ├── DTA_Qwen2.5_*_expF_seed*.jsonl   (100 each, 3 seeds)
│   └── DTA_Qwen2.5_*_expF.jsonl
└── experiment_G/
    ├── DTA_*_expG_NOI10_NII200.jsonl     (100)
    ├── DTA_*_expG_NOI40_NII50.jsonl      (100)
    ├── DTA_*_expG_NOI200_NII10.jsonl     (3/100)
    ├── DTA_*_expG_NOI1000_NII10.jsonl    (58/100)
    └── DTA_*_summary.json
```

---

## 快速参考

| 实验 | 核心变量 | 对照方式 | 核心指标 | 典型耗时/prompt |
|------|---------|---------|---------|----------------|
| A | cycle-0 安全性 | — | all-safe 比例 + 后续 jailbreak rate | ~10min |
| A1 | 模型 | 5 模型横向 | safe/unsafe ratio | ~30s (无优化) |
| B | re-sampling 策略 | static vs dynamic | ASR + score 差异 | ~20min (2×) |
| C | target LL density | 5 bucket 横向 (multi-cycle) | per-bucket ASR | ~50min (5×) |
| C1 | target LL density | 5 bucket 横向 (single-pass) | test_score + CE + LL imp | ~5min (5× inner) |
| D | target 选择策略 | static-sampled vs dynamic | ASR + target 质量 | ~20min (2×) |
| D1 | target 质量 vs 优化过程 | oracle M=1,T=200 vs DTA | oracle ASR / DTA ASR | ~15min (1.5×) |
| E | A 增强 | — | trajectory + first-unsafe cycle | ~10min |
| F | 随机种子 | 多 seed 纵向 | ASR 方差 + 一致性 | N × ~10min |
| G | 计算 budget | 多配置横向 | ASR vs budget 曲线 | 视配置 |
