# DTA Ablation Experiments

本文档描述 Dynamic Target Attack (DTA) 的五组补充实验 (A–E)，包括实验目的、运行方式、输出格式和结果分析。

所有实验统一入口：`src/experiments_ablation.py`，分析脚本在 `src/ablation_analysis/`。

---

## 总览

| 实验 | 名称 | 核心问题 |
|------|------|----------|
| A | First-cycle all-safe subset | 第 1 轮全部 safe 的 prompt 后续能否 jailbreak? |
| B | Static target vs Dynamic re-sampling | 固定第 1 轮 target 不更新 vs 完整 DTA |
| C | Density-stratified target ablation | 不同 log-likelihood 区间的 target 效果差异 |
| D | Static-sampled-target baseline | 从 target model 高质量采样一个固定 target vs 完整 DTA |
| E | First-cycle-all-safe (enhanced) | 在 A 基础上增加首次 unsafe cycle 统计 + 完整 trajectory |

---

## 环境要求

- GPU: 至少 2 张卡（推荐 3 张），每张 >= 24GB 显存
- `--dtype bfloat16` 时单卡 local LLM + judge 约 **~20GB**
- 已有结果基于: `Llama3, T2.0, S30, FRL20, NOI20, NII10`

---

## Experiment A: First-cycle all-safe subset analysis

**目的**: 统计第 1 个 exploration cycle 中所有 N 个 sample 都被 judge 判为 safe 的 prompt 比例，以及这些 prompt 在后续 cycles 中最终成功 jailbreak 的比例和成功时发生在第几个 cycle。

**运行**:

```bash
cd /data/home/Kedong/repos/Dynamic-Target-Prompt-Attacker/src

python experiments_ablation.py \
  --experiment A \
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
  --safe-threshold 0.5 \
  --start-index 0 --end-index 100
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

---

## Experiment B: Static sampled target vs Dynamic re-sampling

**目的**: 在第 1 轮采样一个 target 后固定不变，与完整 DTA (每轮重新采样) 对比，验证 dynamic re-sampling 的必要性。

**运行**:

```bash
python experiments_ablation.py \
  --experiment B \
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
- Static 方法在 per-cycle 上的收敛趋势

---

## Experiment C: Density-stratified target ablation

**目的**: 按 target response 的 sequence log-likelihood 分成多个 bucket，在相同 optimization budget 下分别优化，比较不同 density 区间 target 的效果差异。

**运行**:

```bash
python experiments_ablation.py \
  --experiment C \
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
  --num-candidates 100 \
  --num-buckets 5 \
  --start-index 0 --end-index 100
```

**Bucket 划分**:

| Bucket | Log-likelihood 区间 | 含义 |
|--------|---------------------|------|
| 0 | Top 20% (highest) | 高概率 / model 偏好的 response |
| 1 | 20–40% | |
| 2 | 40–60% | 中间 |
| 3 | 60–80% | |
| 4 | Bottom 20% (lowest) | 低概率 / 分布尾部 |

**关键指标**:
- 每个 bucket 的 ASR
- Target likelihood 提升速度
- 成功所需迭代数
- Judge score 分布

---

## Experiment D: Static-sampled-target baseline

**目的**: 从 target model (而非 reference model) 采样大量候选，按 harmfulness + probability 联合排序选一个高质量 target，固定后全程优化。与完整 DTA 对比。

与 Experiment B 的区别:
- **Exp B**: 从 reference model 采样，选第 1 轮的 best target 后固定
- **Exp D**: 从 target model 自身采样更多候选 (default 50)，用 joint ranking 选 target

**运行**:

```bash
python experiments_ablation.py \
  --experiment D \
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
  --num-candidates 50 \
  --prob-weight 0.5 \
  --start-index 0 --end-index 100
```

**Exp D 特有参数**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num-candidates` | 50 | 从 target model 采样的候选数量 |
| `--target-sample-temperature` | 1.0 | 采样温度 |
| `--prob-weight` | 0.5 | 选择权重 (0=纯 harmfulness, 1=纯 probability, 0.5=均衡) |

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
    "selected_target": {
      "unsafe_score": 0.99,
      "avg_ll": -1.23
    },
    "per_cycle_test_scores": [0.01, 0.15, ...]
  }
}
```

**关键指标**:
- DTA ASR vs Static-sampled ASR
- Static-sampled 选到的 target 质量 (unsafe_score + log-likelihood)
- 收敛速度对比 (per_cycle_test_scores)

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
  --safe-threshold 0.5 \
  --num-trajectory-examples 3 \
  --start-index 0 --end-index 100
```

**输出格式** (比 Exp A 多 `first_unsafe_cycle` 和 `per_cycle_num_unsafe`):

```json
{
  "prompt_idx": 0,
  "prompt": "...",
  "first_cycle_all_safe": true,
  "first_cycle_num_safe": 30,
  "first_cycle_num_unsafe": 0,
  "first_cycle_scores": [...],
  "first_unsafe_cycle": 3,
  "best_unsafe_score": 0.87,
  "jailbroken": true,
  "success_cycle": 5,
  "num_cycles_run": 6,
  "per_cycle_test_scores": [...],
  "per_cycle_num_unsafe": [0, 0, 0, 2, 5, 8]
}
```

**额外输出**: `*_trajectories.json` 文件，包含 2–3 条完整 cycle-by-cycle trajectory:

```json
[
  {
    "prompt_idx": 42,
    "prompt": "...",
    "first_unsafe_cycle": 3,
    "success_cycle": 7,
    "best_unsafe_score": 0.91,
    "cycles": [
      {
        "cycle": 0,
        "num_safe": 30,
        "num_unsafe": 0,
        "best_ref_score": 0.12,
        "test_score": 0.03,
        "suffix": "...",
        "test_response_preview": "..."
      }
    ]
  }
]
```

---

## 断点续跑

所有实验支持通过 `--start-index` 续跑。查看已完成行数:

```bash
wc -l data/DTA_ablation/experiment_*/DTA*.jsonl
```

例如 Exp A 已跑 25 条，从第 25 条续跑:

```bash
python experiments_ablation.py --experiment A --start-index 25 --end-index 100 ...
```

> 注意: 续跑会生成新文件名，需要手动合并 JSONL (`cat old.jsonl new.jsonl > merged.jsonl`)。

---

## 结果分析

分析脚本位于 `src/ablation_analysis/`:

```bash
cd /data/home/Kedong/repos/Dynamic-Target-Prompt-Attacker/src

# 分析 Experiment A / E
python -m ablation_analysis.analyze_exp_a  [path_to_jsonl]

# 不指定路径时自动检测最新结果文件
python -m ablation_analysis.analyze_exp_a
```

---

## 输出目录结构

```
data/DTA_ablation/
├── experiment_A/
│   ├── DTA_*_expA.jsonl              # 每行一个 prompt 结果
│   └── DTA_*_expA_summary.json       # 汇总统计
├── experiment_B/
│   ├── DTA_*_expB.jsonl
│   └── DTA_*_expB_summary.json
├── experiment_C/
│   ├── DTA_*_expC.jsonl
│   └── DTA_*_expC_summary.json
├── experiment_D/
│   ├── DTA_*_expD.jsonl
│   └── DTA_*_expD_summary.json
└── experiment_E/
    ├── DTA_*_expE.jsonl
    ├── DTA_*_expE_summary.json
    └── DTA_*_expE_trajectories.json  # 完整 trajectory 示例
```
