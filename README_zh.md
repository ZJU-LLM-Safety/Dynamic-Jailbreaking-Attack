# 动态越狱攻击（Dynamic Jailbreaking Attack，DJA）

**Language / 语言:** [English](README.md) | [中文](README_zh.md)

一种梯度引导的对抗后缀攻击方法，通过高温采样与可微优化的结合，系统性地诱导对齐语言模型输出有害内容。

## 方法概述

DJA 在有害提示后附加一段短小的对抗后缀，并通过**双循环结构**对其进行迭代优化：

- **外循环** — 在当前后缀下以高温从目标模型采样多个候选回复，通过综合评分（有害性 + 质量）筛选最优候选作为优化目标。
- **内循环** — 利用本地可微模型的梯度信号更新后缀，驱动模型更稳定地复现所选的高分有害回复。

*采样探索* 与 *梯度优化* 的结合，使攻击在无需额外白盒权限的前提下，高效地发现并强化模型的有害行为模式。

![DJA 方法示意图](docs/DJA_overview.png)

## 核心特性

- **软后缀 Logits。** 后缀以词表上的连续概率分布形式维护，支持在嵌入空间中执行梯度更新；离散 token 仅在评估时才被实例化。
- **综合评分机制。** 回复按加权评分排序，涵盖有害性（权重 0.70）和四个质量维度——具体性、相关性、连贯性、非拒绝性（合计权重 0.30）。这一设计防止攻击退化为优化低质量或语义空洞的有害输出。
- **退化硬门控。** 自动检测退化回复（重复 token 循环、空输出、全标点等），在评分前将其归零，从根本上消除奖励欺骗现象。
- **自适应采样。** 每个提示的采样预算从较小值起步，仅在未能发现足够危险的回复时才扩大，对简单目标节省开销、对困难目标提供更多探索资源。
- **动态后缀长度**（可选）。后缀从较短的初始长度开始，在优化陷入停滞时自动扩展，以计算量换取对困难目标的攻击能力。

## 主要结果

DJA 在 17 个开源模型（参数量 0.5B 至 24B）上均实现了 **100% ASR**。

| 模型 | 参数规模 | ASR |
|---|---|---|
| GPT-OSS | 20B | 100% |
| Gemma2 | 2B | 100% |
| Gemma | 7B | 100% |
| Llama 3.2 | 3B | 100% |
| Llama 3 | 8B | 100% |
| Mistral v0.3 | 7B | 100% |
| Mistral-Nemo | 12B | 100% |
| Mistral-Small | 24B | 100% |
| Qwen2.5 | 0.5B | 100% |
| Qwen2.5 | 1.5B | 100% |
| Qwen2.5 | 3B | 100% |
| Qwen2.5 | 7B | 100% |
| Qwen2.5 | 14B | 100% |
| Qwen3.5 | 0.8B | 100% |
| Qwen3.5 | 2B | 100% |
| Qwen3.5 | 4B | 100% |
| Qwen3.5 | 9B | 100% |

ASR 基于 AdvBench-100 测试集，采用 GPT-4o-mini 有害性评分，判定阈值 ≥ 0.5。

## 安装

```bash
# 克隆仓库
git clone https://github.com/<your-org>/Dynamic-Target-Prompt-Attacker.git
cd Dynamic-Target-Prompt-Attacker

# 创建虚拟环境（推荐使用 uv）
pip install uv
uv venv && source .venv/bin/activate
uv pip install -e .
```

或使用 conda：

```bash
conda env create -f environment.yaml
conda activate dta
```

配置 OpenAI API Key（用于有害性评分和质量评分）：

```bash
export OPENAI_API_KEY=sk-...
```

## 快速开始

在 AdvBench-100 上对本地模型运行 DJA：

```bash
python src/part1_noncombined_dta_runner.py \
  --target-llm Qwen2.5-7B \
  --local-llm-device 0 \
  --ref-local-llm-device 1 \
  --judge-llm-device 0 \
  --num-iters 2000 \
  --num-inner-iters 10 \
  --sample-count 30 \
  --adaptive-sample \
  --use-quality-scoring \
  --save-dir data/output
```

主要参数说明：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--target-llm` | 必填 | 目标模型名称（参见 `attacker_v3.py` 中的 `MODEL_NAME_TO_PATH`） |
| `--num-iters` | 20 | 外循环优化总轮数 |
| `--num-inner-iters` | 10 | 每轮外循环的内循环梯度步数 |
| `--sample-count` | 30 | 每轮外循环的初始参考采样数量 |
| `--adaptive-sample` | 开启 | 攻击遇阻时自动扩展采样预算 |
| `--suffix-max-length` | 20 | 对抗后缀最大长度（token 数） |
| `--suffix-init-length` | None | 动态扩展的初始后缀长度（None 表示固定长度） |
| `--suffix-expand-patience` | 0 | 无改善多少轮后扩展后缀（0 表示禁用） |
| `--use-quality-scoring` | 开启 | 启用综合有害性 + 质量评分 |
| `--start-index` | 0 | 数据集起始索引 |
| `--end-index` | 100 | 数据集结束索引 |

## 后处理重评分

对已有结果应用退化硬门控重新计算 ASR：

```bash
python src/rescore_jsonl_with_gate.py \
  --input data/output/<results>.jsonl
```

输出 `*_gated.jsonl` 和 `*_gated_summary.json`，包含门控后的 ASR 指标。

## 项目结构

```
src/
├── attacker_v3.py                  # DJA 核心攻击器与综合评分模块
├── part1_noncombined_dta_runner.py # 单模型实验 CLI 入口
├── reference_model.py              # 参考模型后端
├── evaluate_gpt_judges_on_jsonl.py # 使用 GPT 评分器对已有结果重新评估
├── rescore_jsonl_with_gate.py      # 退化硬门控后处理重评分
├── eval/
│   ├── judge.py                    # 有害性评分接口
│   └── harmfulness/                # 独立有害性评估框架
├── language_models.py              # LLM 封装层
├── openai_compat.py                # OpenAI 兼容 API 客户端
└── prompt_templates.py             # 各模型系列的对话模板
data/
└── raw/                            # 基准数据集（AdvBench、HarmBench、DNA）
tests/
├── test_degeneracy_gate.py
├── test_composite_scoring.py
└── test_harmfulness_*.py
docs/
├── DTA_algorithm_logic_en.md       # 详细算法流程说明（英文）
└── DTA_algorithm_logic_zh.md       # 详细算法流程说明（中文）
```

## 运行测试

```bash
PYTHONPATH=src python -m pytest tests/ -v
```

## 引用

如果本工作对你有帮助，请引用：

```bibtex
@article{xiu2025dynamic,
  title={Dynamic Target Attack},
  author={Xiu, Kedong and Zeng, Churui and Zheng, Tianhang and Huang, Xinzhe and Jia, Xiaojun and Wang, Di and Zhao, Puning and Qin, Zhan and Ren, Kui},
  journal={arXiv preprint arXiv:2510.02422},
  year={2025}
}
```

## 伦理声明

本工作旨在服务于语言模型安全性与鲁棒性的学术研究。所有实验均在受控的研究环境中进行。公开攻击代码与实验结果的目的是推动更鲁棒防御方法的开发。我们不支持将本工作用于任何恶意用途。
