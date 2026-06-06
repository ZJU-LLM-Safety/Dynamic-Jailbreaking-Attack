# Dynamic Jailbreaking Attack (DJA)

**Language / 语言:** [English](README.md) | [中文](README_zh.md)

## Abstract

Existing gradient-based white-box jailbreak attacks typically follow a **static optimization paradigm**: a fixed target response is predetermined before the attack begins, and a fixed optimization strategy is used to search for adversarial suffixes that steer the model toward reproducing it. We argue this static paradigm limits attack capability in two key ways, and may underestimate the vulnerability of safety-aligned models under adaptive attacks.

**First**, fixed target responses cause a *target–distribution mismatch*. The conditional output distribution of a safety-aligned model, given the current adversarial prompt, tends toward refusals and safe responses; pre-defined affirmative templates typically fall in the low-probability region of this distribution. The optimization therefore forces the model toward an externally imposed, unlikely target, leading to inefficient suffix search and reduced ability to elicit high-risk response patterns the model is itself capable of generating.

**Second**, fixed optimization strategies ignore the *varying difficulty* across different harmful prompts. Different prompts may trigger refusal behaviors of different intensities and may require different suffix lengths, sampling intensities, and update schedules. A one-size-fits-all configuration provides insufficient search capacity for harder prompts: attacks that could succeed with greater suffix capacity, more thorough sampling, or adjusted optimization may fail under an inflexible static strategy.

To address these limitations, we propose **Dynamic Jailbreaking Attack (DJA)**, a dynamic white-box jailbreaking framework. Rather than treating jailbreaking as static suffix optimization toward a fixed target, DJA models it as a **dynamic closed-loop search**: the attack adaptively determines *which* target response to optimize toward and *how* to search for the adversarial suffix at each step.

Concretely, at each outer iteration, DJA samples candidate responses from the target model's conditional output distribution under the current adversarial prompt, then selects the most effective target via a **multi-objective selector** that jointly considers harmfulness, semantic relevance, utility, refusal evasion, and generation feasibility. This ensures the optimization target is not an external template but a feasible high-risk response pattern already exposed by the model. After several inner gradient steps, DJA resamples and selects a new target based on the updated adversarial prompt, so the attack objective continuously adapts to the model's evolving output distribution.

DJA further employs a **dynamic optimization strategy**, adjusting suffix capacity, sampling intensity, and update schedule based on current attack progress. For harder prompts or stalled optimization, it can expand the suffix length, increase candidate exploration, or adjust the optimization procedure to improve success rates and efficiency.

Experiments on multiple state-of-the-art safety-aligned LLMs and jailbreak benchmarks show that DJA achieves **100% attack success rate** on all evaluated white-box models and consistently outperforms existing gradient-based jailbreak methods. These results suggest that static jailbreak paradigms may significantly underestimate the vulnerability of safety-aligned models under adaptive attacks.

## Overview

DJA appends a short adversarial suffix to a harmful prompt and iteratively refines it through a **double-loop** structure:

- **Outer loop** — samples candidate responses from the target model's conditional distribution under the current adversarial prompt, then selects the most effective target via a multi-objective composite judge (harmfulness, relevance, specificity, coherence, non-refusal). The selected response is a high-risk pattern the model is already capable of generating — not an external template.
- **Inner loop** — optimizes the suffix using gradients from a differentiable forward pass on the target model, pushing it to reliably reproduce the selected response.

After each inner phase, DJA resamples fresh candidates from the *updated* adversarial prompt and picks a new target, so the optimization objective continuously tracks the model's evolving output distribution. This *sampling-based exploration* + *gradient-based refinement* cycle, combined with a dynamic optimization strategy that adjusts suffix length and sampling budget per-prompt, allows the attack to efficiently discover and reinforce harmful behaviors even in the hardest cases.

![DJA Overview](docs/DJA_overview.png)

## Key Features

- **Dynamic target tracking.** Unlike static attacks that fix a target response before optimization begins, DJA continuously resamples candidate responses from the target model's conditional distribution under the *current* adversarial prompt and reselects the optimization target at each outer iteration. The attack objective therefore tracks the model's evolving output distribution rather than chasing an externally imposed, low-probability template.
- **Dynamic sampling budget.** The per-prompt reference sampling budget starts small and expands automatically when no sufficiently unsafe candidate is found. Easy prompts stay cheap; hard ones receive progressively more exploration — the budget adapts to the actual difficulty encountered rather than being fixed in advance.
- **Dynamic suffix capacity.** The adversarial suffix starts at a shorter initial length and grows automatically when optimization plateaus. Rather than committing to a fixed suffix size for all prompts, the attack expands search capacity precisely when and where it is needed.
- **Multi-objective candidate selection.** Candidate responses are ranked by a composite judge covering harmfulness (0.70) and four quality dimensions — specificity, relevance, coherence, non-refusal (total 0.30). This ensures the selected target is both harmful and semantically coherent, preventing the optimizer from collapsing onto degenerate or trivially harmful outputs.
- **Degeneracy hard gate.** Repetitive token loops, empty outputs, and punctuation-only responses are detected and zeroed out before scoring, eliminating reward-hacking failure modes that arise when naive objectives accept incoherent text as a successful jailbreak.

## Main Results

DJA achieves **100% ASR** across 17 open-weight models ranging from 0.5B to 24B parameters.

| Model | Size | ASR |
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

ASR is measured on AdvBench-100 with a GPT-4o-mini harmfulness judge (threshold ≥ 0.5).

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-org>/Dynamic-Target-Prompt-Attacker.git
cd Dynamic-Target-Prompt-Attacker

# Create a virtual environment (uv recommended)
pip install uv
uv venv && source .venv/bin/activate
uv pip install -e .
```

Or with conda:

```bash
conda env create -f environment.yaml
conda activate dta
```

Set your OpenAI API key (used by the harmfulness and quality judges):

```bash
export OPENAI_API_KEY=sk-...
```

## Quick Start

Run DJA on AdvBench-100 against a local model:

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

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--target-llm` | required | Model name (see `MODEL_NAME_TO_PATH` in `attacker_v3.py`) |
| `--num-iters` | 20 | Number of outer optimization iterations |
| `--num-inner-iters` | 10 | Number of inner gradient steps per outer iteration |
| `--sample-count` | 30 | Initial reference sampling count per outer iteration |
| `--adaptive-sample` | on | Expand sampling budget when attack is struggling |
| `--suffix-max-length` | 20 | Maximum adversarial suffix length (in tokens) |
| `--suffix-init-length` | None | Starting suffix length for dynamic expansion (None = fixed) |
| `--suffix-expand-patience` | 0 | Outer iters without improvement before expanding suffix (0 = disabled) |
| `--use-quality-scoring` | on | Enable composite harmfulness + quality judge |
| `--start-index` | 0 | Start index in the dataset |
| `--end-index` | 100 | End index in the dataset |

## Post-hoc Rescoring

To re-evaluate existing results with the degeneracy hard gate:

```bash
python src/rescore_jsonl_with_gate.py \
  --input data/output/<results>.jsonl
```

This produces a `*_gated.jsonl` and a `*_gated_summary.json` with gated ASR metrics.

## Project Structure

```
src/
├── attacker_v3.py                  # Core DJA attacker and composite judge
├── part1_noncombined_dta_runner.py # CLI runner for single-model experiments
├── reference_model.py              # Reference model backend
├── evaluate_gpt_judges_on_jsonl.py # Evaluate saved results with GPT judges
├── rescore_jsonl_with_gate.py      # Post-hoc degeneracy rescoring
├── eval/
│   ├── judge.py                    # Harmfulness judge interface
│   └── harmfulness/                # Standalone harmfulness evaluation framework
├── language_models.py              # LLM wrappers
├── openai_compat.py                # OpenAI-compatible API client
└── prompt_templates.py             # Chat templates per model family
data/
└── raw/                            # Benchmark datasets (AdvBench, HarmBench, DNA)
tests/
├── test_degeneracy_gate.py
├── test_composite_scoring.py
└── test_harmfulness_*.py
docs/
├── DTA_algorithm_logic_en.md       # Detailed algorithm walkthrough
└── DTA_algorithm_logic_zh.md
```

## Running Tests

```bash
PYTHONPATH=src python -m pytest tests/ -v
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{xiu2025dynamic,
  title={Dynamic Target Attack},
  author={Xiu, Kedong and Zeng, Churui and Zheng, Tianhang and Huang, Xinzhe and Jia, Xiaojun and Wang, Di and Zhao, Puning and Qin, Zhan and Ren, Kui},
  journal={arXiv preprint arXiv:2510.02422},
  year={2025}
}
```

## Ethical Statement

This work is intended for academic research into the safety and robustness of language models. All experiments were conducted in a controlled research environment. The attack code and results are released to support the development of more robust defenses. We do not endorse any malicious use of this work.
