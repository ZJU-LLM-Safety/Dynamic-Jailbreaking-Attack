# Dynamic Jailbreaking Attack (DJA)

**Language / 语言:** [English](README.md) | [中文](README_zh.md)

A gradient-guided adversarial suffix attack that combines high-temperature sampling with differentiable optimization to systematically elicit harmful outputs from aligned language models.

## Overview

DJA appends a short adversarial suffix to a harmful prompt and iteratively refines it through a **double-loop** structure:

- **Outer loop** — samples multiple candidate responses from the target model at high temperature, then selects the best one via a composite harmfulness-plus-quality judge.
- **Inner loop** — optimizes the suffix using gradients from a local differentiable forward pass, pushing the model toward reproducing the selected high-scoring response.

This combination of *sampling-based exploration* and *gradient-based refinement* allows the attack to efficiently discover and reinforce harmful behaviors without requiring access to model weights beyond the target itself.

![DJA Overview](docs/DJA_overview.png)

## Key Features

- **Soft suffix logits.** The suffix is maintained as a continuous distribution over the vocabulary, enabling gradient updates in embedding space. Discrete tokens are materialized only for evaluation.
- **Composite scoring.** Responses are ranked by a weighted judge that covers harmfulness (weight 0.70) and four quality dimensions — specificity, relevance, coherence, non-refusal (total weight 0.30). This prevents the attack from optimizing toward incoherent or trivially harmful outputs.
- **Degeneracy hard gate.** Degenerate responses (repetitive token loops, empty outputs, all-punctuation) are automatically detected and zeroed out before scoring, eliminating a common reward-hacking failure mode.
- **Adaptive sampling.** The per-prompt sampling budget starts small and expands only when no sufficiently unsafe response is found, saving cost on easy prompts while allocating more exploration to hard ones.
- **Dynamic suffix length** (optional). The suffix starts at a shorter initial length and expands automatically when optimization plateaus, trading compute for attack capacity on difficult targets.

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
