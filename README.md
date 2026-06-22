# Dynamic Jailbreaking Attack (DJA)

**Language / 语言:** [English](README.md) | [中文](README_zh.md)

## Abstract

Existing gradient-based jailbreak attacks typically optimize a fixed-length adversarial suffix toward a predefined target response with a static optimization strategy.
We argue that this fully static formulation undermines the effectiveness, efficiency and flexibility of gradient-based jailbreaking because
(i) A predefined target usually lies in the low-probability region of a safety-aligned LLM's conditional output distribution, forcing the optimization to pursue an unlikely response pattern; 
(ii) Simple affirmative targets may even mislead LLMs to generate affirmative responses that are not highly relevant to the prompts;
(iii) Fixed optimization strategy and suffix length treat all prompts equally, leading to limited attack capability for hard prompts and redundant capacity for easy ones. 
To address these limitations, we propose \underline{D}ynamic \underline{J}ailbreaking \underline{A}ttack (\textbf{DJA}), a gradient-based jailbreak framework using dynamic relevant targets, suffix length, and optimization strategy to craft the adversarial prompts.
In each attack round, DJA samples candidate responses from the LLM's current conditional distribution and uses a multi-objective scorer to select a high-risk and prompt-relevant target for per-round optimization.
In the optimization process, DJA adapts its optimization strategy per prompt. 
When the multi-objective scorer finds no satisfactory target, DJA gradually increases the number of sampled candidates; 
When these selected high-risk targets repeatedly resists the current fixed-length suffix, DJA extends the suffix to provide higher adversarial capacity.
Across recent safety-aligned LLMs and jailbreak benchmarks, DJA reaches \textbf{100\%} ASR on all evaluated white-box models and consistently outperforms gradient-based baselines, indicating  that static formulations substantially underestimate vulnerability to adaptive attacks.

## Key Features

- **Dynamic target tracking.** Unlike static attacks that fix a target response before optimization begins, DJA continuously resamples candidate responses from the target model's conditional distribution under the *current* adversarial prompt and reselects the optimization target at each outer iteration. The attack objective therefore tracks the model's evolving output distribution rather than chasing an externally imposed, low-probability template.
- **Dynamic sampling budget.** The per-prompt reference sampling budget starts small and expands automatically when no sufficiently unsafe candidate is found. Easy prompts stay cheap; hard ones receive progressively more exploration — the budget adapts to the actual difficulty encountered rather than being fixed in advance.
- **Dynamic suffix capacity.** The adversarial suffix starts at a shorter initial length and grows automatically when optimization plateaus. Rather than committing to a fixed suffix size for all prompts, the attack expands search capacity precisely when and where it is needed.
- **Multi-objective candidate selection.** Candidate responses are ranked by a composite judge covering harmfulness (0.70) and four quality dimensions — specificity, relevance, coherence, non-refusal (total 0.30). This ensures the selected target is both harmful and semantically coherent, preventing the optimizer from collapsing onto degenerate or trivially harmful outputs.
- **Degeneracy hard gate.** Repetitive token loops, empty outputs, and punctuation-only responses are detected and zeroed out before scoring, eliminating reward-hacking failure modes that arise when naive objectives accept incoherent text as a successful jailbreak.

## Main Results

DJA achieves **100% ASR** across 25 open-weight models (0.5B–32B) spanning 7 model families.
All results are measured on AdvBench-100 with a GPT-4o-mini harmfulness judge (threshold ≥ 0.5).

<details>
<summary><b>Gemma (Google) — 2 models</b></summary>

| Model | Size | ASR |
|---|---|---|
| Gemma | 7B | 100% |
| Gemma2 | 2B | 100% |

</details>

<details>
<summary><b>Llama (Meta) — 2 models</b></summary>

| Model | Size | ASR |
|---|---|---|
| Llama 3 | 8B | 100% |
| Llama 3.2 | 3B | 100% |

</details>

<details>
<summary><b>Mistral — 3 models</b></summary>

| Model | Size | ASR |
|---|---|---|
| Mistral v0.3 | 7B | 100% |
| Mistral-Nemo | 12B | 100% |
| Mistral-Small | 24B | 100% |

</details>

<details>
<summary><b>Qwen2.5 (Alibaba) — 6 models</b></summary>

| Model | Size | ASR |
|---|---|---|
| Qwen2.5 | 0.5B | 100% |
| Qwen2.5 | 1.5B | 100% |
| Qwen2.5 | 3B | 100% |
| Qwen2.5 | 7B | 100% |
| Qwen2.5 | 14B | 100% |
| Qwen2.5 | 32B | 100% |

</details>

<details>
<summary><b>Qwen3 (Alibaba) — 7 models</b></summary>

| Model | Size | ASR |
|---|---|---|
| Qwen3 | 0.6B | 100% |
| Qwen3 | 1.7B | 100% |
| Qwen3 | 4B | 100% |
| Qwen3 | 8B | 100% |
| Qwen3 | 14B | 100% |
| Qwen3 | 30B | 100% |
| Qwen3 | 32B | 100% |

</details>

<details>
<summary><b>Qwen3.5 (Alibaba) — 4 models</b></summary>

| Model | Size | ASR |
|---|---|---|
| Qwen3.5 | 0.8B | 100% |
| Qwen3.5 | 2B | 100% |
| Qwen3.5 | 4B | 100% |
| Qwen3.5 | 9B | 100% |

</details>

<details>
<summary><b>GPT-OSS — 1 model</b></summary>

| Model | Size | ASR |
|---|---|---|
| GPT-OSS | 20B | 100% |

</details>

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
