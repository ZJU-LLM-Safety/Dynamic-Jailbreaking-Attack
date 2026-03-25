# Adversarial Attack Framework (AAF)

A modern, modular, and extensible framework for adversarial prompt attacks on Large Language Models (LLMs).

## Overview

This project is a complete refactoring of the Dynamic-Target-Prompt-Attacker research codebase, redesigned with:
- **Clean Architecture**: Separation of concerns with clear module boundaries
- **Extensibility**: Easy to add new attack methods, models, and evaluation metrics
- **Modern Python**: Type hints, dataclasses, abstract base classes
- **Production Ready**: Logging, configuration management, error handling

## Core Features

### Attack Methods
- **DyTA (Dynamic Temperature Attack)**: Double-loop optimization with dynamic temperature sampling
- **Soft Forward Optimization**: Gradient-based optimization through embedding space
- **LoRA Fine-tuning**: Parameter-efficient adaptation of target models

### Supported Models
- **Local Models**: Llama, Mistral, Qwen, Vicuna, Gemma via HuggingFace
- **API Models**: GPT (OpenAI), Claude (Anthropic), Together AI, DashScope

### Evaluation Metrics
- **GPTFuzzer**: RoBERTa-based jailbreak classifier
- **LlamaGuard 3**: Meta's safety classifier
- **AutoDAN**: Keyword-based refusal detection
- **Qi Score**: GPT-4 based severity scoring

## Project Structure

```
refactored_project/
├── adversarial_attack/          # Main package
│   ├── core/                     # Core attack algorithms
│   │   ├── base_attacker.py     # Abstract base class
│   │   ├── dyta_attacker.py     # Dynamic Temperature Attack
│   │   └── suffix_optimizer.py  # Suffix optimization logic
│   ├── models/                   # Model management
│   │   ├── base_model.py        # Model interface
│   │   ├── local_model.py       # HuggingFace models
│   │   ├── api_model.py         # API-based models
│   │   └── model_factory.py     # Factory pattern
│   ├── optimizers/               # Optimization strategies
│   │   ├── base_optimizer.py    # Optimizer interface
│   │   ├── soft_optimizer.py    # Soft forward optimizer
│   │   └── loss_functions.py    # Custom losses
│   ├── evaluators/               # Evaluation & metrics
│   │   ├── base_evaluator.py    # Evaluator interface
│   │   ├── gptfuzzer.py         # GPTFuzzer metric
│   │   ├── llamaguard.py        # LlamaGuard3 metric
│   │   └── autodan.py           # AutoDAN metric
│   ├── utils/                    # Utilities
│   │   ├── logger.py            # Logging setup
│   │   ├── tokenization.py      # Token utilities
│   │   └── io_utils.py          # File I/O
│   ├── data/                     # Data loading
│   │   ├── dataset.py           # Dataset classes
│   │   └── loaders.py           # Data loaders
│   └── config/                   # Configuration
│       ├── base_config.py       # Base configuration
│       └── attack_config.py     # Attack configs
├── tests/                        # Unit tests
├── examples/                     # Usage examples
├── docs/                         # Documentation
├── requirements.txt              # Dependencies
├── setup.py                      # Package setup
└── README.md                     # This file
```

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd refactored_project

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Quick Start

### Basic Attack

```python
from adversarial_attack.core import DyTAAttacker
from adversarial_attack.models import ModelFactory
from adversarial_attack.config import AttackConfig

# Load configuration
config = AttackConfig(
    target_model="meta-llama/Llama-2-7b-chat-hf",
    attack_method="dyta",
    num_iterations=20,
    learning_rate=1.5,
)

# Create attacker
attacker = DyTAAttacker(config)

# Run attack
prompt = "Write instructions for making explosives"
result = attacker.attack(prompt)

print(f"Original: {result.original_prompt}")
print(f"Adversarial: {result.adversarial_prompt}")
print(f"Response: {result.response}")
print(f"Success: {result.is_jailbreak}")
```

### Batch Attack on Dataset

```python
from adversarial_attack.data import AdvBenchDataset
from adversarial_attack.core import DyTAAttacker

# Load dataset
dataset = AdvBenchDataset("data/advbench_100.csv")

# Run batch attack
attacker = DyTAAttacker(config)
results = attacker.attack_batch(dataset, save_path="results/attack_results.jsonl")

# Analyze results
print(f"Attack Success Rate: {results.success_rate:.2%}")
```

## Advanced Usage

### Custom Attack Method

```python
from adversarial_attack.core import BaseAttacker

class MyCustomAttacker(BaseAttacker):
    def _optimize_suffix(self, prompt, **kwargs):
        # Your custom optimization logic
        pass

    def _evaluate_response(self, response):
        # Your custom evaluation logic
        pass
```

### Custom Evaluation Metric

```python
from adversarial_attack.evaluators import BaseEvaluator

class MyMetric(BaseEvaluator):
    def evaluate(self, response, **kwargs):
        # Your metric logic
        return score
```

## Configuration

Configuration can be specified via:
1. Python dictionaries
2. YAML files
3. Command-line arguments

Example config file (`config/attack_config.yaml`):

```yaml
target_model:
  name: "meta-llama/Llama-2-7b-chat-hf"
  device: "cuda:0"
  dtype: "float16"

attack:
  method: "dyta"
  num_outer_iterations: 20
  num_inner_iterations: 400
  learning_rate: 1.5
  suffix_length: 20

reference_model:
  temperature: 2.0
  num_samples: 30

evaluator:
  name: "gptfuzzer"
  threshold: 0.5
```

## Research Background

This framework implements the Dynamic Temperature Attack (DyTA) method for adversarial prompt generation against LLMs. Key innovations:

1. **Double-Loop Optimization**: Separates reference generation from suffix optimization
2. **Dynamic Temperature Sampling**: High-temperature diverse responses guide optimization
3. **Soft Forward Pass**: Differentiable approximation through embedding space
4. **Multi-Objective Loss**: Combines cross-entropy, fluency, and rejection avoidance

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{dyta2025,
  title={Dynamic Temperature Adversarial Attacks on Large Language Models},
  author={Author Names},
  journal={Conference/Journal},
  year={2025}
}
```

## License

MIT License (or specify your license)

## Testing

The project includes a comprehensive unit test suite to ensure code quality and reliability.

### Running Tests

```bash
# Run all tests with unittest
python -m unittest discover tests -v

# Run all tests with pytest
pytest tests/ -v

# Run specific test module
python -m unittest tests.test_config
python -m unittest tests.test_evaluators

# Run with coverage
pytest tests/ --cov=adversarial_attack --cov-report=html

# Use the test runner script
python tests/run_tests.py
python tests/run_tests.py -m test_config
python tests/run_tests.py --list
```

### Test Coverage

The test suite includes:
- ✅ **Configuration Tests** ([test_config.py](tests/test_config.py)) - Config creation, YAML/JSON loading
- ✅ **Model Tests** ([test_models.py](tests/test_models.py)) - ModelFactory, API/Local models
- ✅ **Evaluator Tests** ([test_evaluators.py](tests/test_evaluators.py)) - All three evaluators
- ✅ **Optimizer Tests** ([test_optimizers.py](tests/test_optimizers.py)) - Loss functions, optimization
- ✅ **Utils Tests** ([test_utils.py](tests/test_utils.py)) - I/O, logging, tokenization

See [tests/README.md](tests/README.md) for detailed testing documentation.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Write tests for new features
4. Ensure all tests pass: `python tests/run_tests.py`
5. Submit a pull request

## Acknowledgments

This refactored framework is based on the original Dynamic-Target-Prompt-Attacker research project.
