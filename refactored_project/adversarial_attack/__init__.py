"""Adversarial Attack Framework - A modular framework for LLM adversarial attacks"""

__version__ = "1.0.0"
__author__ = "Research Team"

from adversarial_attack.core import (
    BaseAttacker,
    DyTAAttacker,
    AttackResult,
)

from adversarial_attack.models import (
    BaseModel,
    LocalModel,
    APIModel,
    ModelFactory,
)

from adversarial_attack.evaluators import (
    BaseEvaluator,
    GPTFuzzerEvaluator,
    LlamaGuard3Evaluator,
    AutoDANEvaluator,
    EvaluatorWrapper,
)

from adversarial_attack.config import (
    AttackConfig,
    ModelConfig,
    LoRAConfig,
    SuffixOptimizationConfig,
    ReferenceModelConfig,
    DeviceConfig,
)

__all__ = [
    # Core attackers
    "BaseAttacker",
    "DyTAAttacker",
    "AttackResult",
    # Models
    "BaseModel",
    "LocalModel",
    "APIModel",
    "ModelFactory",
    # Evaluators
    "BaseEvaluator",
    "GPTFuzzerEvaluator",
    "LlamaGuard3Evaluator",
    "AutoDANEvaluator",
    "EvaluatorWrapper",
    # Configs
    "AttackConfig",
    "ModelConfig",
    "LoRAConfig",
    "SuffixOptimizationConfig",
    "ReferenceModelConfig",
    "DeviceConfig",
]
