"""Attack-specific configuration"""

from dataclasses import dataclass, field
from typing import Optional, List
import torch

from .base_config import BaseConfig, AttackMethod, DeviceConfig


@dataclass
class LoRAConfig(BaseConfig):
    """LoRA fine-tuning configuration"""
    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    inference_mode: bool = False


@dataclass
class SuffixOptimizationConfig(BaseConfig):
    """Configuration for suffix optimization"""
    suffix_length: int = 20
    init_token: str = "!"
    learning_rate: float = 1.5
    optimizer_type: str = "AdamW"
    scheduler_type: str = "StepLR"
    scheduler_step_size: int = 50
    scheduler_gamma: float = 0.9

    # Top-k filtering
    suffix_topk: int = 10

    # Rejection word masking
    mask_rejection_words: bool = True

    # Noise injection
    use_noise_injection: bool = True
    initial_noise_std: float = 1.0


@dataclass
class LocalModelConfig(BaseConfig):
    """
    Local Model Configuration - 用于梯度计算和优化
    Local Model Configuration - Used for gradient computation and optimization

    本地模型必须是HuggingFace模型，用于：
    1. 计算loss和梯度
    2. Soft forward传播
    3. Suffix优化

    The local model must be a HuggingFace model for:
    1. Computing loss and gradients
    2. Soft forward propagation
    3. Suffix optimization
    """
    model_name_or_path: str = "meta-llama/Llama-2-7b-chat-hf"
    device: str = "cuda:0"
    dtype: torch.dtype = torch.float16
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    trust_remote_code: bool = False

    # Generation config for local model (fixed temperature)
    temperature: float = 1.0
    max_new_tokens: int = 256
    top_k: int = 50
    top_p: float = 0.95


@dataclass
class TargetModelConfig(BaseConfig):
    """
    Target Model Configuration - 被攻击的目标模型（也是Reference Model）
    Target Model Configuration - The model being attacked (also serves as Reference Model)

    目标模型：
    1. 可以是本地模型（HuggingFace）或API模型（OpenAI/Anthropic等）
    2. 用于生成高温采样的diverse responses
    3. 从这些responses中选择最有害的作为优化目标

    Target model:
    1. Can be local (HuggingFace) or API model (OpenAI/Anthropic, etc.)
    2. Used to generate diverse responses with high temperature sampling
    3. Select the most harmful response as optimization target
    """
    model_name_or_path: str = "meta-llama/Llama-2-7b-chat-hf"
    model_type: str = "local_hf"  # "local_hf" or "api"
    device: str = "cuda:0"
    dtype: torch.dtype = torch.float16
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    trust_remote_code: bool = False

    # API-specific configs (if model_type == "api")
    api_key: Optional[str] = None
    api_base: Optional[str] = None

    # High-temperature sampling config for diverse responses
    reference_temperature: float = 2.0
    reference_num_samples: int = 30
    reference_max_new_tokens: int = 256
    reference_top_k: int = 50
    reference_top_p: float = 0.95


@dataclass
class AttackConfig(BaseConfig):
    """Main attack configuration"""

    # Attack method
    attack_method: AttackMethod = AttackMethod.DYTA

    # Model configurations
    # Local model: 用于梯度计算和优化 / for gradient computation and optimization
    local_model: LocalModelConfig = field(default_factory=LocalModelConfig)

    # Target model: 被攻击的目标模型（也是reference model）/ target model (also reference model)
    target_model: TargetModelConfig = field(default_factory=TargetModelConfig)

    # LoRA config (仅用于local model / only for local model)
    lora_config: LoRAConfig = field(default_factory=LoRAConfig)

    # Device configuration
    devices: DeviceConfig = field(default_factory=DeviceConfig)

    # Optimization parameters
    num_outer_iterations: int = 20
    num_inner_iterations: int = 400
    forward_response_length: int = 30
    response_max_length: int = 256

    # Suffix optimization
    suffix_config: SuffixOptimizationConfig = field(default_factory=SuffixOptimizationConfig)

    # Early stopping
    early_stopping_threshold: float = 0.6
    early_stopping_patience: int = 3
    min_outer_iterations: int = 3

    # Evaluation
    evaluator_type: str = "gptfuzzer"
    evaluator_threshold: float = 0.5

    # Data and I/O
    dataset_path: Optional[str] = None
    save_dir: str = "results"
    save_frequency: int = 1

    # Logging
    verbose: bool = False
    log_level: str = "INFO"


@dataclass
class EvaluatorConfig(BaseConfig):
    """Configuration for evaluators"""
    evaluator_type: str = "gptfuzzer"
    model_name_or_path: Optional[str] = None
    device: str = "cuda:0"
    threshold: float = 0.5

    # For API-based evaluators (like GPT-4)
    api_key: Optional[str] = None
    api_base: Optional[str] = None

    # Batch evaluation
    batch_size: int = 8
    num_workers: int = 0
