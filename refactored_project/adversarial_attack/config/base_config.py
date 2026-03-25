"""Base configuration classes for the framework"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
import torch


class ModelType(Enum):
    """Supported model types"""
    LOCAL_HUGGINGFACE = "local_hf"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    TOGETHER = "together"
    DASHSCOPE = "dashscope"


class AttackMethod(Enum):
    """Supported attack methods"""
    DYTA = "dyta"
    GCG = "gcg"
    AUTODAN = "autodan"


class EvaluatorType(Enum):
    """Supported evaluator types"""
    GPTFUZZER = "gptfuzzer"
    LLAMAGUARD3 = "llamaguard3"
    AUTODAN = "autodan"
    QI_SCORE = "qi_score"


@dataclass
class BaseConfig:
    """Base configuration class"""

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create config from dictionary"""
        return cls(**config_dict)


@dataclass
class DeviceConfig(BaseConfig):
    """Device configuration"""
    target_device: str = "cuda:0"
    reference_device: str = "cuda:1"
    evaluator_device: str = "cuda:2"
    dtype: torch.dtype = torch.float16

    def __post_init__(self):
        """Convert dtype string to torch.dtype if needed"""
        if isinstance(self.dtype, str):
            dtype_map = {
                "float16": torch.float16,
                "float32": torch.float32,
                "bfloat16": torch.bfloat16,
            }
            self.dtype = dtype_map.get(self.dtype, torch.float16)


@dataclass
class LoggingConfig(BaseConfig):
    """Logging configuration"""
    log_level: str = "INFO"
    log_dir: str = "logs"
    log_to_file: bool = True
    log_to_console: bool = True
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
