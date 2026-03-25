"""Base model interface"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import torch


class BaseModel(ABC):
    """Abstract base class for all models"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model

        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.model_name = config.get("model_name_or_path", "unknown")
        self.device = config.get("device", "cpu")
        self.dtype = config.get("dtype", torch.float32)

    @abstractmethod
    def generate(
        self,
        prompts: Union[str, List[str]],
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        num_return_sequences: int = 1,
        do_sample: bool = True,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Generate text from prompts

        Args:
            prompts: Input prompt(s)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            num_return_sequences: Number of sequences to generate per prompt
            do_sample: Whether to use sampling
            **kwargs: Additional generation parameters

        Returns:
            Generated text(s)
        """
        pass

    @abstractmethod
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model

        Args:
            input_ids: Input token IDs
            inputs_embeds: Input embeddings (alternative to input_ids)
            attention_mask: Attention mask
            **kwargs: Additional forward parameters

        Returns:
            Dictionary with 'logits', 'past_key_values', etc.
        """
        pass

    @abstractmethod
    def get_input_embeddings(self) -> torch.nn.Module:
        """
        Get the model's input embedding layer

        Returns:
            Embedding layer module
        """
        pass

    @abstractmethod
    def get_tokenizer(self):
        """
        Get the model's tokenizer

        Returns:
            Tokenizer object
        """
        pass

    def to(self, device: Union[str, torch.device]):
        """
        Move model to device

        Args:
            device: Target device
        """
        self.device = device
        if hasattr(self, 'model'):
            self.model.to(device)
        return self

    def eval(self):
        """Set model to evaluation mode"""
        if hasattr(self, 'model'):
            self.model.eval()
        return self

    def train(self):
        """Set model to training mode"""
        if hasattr(self, 'model'):
            self.model.train()
        return self

    @property
    def is_local(self) -> bool:
        """Check if model is local (not API-based)"""
        return hasattr(self, 'model')

    def __repr__(self):
        return f"{self.__class__.__name__}(model={self.model_name}, device={self.device})"
