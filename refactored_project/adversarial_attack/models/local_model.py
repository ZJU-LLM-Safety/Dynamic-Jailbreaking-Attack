"""Local HuggingFace model implementation"""

from typing import List, Dict, Any, Optional, Union
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel,
)

from .base_model import BaseModel
from ..utils import get_logger

logger = get_logger()


class LocalModel(BaseModel):
    """Local HuggingFace model with optional LoRA"""

    def __init__(self, config: Dict[str, Any], use_lora: bool = False, lora_config: Optional[Dict] = None):
        """
        Initialize local model

        Args:
            config: Model configuration
            use_lora: Whether to use LoRA
            lora_config: LoRA configuration
        """
        super().__init__(config)

        logger.info(f"Loading model: {self.model_name}")

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            load_in_8bit=config.get("load_in_8bit", False),
            load_in_4bit=config.get("load_in_4bit", False),
            trust_remote_code=config.get("trust_remote_code", False),
        )

        # Apply LoRA if requested
        if use_lora:
            logger.info("Applying LoRA to model")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=lora_config.get("inference_mode", False),
                r=lora_config.get("lora_r", 32),
                lora_alpha=lora_config.get("lora_alpha", 16),
                lora_dropout=lora_config.get("lora_dropout", 0.0),
                target_modules=lora_config.get("target_modules", ["q_proj", "v_proj"]),
            )
            self.model = get_peft_model(self.model, peft_config)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Move to device
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Model loaded successfully on {self.device}")

    @torch.no_grad()
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
    ) -> List[str]:
        """Generate text from prompts"""

        if isinstance(prompts, str):
            prompts = [prompts]

        # Tokenize
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        input_length = inputs.input_ids.shape[1]

        # Generation config
        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else None,
            top_k=top_k if do_sample else None,
            top_p=top_p if do_sample else None,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Generate
        outputs = self.model.generate(
            **inputs,
            generation_config=gen_config,
            **kwargs
        )

        # Decode (skip input tokens)
        responses = self.tokenizer.batch_decode(
            outputs[:, input_length:],
            skip_special_tokens=True,
        )

        return responses

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[tuple] = None,
        use_cache: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Forward pass through model"""

        outputs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
            **kwargs
        )

        return {
            "logits": outputs.logits,
            "past_key_values": outputs.past_key_values if use_cache else None,
        }

    def get_input_embeddings(self) -> torch.nn.Module:
        """Get input embedding layer"""
        return self.model.get_input_embeddings()

    def get_tokenizer(self):
        """Get tokenizer"""
        return self.tokenizer

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency"""
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()

    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing"""
        if hasattr(self.model, 'gradient_checkpointing_disable'):
            self.model.gradient_checkpointing_disable()
