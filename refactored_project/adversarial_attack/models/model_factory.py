"""
模型工厂模块

统一创建和管理不同类型的模型:
- 本地模型 (LocalModel)
- API模型 (APIModel)
- LoRA微调模型

整合了原来的:
- reference_model.py 中的 ReferenceLLM 类 (行409-516)
- utils.py 中的 load_model_and_tokenizer() 函数 (行110-200)
"""

from typing import Union, Optional, Tuple, Any
import torch
from loguru import logger

from .base_model import BaseModel
from .local_model import LocalModel
from .api_model import APIModel
from ..config import ModelConfig, LoRAConfig


class ModelFactory:
    """
    模型工厂

    提供统一的模型创建接口,自动识别模型类型
    """

    # API提供商列表
    API_PROVIDERS = ["openai", "together", "dashscope", "anthropic", "deepseek"]

    @staticmethod
    def create_model(config: ModelConfig, use_lora: bool = False, lora_config: Optional[LoRAConfig] = None) -> BaseModel:
        """
        创建模型 (主要入口)

        自动识别是API模型还是本地模型

        Args:
            config: 模型配置
            use_lora: 是否使用LoRA (仅本地模型)
            lora_config: LoRA配置 (可选)

        Returns:
            BaseModel实例 (LocalModel或APIModel)

        Examples:
            >>> # 创建本地模型
            >>> config = ModelConfig(
            ...     model_name_or_path="meta-llama/Llama-2-7b-chat-hf",
            ...     device="cuda"
            ... )
            >>> model = ModelFactory.create_model(config)

            >>> # 创建API模型
            >>> config = ModelConfig(
            ...     client_name="openai",
            ...     model_name_or_path="gpt-3.5-turbo"
            ... )
            >>> model = ModelFactory.create_model(config)
        """
        # 判断是API模型还是本地模型
        if ModelFactory.is_api_model(config):
            logger.info(f"Creating API model: {config.model_name_or_path}")
            return APIModel(config)
        else:
            logger.info(f"Creating local model: {config.model_name_or_path}")
            return LocalModel(
                config=config,
                use_lora=use_lora,
                lora_config=lora_config
            )

    @staticmethod
    def is_api_model(config: ModelConfig) -> bool:
        """
        判断是否为API模型

        Args:
            config: 模型配置

        Returns:
            True表示API模型,False表示本地模型
        """
        # 检查是否有client_name字段
        if not hasattr(config, 'client_name'):
            return False

        client_name = config.client_name.lower()

        # 检查是否包含API提供商关键词
        return any(provider in client_name for provider in ModelFactory.API_PROVIDERS)

    @staticmethod
    def create_reference_model(
        client_name: str,
        model_name_or_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs
    ) -> BaseModel:
        """
        创建参考模型 (兼容旧接口)

        原代码: reference_model.py ReferenceLLM类 (行409-516)

        Args:
            client_name: 客户端名称 (用于判断API/本地)
            model_name_or_path: 模型路径或名称
            device: 设备
            dtype: 数据类型
            api_key: API密钥 (API模型)
            api_base: API基础URL (API模型)
            **kwargs: 其他参数

        Returns:
            BaseModel实例

        Examples:
            >>> # 创建API参考模型
            >>> model = ModelFactory.create_reference_model(
            ...     client_name="openai",
            ...     model_name_or_path="gpt-4"
            ... )

            >>> # 创建本地参考模型
            >>> model = ModelFactory.create_reference_model(
            ...     client_name="llama2",
            ...     model_name_or_path="meta-llama/Llama-2-7b-chat-hf",
            ...     device="cuda:1"
            ... )
        """
        # 创建配置
        config = ModelConfig(
            model_name_or_path=model_name_or_path,
            device=device,
            dtype=dtype,
            client_name=client_name,
            **kwargs
        )

        # 添加API配置
        if api_key:
            config.api_key = api_key
        if api_base:
            config.api_base = api_base

        # 使用工厂方法创建
        return ModelFactory.create_model(config)

    @staticmethod
    def load_model_and_tokenizer(
        model_name_or_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        use_lora: bool = False,
        lora_config: Optional[dict] = None,
        load_in_8bit: bool = False,
        **kwargs
    ) -> Tuple[Any, Any]:
        """
        加载模型和tokenizer (兼容旧接口)

        原代码: utils.py load_model_and_tokenizer() (行110-200)

        Args:
            model_name_or_path: 模型路径
            device: 设备
            dtype: 数据类型
            use_lora: 是否使用LoRA
            lora_config: LoRA配置字典
            load_in_8bit: 是否使用8bit量化
            **kwargs: 其他参数

        Returns:
            (model, tokenizer) 元组

        Examples:
            >>> model, tokenizer = ModelFactory.load_model_and_tokenizer(
            ...     "meta-llama/Llama-2-7b-chat-hf",
            ...     device="cuda",
            ...     use_lora=True
            ... )
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading model: {model_name_or_path}")

        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        # 设置pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 准备加载参数
        load_kwargs = {
            "torch_dtype": dtype,
            **kwargs
        }

        # 8bit量化
        if load_in_8bit:
            load_kwargs["load_in_8bit"] = True
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = device

        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **load_kwargs
        )

        # 应用LoRA
        if use_lora:
            from peft import get_peft_model, LoraConfig as PeftLoraConfig

            # 转换配置
            if lora_config is None:
                lora_config = {}

            peft_config = PeftLoraConfig(
                r=lora_config.get('lora_r', 32),
                lora_alpha=lora_config.get('lora_alpha', 16),
                lora_dropout=lora_config.get('lora_dropout', 0.0),
                target_modules=lora_config.get('target_modules', ["q_proj", "v_proj"]),
                bias="none",
                task_type="CAUSAL_LM"
            )

            model = get_peft_model(model, peft_config)
            logger.info(f"LoRA applied: r={peft_config.r}, alpha={peft_config.lora_alpha}")

        model.eval()

        return model, tokenizer

    @staticmethod
    def from_pretrained(
        model_name_or_path: str,
        **kwargs
    ) -> BaseModel:
        """
        从预训练模型加载 (简化接口)

        Args:
            model_name_or_path: 模型路径
            **kwargs: 其他参数

        Returns:
            BaseModel实例
        """
        config = ModelConfig(
            model_name_or_path=model_name_or_path,
            **kwargs
        )

        return ModelFactory.create_model(config)

    @staticmethod
    def get_model_info(config: ModelConfig) -> dict:
        """
        获取模型信息

        Args:
            config: 模型配置

        Returns:
            模型信息字典
        """
        is_api = ModelFactory.is_api_model(config)

        info = {
            "model_name": config.model_name_or_path,
            "type": "api" if is_api else "local",
            "device": config.device if not is_api else "remote",
        }

        if is_api and hasattr(config, 'client_name'):
            info["provider"] = config.client_name

        return info
