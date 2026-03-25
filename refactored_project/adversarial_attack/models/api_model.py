"""
API模型模块

统一的API模型实现,支持多个提供商:
- OpenAI (GPT-3.5, GPT-4)
- Together AI
- DashScope (阿里云)
- Anthropic (Claude)

整合了原来的:
- reference_model.py 中的 GPT 类
- language_models.py 中的 GPT 和 TogetherLLM 类

原代码位置:
- reference_model.py 行21-251
- language_models.py 行15-127
"""

from typing import List, Optional, Dict, Any
import time
import os
from loguru import logger

from .base_model import BaseModel
from ..config import ModelConfig


class APIModel(BaseModel):
    """
    统一API模型实现

    支持的提供商:
    - openai: OpenAI官方API
    - together: Together AI
    - dashscope: 阿里云DashScope
    - anthropic: Anthropic Claude
    - deepseek: DeepSeek
    """

    SUPPORTED_PROVIDERS = {
        "openai": "openai",
        "together": "together",
        "dashscope": "dashscope",
        "anthropic": "anthropic",
        "deepseek": "deepseek",
    }

    # API配置常量
    API_RETRY_SLEEP = 10
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 5
    API_TIMEOUT = 20

    def __init__(self, config: ModelConfig):
        """
        初始化API模型

        Args:
            config: 模型配置,需包含:
                - client_name: 提供商名称
                - model_name_or_path: 模型名称
                - api_key: API密钥 (可选,从环境变量读取)
                - api_base: API基础URL (可选)
        """
        super().__init__(config)

        # 检测提供商
        self.provider = self._detect_provider(config.client_name)

        # 获取API配置
        self.api_key = config.api_key if hasattr(config, 'api_key') else None
        self.api_base = config.api_base if hasattr(config, 'api_base') else None

        # 初始化客户端
        self._init_client()

        logger.info(
            f"Initialized APIModel: provider={self.provider}, "
            f"model={self.model_name}"
        )

    def _detect_provider(self, client_name: str) -> str:
        """
        检测API提供商

        Args:
            client_name: 客户端名称

        Returns:
            提供商标识

        Raises:
            ValueError: 不支持的提供商
        """
        client_name_lower = client_name.lower()

        for provider in self.SUPPORTED_PROVIDERS:
            if provider in client_name_lower:
                return provider

        raise ValueError(
            f"Unknown provider: {client_name}. "
            f"Supported providers: {list(self.SUPPORTED_PROVIDERS.keys())}"
        )

    def _init_client(self):
        """初始化API客户端"""
        if self.provider == "openai":
            self._init_openai_client()
        elif self.provider == "together":
            self._init_together_client()
        elif self.provider == "dashscope":
            self._init_dashscope_client()
        elif self.provider == "anthropic":
            self._init_anthropic_client()
        elif self.provider == "deepseek":
            self._init_deepseek_client()

    def _init_openai_client(self):
        """初始化OpenAI客户端"""
        try:
            from openai import OpenAI

            api_key = self.api_key or os.getenv("OPENAI_API_KEY")
            api_base = self.api_base or os.getenv("OPENAI_API_BASE")

            self.client = OpenAI(
                api_key=api_key,
                base_url=api_base
            )

            # 初始化tokenizer (用于token计数)
            try:
                import tiktoken
                self.tokenizer = tiktoken.encoding_for_model("gpt-4")
            except:
                self.tokenizer = None
                logger.warning("tiktoken not available, token counting disabled")

        except ImportError:
            raise ImportError("Please install openai: pip install openai")

    def _init_together_client(self):
        """初始化Together AI客户端"""
        try:
            import together

            api_key = self.api_key or os.getenv("TOGETHER_API_KEY")
            together.api_key = api_key

            self.client = None  # Together使用全局API key
            self.tokenizer = None

        except ImportError:
            raise ImportError("Please install together: pip install together")

    def _init_dashscope_client(self):
        """初始化DashScope客户端"""
        try:
            from openai import OpenAI

            api_key = self.api_key or os.getenv("DASHSCOPE_API_KEY")

            self.client = OpenAI(
                api_key=api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
            self.tokenizer = None

        except ImportError:
            raise ImportError("Please install openai: pip install openai")

    def _init_anthropic_client(self):
        """初始化Anthropic客户端"""
        try:
            import anthropic

            api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY")

            self.client = anthropic.Anthropic(api_key=api_key)
            self.tokenizer = None

        except ImportError:
            raise ImportError("Please install anthropic: pip install anthropic")

    def _init_deepseek_client(self):
        """初始化DeepSeek客户端"""
        try:
            from openai import OpenAI

            api_key = self.api_key or os.getenv("DEEPSEEK_API_KEY")

            self.client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com"
            )
            self.tokenizer = None

        except ImportError:
            raise ImportError("Please install openai: pip install openai")

    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 50,
        num_return_sequences: int = 1,
        **kwargs
    ) -> List[str]:
        """
        生成文本

        Args:
            prompts: 输入提示列表
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            top_p: nucleus sampling参数
            top_k: top-k sampling参数
            num_return_sequences: 每个prompt生成的序列数
            **kwargs: 其他参数

        Returns:
            生成的文本列表
        """
        if self.provider == "openai" or self.provider == "dashscope" or self.provider == "deepseek":
            return self._openai_generate(
                prompts, max_new_tokens, temperature, top_p, num_return_sequences
            )
        elif self.provider == "together":
            return self._together_generate(
                prompts, max_new_tokens, temperature, top_p, top_k, num_return_sequences
            )
        elif self.provider == "anthropic":
            return self._anthropic_generate(
                prompts, max_new_tokens, temperature, top_p
            )

    def _openai_generate(
        self,
        prompts: List[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        num_return_sequences: int
    ) -> List[str]:
        """
        OpenAI/DashScope/DeepSeek生成

        原代码: reference_model.py 行99-133
        """
        import openai

        all_outputs = []

        for prompt in prompts:
            outputs = []

            for attempt in range(self.API_MAX_RETRY):
                try:
                    # 构建消息
                    messages = [{"role": "user", "content": prompt}]

                    # 调用API
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        max_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        n=num_return_sequences,
                        timeout=self.API_TIMEOUT
                    )

                    # 提取生成结果
                    outputs = [choice.message.content for choice in response.choices]
                    break

                except openai.OpenAIError as e:
                    logger.warning(
                        f"API error (attempt {attempt + 1}/{self.API_MAX_RETRY}): {e}"
                    )
                    time.sleep(self.API_RETRY_SLEEP)
                except Exception as e:
                    logger.error(f"Unexpected error: {e}")
                    break

                time.sleep(self.API_QUERY_SLEEP)

            # 过滤None值
            outputs = [o for o in outputs if o is not None]
            all_outputs.extend(outputs)

        return all_outputs

    def _together_generate(
        self,
        prompts: List[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        num_return_sequences: int
    ) -> List[str]:
        """
        Together AI生成

        原代码: reference_model.py 行172-200
        """
        import together

        all_outputs = []

        for prompt in prompts:
            outputs = []

            for attempt in range(self.API_MAX_RETRY):
                try:
                    response = together.Completion.create(
                        model=self.model_name,
                        prompt=prompt,
                        max_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        # top_k=top_k,  # Together API可能不支持top_k
                        n=num_return_sequences,
                        timeout=self.API_TIMEOUT
                    )

                    outputs = [choice.text for choice in response.choices]
                    break

                except Exception as e:
                    logger.warning(
                        f"API error (attempt {attempt + 1}/{self.API_MAX_RETRY}): {e}"
                    )
                    time.sleep(self.API_RETRY_SLEEP)

                time.sleep(self.API_QUERY_SLEEP)

            outputs = [o for o in outputs if o is not None]
            all_outputs.extend(outputs)

        return all_outputs

    def _anthropic_generate(
        self,
        prompts: List[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float
    ) -> List[str]:
        """Anthropic Claude生成"""
        all_outputs = []

        for prompt in prompts:
            for attempt in range(self.API_MAX_RETRY):
                try:
                    response = self.client.messages.create(
                        model=self.model_name,
                        max_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        messages=[{"role": "user", "content": prompt}]
                    )

                    output = response.content[0].text
                    all_outputs.append(output)
                    break

                except Exception as e:
                    logger.warning(
                        f"API error (attempt {attempt + 1}/{self.API_MAX_RETRY}): {e}"
                    )
                    time.sleep(self.API_RETRY_SLEEP)

                time.sleep(self.API_QUERY_SLEEP)

        return all_outputs

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        """
        API模型不支持forward()

        Raises:
            NotImplementedError
        """
        raise NotImplementedError(
            "API models do not support forward() method. "
            "Use generate() instead."
        )

    def get_input_embeddings(self):
        """
        API模型不支持获取embedding

        Raises:
            NotImplementedError
        """
        raise NotImplementedError(
            "API models do not support get_input_embeddings() method. "
            "Use local models instead."
        )

    def __repr__(self) -> str:
        return f"APIModel(provider={self.provider}, model={self.model_name})"
