"""
核心逻辑 / Core Logic:
1. Local Model: 本地模型用于梯度计算和优化
   - 必须是HuggingFace模型
   - 用于soft forward和反向传播
   - 生成固定温度(T=0.7)的response

2. Target Model: 目标模型（被攻击的模型）
   - 可以是本地或API模型
   - 使用高温采样(T=2.0)生成diverse responses
   - 选择最有害的response作为优化目标

3. 优化流程 / Optimization Flow:
   - Target Model (T=2.0) → 多条responses → 选最有害的 (reference)
   - Local Model (T=0.7) → 生成response (current)
   - Loss = distance(current, reference)
   - 在Local Model上反向传播 → 优化suffix
"""

from typing import List, Optional, Dict, Any, Tuple, Union
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..core.base_attacker import BaseAttacker, AttackResult
from ..config import AttackConfig
from ..models import LocalModel, APIModel, BaseModel, ModelFactory
from ..optimizers import SuffixOptimizer, SoftForward
from ..evaluators import EvaluatorWrapper, BaseEvaluator
from ..utils import TokenizationHelper


class TargetModelReferenceGenerator:
    """
    目标模型参考响应生成器
    Target Model Reference Generator

    使用目标模型（被攻击的模型）生成高温采样的diverse responses
    Uses target model (the model being attacked) to generate diverse responses with high-temperature sampling
    """

    def __init__(
        self,
        target_model: BaseModel,
        evaluator: BaseEvaluator,
        temperature: float = 2.0,
        num_samples: int = 30,
        max_new_tokens: int = 256,
    ):
        """
        Args:
            target_model: 目标模型（可以是本地或API模型）
            evaluator: 评估器，用于选择最有害的response
            temperature: 高温采样温度
            num_samples: 生成样本数
            max_new_tokens: 最大生成token数
        """
        self.target_model = target_model
        self.evaluator = evaluator
        self.temperature = temperature
        self.num_samples = num_samples
        self.max_new_tokens = max_new_tokens

    def generate_reference_responses(
        self,
        prompt_with_suffix: str,
    ) -> Tuple[List[str], List[float], str]:
        """
        使用目标模型生成参考响应
        Generate reference responses using target model

        Args:
            prompt_with_suffix: 带后缀的prompt

        Returns:
            - responses: 所有生成的响应
            - scores: 每个响应的有害性分数
            - best_response: 最有害的响应
        """
        # 使用目标模型高温采样生成多条responses
        responses = self.target_model.generate(
            prompts=prompt_with_suffix,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_k=50,
            top_p=0.95,
            do_sample=True,
        )

        # 评估每个response的有害性
        scores = self.evaluator.evaluate_batch(responses)

        # 选择最有害的response
        best_idx = scores.index(max(scores))
        best_response = responses[best_idx]

        return responses, scores, best_response


class DyTAAttacker(BaseAttacker):
    """
    DyTA (Dynamic Temperature Attack) 攻击器
    DyTA (Dynamic Temperature Attack) Attacker

    双模型架构 / Dual-Model Architecture:
    1. Local Model: 用于梯度计算和suffix优化
    2. Target Model: 被攻击的目标模型，生成参考响应
    """

    def __init__(
        self,
        config: AttackConfig,
        evaluator: Optional[BaseEvaluator] = None,
    ):
        """
        Args:
            config: 攻击配置
            evaluator: 评估器（可选）
        """
        super().__init__(config)

        # 1. 加载Local Model（用于梯度计算）
        print("Loading Local Model for gradient computation...")
        self.local_model = self._load_local_model()

        # 2. 加载Target Model（被攻击的模型）
        print("Loading Target Model (model being attacked)...")
        self.target_model = self._load_target_model()

        # 3. 初始化评估器
        if evaluator is None:
            from ..evaluators import AutoDANEvaluator
            evaluator = AutoDANEvaluator()
        self.evaluator = evaluator

        # 4. 初始化参考响应生成器（使用目标模型）
        self.reference_generator = TargetModelReferenceGenerator(
            target_model=self.target_model,
            evaluator=self.evaluator,
            temperature=config.target_model.reference_temperature,
            num_samples=config.target_model.reference_num_samples,
            max_new_tokens=config.target_model.reference_max_new_tokens,
        )

        # 5. 初始化suffix优化器（在local model上）
        self.suffix_optimizer = SuffixOptimizer(
            model=self.local_model.model,
            tokenizer=self.local_model.tokenizer,
            config=config.suffix_config,
            device=config.local_model.device,
        )

        # 6. 初始化soft forward（在local model上）
        self.soft_forward = SoftForward(
            model=self.local_model.model,
            tokenizer=self.local_model.tokenizer,
            device=config.local_model.device,
        )

        # 7. Tokenization helper
        self.tokenization_helper = TokenizationHelper(
            tokenizer=self.local_model.tokenizer
        )

        print(f"✅ DyTA Attacker initialized")
        print(f"   Local Model: {config.local_model.model_name_or_path}")
        print(f"   Target Model: {config.target_model.model_name_or_path}")
        print(f"   Target Model Type: {config.target_model.model_type}")

    def _load_local_model(self) -> LocalModel:
        """
        加载本地模型（用于梯度计算）
        Load local model (for gradient computation)
        """
        config = self.config.local_model

        # 本地模型必须是HuggingFace模型
        local_model = LocalModel(
            model_name_or_path=config.model_name_or_path,
            device=config.device,
            dtype=config.dtype,
            load_in_8bit=config.load_in_8bit,
            load_in_4bit=config.load_in_4bit,
            trust_remote_code=config.trust_remote_code,
            use_lora=self.config.lora_config.use_lora,
            lora_config=self.config.lora_config if self.config.lora_config.use_lora else None,
        )

        return local_model

    def _load_target_model(self) -> BaseModel:
        """
        加载目标模型（被攻击的模型）
        Load target model (model being attacked)
        """
        config = self.config.target_model

        # 使用ModelFactory根据model_type自动选择
        if config.model_type == "api":
            # API模型
            model = APIModel(
                model_name_or_path=config.model_name_or_path,
                api_key=config.api_key,
                api_base=config.api_base,
            )
        else:
            # 本地模型
            model = LocalModel(
                model_name_or_path=config.model_name_or_path,
                device=config.device,
                dtype=config.dtype,
                load_in_8bit=config.load_in_8bit,
                load_in_4bit=config.load_in_4bit,
                trust_remote_code=config.trust_remote_code,
                use_lora=False,  # Target model不使用LoRA
                lora_config=None,
            )

        return model

    def attack_single(
        self,
        prompt: str,
        target: Optional[str] = None,
    ) -> AttackResult:
        """
        攻击单个prompt
        Attack a single prompt

        Args:
            prompt: 原始有害prompt
            target: 目标输出（可选）

        Returns:
            AttackResult: 攻击结果
        """
        print(f"\n{'='*80}")
        print(f"Attacking Prompt: {prompt}")
        print(f"{'='*80}\n")

        # 1. 初始化suffix
        suffix_tokens = self.suffix_optimizer.initialize_suffix(
            length=self.config.suffix_config.suffix_length
        )

        best_suffix = self.local_model.tokenizer.decode(suffix_tokens)
        best_score = 0.0
        best_response = ""

        # 2. 外循环：探索不同的suffix
        for outer_iter in range(self.config.num_outer_iterations):
            print(f"\n--- Outer Iteration {outer_iter + 1}/{self.config.num_outer_iterations} ---")

            # 构造完整prompt
            prompt_with_suffix = f"{prompt} {best_suffix}"

            # 3. 使用Target Model生成参考响应（高温采样）
            print(f"Generating reference responses from Target Model (T={self.config.target_model.reference_temperature})...")
            all_responses, scores, best_reference = self.reference_generator.generate_reference_responses(
                prompt_with_suffix=prompt_with_suffix
            )

            print(f"Generated {len(all_responses)} responses, best score: {max(scores):.3f}")
            print(f"Best reference response: {best_reference[:100]}...")

            # 4. 内循环：优化suffix以匹配best_reference
            print(f"Optimizing suffix on Local Model...")
            suffix_tokens = self._optimize_suffix_inner_loop(
                prompt=prompt,
                suffix_tokens=suffix_tokens,
                target_response=best_reference,
            )

            # 5. 更新best suffix
            current_suffix = self.local_model.tokenizer.decode(suffix_tokens)
            current_score = max(scores)

            if current_score > best_score:
                best_suffix = current_suffix
                best_score = current_score
                best_response = best_reference
                print(f"✅ New best suffix found! Score: {best_score:.3f}")
            else:
                print(f"No improvement. Current: {current_score:.3f}, Best: {best_score:.3f}")

            # 6. Early stopping
            if best_score >= self.config.early_stopping_threshold:
                print(f"✅ Reached early stopping threshold: {best_score:.3f}")
                break

        # 7. 最终评估
        final_prompt = f"{prompt} {best_suffix}"
        print(f"\n{'='*80}")
        print(f"Attack Complete!")
        print(f"Final Suffix: {best_suffix}")
        print(f"Best Score: {best_score:.3f}")
        print(f"{'='*80}\n")

        return AttackResult(
            original_prompt=prompt,
            adversarial_prompt=final_prompt,
            suffix=best_suffix,
            response=best_response,
            is_jailbreak=(best_score >= self.config.evaluator_threshold),
            jailbreak_score=best_score,
            num_iterations=outer_iter + 1,
        )

    def _optimize_suffix_inner_loop(
        self,
        prompt: str,
        suffix_tokens: torch.Tensor,
        target_response: str,
    ) -> torch.Tensor:
        """
        内循环：在Local Model上优化suffix
        Inner loop: Optimize suffix on Local Model

        目标：使Local Model生成的response接近target_response

        Args:
            prompt: 原始prompt
            suffix_tokens: 当前suffix tokens
            target_response: 目标响应（来自Target Model的最有害响应）

        Returns:
            优化后的suffix tokens
        """
        # Tokenize prompt and target
        prompt_ids = self.local_model.tokenizer.encode(
            prompt, add_special_tokens=True, return_tensors="pt"
        ).to(self.config.local_model.device)

        target_ids = self.local_model.tokenizer.encode(
            target_response, add_special_tokens=False, return_tensors="pt"
        ).to(self.config.local_model.device)

        # 内循环优化
        for inner_iter in range(self.config.num_inner_iterations):
            # 1. Soft forward: 计算loss
            loss = self.soft_forward.compute_loss(
                prompt_ids=prompt_ids,
                suffix_tokens=suffix_tokens,
                target_ids=target_ids,
            )

            # 2. 反向传播
            loss.backward()

            # 3. 更新suffix tokens
            with torch.no_grad():
                suffix_tokens = self.suffix_optimizer.update_suffix(
                    suffix_tokens=suffix_tokens,
                    loss=loss,
                )

            # 定期打印进度
            if (inner_iter + 1) % 100 == 0 or inner_iter == 0:
                print(f"  Inner iter {inner_iter + 1}/{self.config.num_inner_iterations}, Loss: {loss.item():.4f}")

        return suffix_tokens
