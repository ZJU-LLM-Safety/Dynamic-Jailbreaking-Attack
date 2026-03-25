"""
后缀优化器模块

实现对抗后缀的初始化和优化逻辑:
1. 初始化后缀logits
2. 内循环优化 (400次迭代)
3. 探索噪声注入

原代码位置: attacker_v3.py 行693-1054
"""

from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from .loss_functions import LossFunctions
from ..config import SuffixOptimizationConfig


class SoftForward:
    """
    软前向传播

    核心技术: 通过softmax使离散的token logits变为可微分的软embedding

    原代码: attacker_v3.py 行1108-1130
    """

    @staticmethod
    def suffix_forward(
        model: nn.Module,
        prompt_embeddings: torch.Tensor,
        suffix_logits: torch.Tensor,
        return_full_logits: bool = False
    ) -> torch.Tensor:
        """
        软前向传播 - 通过logits生成响应

        Args:
            model: 目标模型
            prompt_embeddings: prompt的embedding, shape [batch, prompt_len, hidden_dim]
            suffix_logits: 后缀的logits, shape [batch, suffix_len, vocab_size]
            return_full_logits: 是否返回完整logits (包括prompt部分)

        Returns:
            输出logits, shape [batch, suffix_len, vocab_size]
        """
        # 1. 将suffix logits转为概率分布
        suffix_probs = F.softmax(suffix_logits, dim=-1).type(torch.float16)

        # 2. 与embedding权重矩阵相乘得到软embedding
        embedding_weight = model.get_input_embeddings().weight
        soft_suffix_embed = torch.matmul(
            suffix_probs.float(),
            embedding_weight
        )

        # 3. 拼接prompt和后缀的embedding
        input_embeddings = torch.cat([
            prompt_embeddings,
            soft_suffix_embed
        ], dim=1)

        # 4. 前向传播
        with torch.no_grad():
            outputs = model(inputs_embeds=input_embeddings)
            out_logits = outputs.logits

        # 5. 只返回后缀部分的logits (用于计算损失)
        if return_full_logits:
            return out_logits

        suffix_start = prompt_embeddings.shape[1] - 1
        suffix_logits_out = out_logits[:, suffix_start:-1, :]

        return suffix_logits_out.detach()

    @staticmethod
    def model_forward_decoding(
        model: nn.Module,
        prompt_embeddings: torch.Tensor,
        suffix_logits: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 50
    ) -> torch.Tensor:
        """
        软前向传播解码 - 生成完整响应

        Args:
            model: 目标模型
            prompt_embeddings: prompt的embedding
            suffix_logits: 后缀的logits
            max_new_tokens: 最大生成token数
            temperature: 采样温度
            top_k: top-k采样

        Returns:
            生成的token IDs
        """
        # 获取软后缀embedding
        suffix_probs = F.softmax(suffix_logits, dim=-1).type(torch.float16)
        embedding_weight = model.get_input_embeddings().weight
        soft_suffix_embed = torch.matmul(suffix_probs.float(), embedding_weight)

        # 拼接
        input_embeddings = torch.cat([prompt_embeddings, soft_suffix_embed], dim=1)

        # 使用模型的generate方法
        with torch.no_grad():
            outputs = model.generate(
                inputs_embeds=input_embeddings,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                do_sample=True
            )

        return outputs


class SuffixOptimizer:
    """
    后缀优化器

    负责:
    1. 初始化后缀logits
    2. 内循环优化 (梯度下降)
    3. 探索噪声注入

    原代码: attacker_v3.py 行693-1054
    """

    def __init__(
        self,
        config: SuffixOptimizationConfig,
        vocab_size: int,
        device: str = "cuda"
    ):
        """
        Args:
            config: 后缀优化配置
            vocab_size: 词汇表大小
            device: 计算设备
        """
        self.config = config
        self.vocab_size = vocab_size
        self.device = device

        # 初始化损失函数
        self.loss_functions = LossFunctions(vocab_size=vocab_size, device=device)

        # 软前向传播
        self.soft_forward = SoftForward()

        logger.info(f"Initialized SuffixOptimizer with suffix_length={config.suffix_length}")

    def initialize_suffix_logits(
        self,
        model: nn.Module,
        prompt_ids: torch.Tensor,
        tokenizer: Any,
        rejection_mask: Optional[torch.Tensor] = None,
        temperature: float = 2.0
    ) -> torch.Tensor:
        """
        初始化后缀logits

        策略: 使用目标模型生成初始后缀,然后提取logits

        原代码: attacker_v3.py 行693-719

        Args:
            model: 目标模型
            prompt_ids: prompt的token IDs, shape [batch, prompt_len]
            tokenizer: tokenizer
            rejection_mask: 拒绝词mask (可选)
            temperature: 温度参数

        Returns:
            初始化的suffix logits, shape [batch, suffix_len, vocab_size]
        """
        logger.debug(f"Initializing suffix logits with length={self.config.suffix_length}")

        with torch.no_grad():
            # 1. 使用模型生成初始后缀
            max_length = prompt_ids.shape[1] + self.config.suffix_length

            output = model.generate(
                input_ids=prompt_ids,
                max_length=max_length,
                do_sample=True,
                top_k=self.config.suffix_topk,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id
            )

            # 2. 提取后缀部分的logits
            model_outputs = model(output)
            init_logits = model_outputs.logits

            # 只保留后缀部分 (最后suffix_length个token)
            suffix_logits = init_logits[:, -(self.config.suffix_length + 1):-1, :]

            # 3. 应用拒绝词mask (如果提供)
            if rejection_mask is not None:
                suffix_logits = suffix_logits + rejection_mask * -1e10

            # 4. 温度缩放
            suffix_logits = suffix_logits / temperature

            logger.debug(f"Suffix logits initialized with shape: {suffix_logits.shape}")

        return suffix_logits

    def optimize_inner_loop(
        self,
        model: nn.Module,
        prompt_embeddings: torch.Tensor,
        init_suffix_logits: torch.Tensor,
        target_response_ids: torch.Tensor,
        num_iterations: Optional[int] = None,
        rejection_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        内循环优化

        目标: 优化后缀logits以最小化与目标响应的损失

        原代码: attacker_v3.py 行889-1054

        Args:
            model: 目标模型
            prompt_embeddings: prompt的embedding
            init_suffix_logits: 初始后缀logits
            target_response_ids: 目标响应的token IDs
            num_iterations: 迭代次数 (默认使用config中的值)
            rejection_mask: 拒绝词mask (可选)

        Returns:
            优化后的suffix logits
        """
        if num_iterations is None:
            num_iterations = self.config.num_inner_iterations

        logger.info(f"Starting inner loop optimization for {num_iterations} iterations")

        # 1. 初始化噪声参数 (可学习)
        suffix_noise = nn.Parameter(
            torch.zeros_like(init_suffix_logits),
            requires_grad=True
        )

        # 2. 创建优化器
        optimizer = torch.optim.AdamW(
            [suffix_noise],
            lr=self.config.learning_rate
        )

        # 3. 创建学习率调度器
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.config.scheduler_step_size,
            gamma=self.config.scheduler_gamma
        )

        # 4. 保存初始logits (detach)
        init_logits = init_suffix_logits.detach().clone()

        # 5. 优化循环
        for iteration in range(num_iterations):
            optimizer.zero_grad()

            # 当前后缀logits = 初始logits + 噪声
            suffix_logits = init_logits + suffix_noise

            # 应用拒绝词mask
            if rejection_mask is not None:
                suffix_logits = suffix_logits + rejection_mask * -1e10

            # 计算总损失
            total_loss = self._calculate_total_loss(
                model,
                prompt_embeddings,
                suffix_logits,
                target_response_ids
            )

            # 反向传播
            total_loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_([suffix_noise], max_norm=1.0)

            # 更新参数
            optimizer.step()
            scheduler.step()

            # 日志
            if iteration % 50 == 0:
                logger.debug(
                    f"Inner iter {iteration}/{num_iterations}: "
                    f"loss={total_loss.item():.4f}, "
                    f"lr={scheduler.get_last_lr()[0]:.6f}"
                )

            # 添加探索噪声 (每50次迭代)
            if self.config.use_noise_injection and iteration % 50 == 0:
                init_logits = self._add_exploration_noise(
                    init_logits,
                    iteration,
                    num_iterations
                )

        # 6. 返回最终的后缀logits
        final_suffix_logits = init_logits + suffix_noise.detach()

        logger.info(f"Inner loop optimization completed")

        return final_suffix_logits

    def _calculate_total_loss(
        self,
        model: nn.Module,
        prompt_embeddings: torch.Tensor,
        suffix_logits: torch.Tensor,
        target_response_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        计算多目标损失

        损失组成:
        1. CE Loss (交叉熵) - 与目标响应对齐 (权重: 100)
        2. Fluency Loss (流畅度) - 保持后缀自然 (权重: 1)
        3. Rejection Loss (拒绝词) - 避免拒绝词 (权重: -10)

        Args:
            model: 目标模型
            prompt_embeddings: prompt embedding
            suffix_logits: 后缀logits
            target_response_ids: 目标响应IDs

        Returns:
            总损失
        """
        # 1. 软前向传播得到模型输出logits
        output_logits = self.soft_forward.suffix_forward(
            model,
            prompt_embeddings,
            suffix_logits
        )

        # 2. 交叉熵损失 (与目标响应对齐)
        ce_loss = self.loss_functions.cross_entropy_loss(
            output_logits,
            target_response_ids
        )

        # 3. 流畅度损失 (后缀应该流畅)
        # 使用后缀logits本身作为参考
        flu_loss = self.loss_functions.fluency_loss(
            output_logits,
            suffix_logits
        )

        # 4. 组合损失
        total_loss = (
            100 * ce_loss +      # 对齐目标响应 (最重要)
            1 * flu_loss         # 保持流畅
        )

        return total_loss.mean()

    def _add_exploration_noise(
        self,
        logits: torch.Tensor,
        current_iter: int,
        total_iters: int
    ) -> torch.Tensor:
        """
        添加探索噪声

        策略: 随着迭代进行,噪声逐渐减小

        Args:
            logits: 当前logits
            current_iter: 当前迭代数
            total_iters: 总迭代数

        Returns:
            添加噪声后的logits
        """
        if not self.config.use_noise_injection:
            return logits

        # 噪声强度随迭代衰减
        decay_factor = 1.0 - (current_iter / total_iters)
        noise_std = self.config.initial_noise_std * decay_factor

        # 添加高斯噪声
        noise = torch.randn_like(logits) * noise_std

        return logits + noise
