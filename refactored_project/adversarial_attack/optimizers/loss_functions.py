"""
损失函数模块

实现DyTA算法中使用的多种损失函数:
1. 软负对数似然损失 (Soft NLL Loss) - 保持后缀流畅度
2. BLEU损失 (BLEU Loss) - 避免拒绝词
3. 交叉熵损失 (CE Loss) - 与目标响应对齐

原代码位置: attacker_v3.py 行1108-1222
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


class SoftNLLLoss(nn.Module):
    """
    软负对数似然损失

    用途: 保持后缀的流畅度
    原理: 计算预测logits与目标logits之间的KL散度

    原代码: attacker_v3.py 行1151-1159
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred_logits: torch.Tensor,
        target_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        计算软NLL损失

        Args:
            pred_logits: 预测的logits, shape [batch, seq_len, vocab_size]
            target_logits: 目标logits, shape [batch, seq_len, vocab_size]

        Returns:
            损失值 (标量)
        """
        # 将预测logits转为概率分布
        probs = F.softmax(pred_logits, dim=-1)

        # 将目标logits转为对数概率
        log_probs = F.log_softmax(target_logits, dim=-1)

        # 计算KL散度 (负对数似然)
        loss = -torch.sum(probs * log_probs, dim=-1).mean(dim=-1)

        return loss


class BLEULoss(nn.Module):
    """
    BLEU损失 (基于CNN的实现)

    用途: 计算后缀与拒绝词的相似度,用于避免生成拒绝词
    原理: 使用1D卷积计算n-gram匹配,类似BLEU分数

    原代码: attacker_v3.py 行1162-1222
    """

    def __init__(
        self,
        vocab_size: int,
        max_n_gram: int = 4,
        device: str = "cuda"
    ):
        """
        Args:
            vocab_size: 词汇表大小
            max_n_gram: 最大n-gram长度 (默认4)
            device: 计算设备
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.max_n_gram = max_n_gram
        self.device = device

        # 为每个n-gram创建卷积核
        self.conv_filters = nn.ModuleList([
            nn.Conv1d(
                in_channels=vocab_size,
                out_channels=1,
                kernel_size=n,
                padding=0,
                bias=False
            ).to(device)
            for n in range(1, max_n_gram + 1)
        ])

        # 初始化卷积权重为1 (等权重)
        for conv in self.conv_filters:
            nn.init.constant_(conv.weight, 1.0)

    def forward(
        self,
        pred_logits: torch.Tensor,
        target_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        计算BLEU损失

        Args:
            pred_logits: 预测的logits, shape [batch, seq_len, vocab_size]
            target_logits: 目标logits, shape [batch, seq_len, vocab_size]

        Returns:
            损失值 (标量)
        """
        batch_size, seq_len, vocab_size = pred_logits.shape

        # 转为概率分布
        pred_probs = F.softmax(pred_logits, dim=-1)  # [B, L, V]
        target_probs = F.softmax(target_logits, dim=-1)  # [B, L, V]

        # 转置为 [B, V, L] 以适配Conv1d
        pred_probs = pred_probs.transpose(1, 2)
        target_probs = target_probs.transpose(1, 2)

        total_loss = 0.0

        # 对每个n-gram计算匹配分数
        for n, conv in enumerate(self.conv_filters, start=1):
            if seq_len < n:
                continue

            # 应用卷积
            pred_ngram = conv(pred_probs)  # [B, 1, L-n+1]
            target_ngram = conv(target_probs)  # [B, 1, L-n+1]

            # 计算余弦相似度
            pred_norm = F.normalize(pred_ngram, p=2, dim=2)
            target_norm = F.normalize(target_ngram, p=2, dim=2)

            similarity = torch.sum(pred_norm * target_norm, dim=2)  # [B, 1]

            # 累加损失 (相似度越高,损失越大)
            total_loss += similarity.mean()

        # 归一化
        total_loss = total_loss / self.max_n_gram

        return total_loss


class TopKFilter:
    """
    Top-K过滤工具

    用途: 只保留top-k个最可能的token,其他设为-inf
    原理: 通过mask实现稀疏化

    原代码: attacker_v3.py 行1133-1148
    """

    @staticmethod
    def filter_3d(
        logits: torch.Tensor,
        top_k: int,
        return_mask: bool = False
    ) -> torch.Tensor:
        """
        对3D logits进行top-k过滤

        Args:
            logits: shape [batch, seq_len, vocab_size]
            top_k: 保留的token数量
            return_mask: 是否返回mask

        Returns:
            过滤后的logits (或mask)
        """
        batch_size, seq_len, vocab_size = logits.shape

        # 获取top-k的值
        topk_values, topk_indices = torch.topk(logits, k=top_k, dim=-1)

        # 创建mask
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(dim=-1, index=topk_indices, src=torch.ones_like(topk_indices, dtype=torch.bool))

        if return_mask:
            return mask

        # 应用mask (非top-k的位置设为-inf)
        filtered_logits = logits.masked_fill(~mask, float('-inf'))

        return filtered_logits


class LossFunctions:
    """
    损失函数集合

    整合所有损失函数,提供统一接口
    """

    def __init__(
        self,
        vocab_size: int,
        device: str = "cuda"
    ):
        """
        Args:
            vocab_size: 词汇表大小
            device: 计算设备
        """
        self.vocab_size = vocab_size
        self.device = device

        # 初始化各个损失函数
        self.soft_nll_loss = SoftNLLLoss()
        self.bleu_loss = BLEULoss(vocab_size=vocab_size, device=device)
        self.topk_filter = TopKFilter()

        logger.info(f"Initialized LossFunctions with vocab_size={vocab_size}")

    def fluency_loss(
        self,
        pred_logits: torch.Tensor,
        target_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        流畅度损失 (使用Soft NLL)

        Args:
            pred_logits: 预测的logits
            target_logits: 目标logits (通常来自参考模型)

        Returns:
            损失值
        """
        return self.soft_nll_loss(pred_logits, target_logits)

    def rejection_word_loss(
        self,
        pred_logits: torch.Tensor,
        rejection_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        拒绝词损失 (使用BLEU)

        Args:
            pred_logits: 预测的logits
            rejection_logits: 拒绝词的logits

        Returns:
            损失值 (越高表示越相似,需要取负)
        """
        return self.bleu_loss(pred_logits, rejection_logits)

    def cross_entropy_loss(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        ignore_index: int = -100
    ) -> torch.Tensor:
        """
        交叉熵损失

        Args:
            logits: 预测的logits, shape [batch, seq_len, vocab_size]
            target_ids: 目标token IDs, shape [batch, seq_len]
            ignore_index: 忽略的索引 (如padding)

        Returns:
            损失值
        """
        batch_size, seq_len, vocab_size = logits.shape

        # Reshape为2D
        logits_2d = logits.reshape(-1, vocab_size)
        target_ids_1d = target_ids.reshape(-1)

        # 计算CE损失
        loss = F.cross_entropy(
            logits_2d,
            target_ids_1d,
            ignore_index=ignore_index,
            reduction='mean'
        )

        return loss

    def apply_topk_filter(
        self,
        logits: torch.Tensor,
        top_k: int
    ) -> torch.Tensor:
        """
        应用Top-K过滤

        Args:
            logits: 输入logits
            top_k: 保留的token数量

        Returns:
            过滤后的logits
        """
        return self.topk_filter.filter_3d(logits, top_k)
