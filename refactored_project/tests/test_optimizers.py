"""
优化器模块单元测试

测试内容:
- 损失函数计算
- SoftForward功能
- Top-K过滤
"""

import unittest
from unittest.mock import Mock, MagicMock
import torch
import torch.nn.functional as F

from adversarial_attack.optimizers import (
    LossFunctions,
    SoftNLLLoss,
    BLEULoss,
    TopKFilter,
)


class TestSoftNLLLoss(unittest.TestCase):
    """测试SoftNLLLoss"""

    def setUp(self):
        """测试前准备"""
        self.loss_fn = SoftNLLLoss()

    def test_forward_basic(self):
        """测试基础前向传播"""
        batch_size, seq_len, vocab_size = 2, 10, 1000

        # 创建随机logits
        pred_logits = torch.randn(batch_size, seq_len, vocab_size)
        target_logits = torch.randn(batch_size, seq_len, vocab_size)

        # 计算损失
        loss = self.loss_fn(pred_logits, target_logits)

        # 检查输出
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 1)  # 应该是[batch_size]
        self.assertEqual(loss.shape[0], batch_size)

    def test_loss_is_positive(self):
        """测试损失值为正"""
        pred_logits = torch.randn(1, 5, 100)
        target_logits = torch.randn(1, 5, 100)

        loss = self.loss_fn(pred_logits, target_logits)

        self.assertGreater(loss.item(), 0, "Loss should be positive")

    def test_identical_distributions(self):
        """测试相同分布的损失"""
        logits = torch.randn(1, 5, 100)

        # 相同的logits应该产生较小的损失
        loss = self.loss_fn(logits, logits)

        # KL散度应该接近0
        self.assertLess(loss.item(), 1.0)


class TestBLEULoss(unittest.TestCase):
    """测试BLEULoss"""

    def setUp(self):
        """测试前准备"""
        self.vocab_size = 100
        self.loss_fn = BLEULoss(vocab_size=self.vocab_size, max_n_gram=4, device="cpu")

    def test_forward_basic(self):
        """测试基础前向传播"""
        batch_size, seq_len = 2, 10

        pred_logits = torch.randn(batch_size, seq_len, self.vocab_size)
        target_logits = torch.randn(batch_size, seq_len, self.vocab_size)

        loss = self.loss_fn(pred_logits, target_logits)

        # 检查输出
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(loss.dim() == 0 or loss.dim() == 1)  # 标量或1D

    def test_loss_range(self):
        """测试损失值范围"""
        pred_logits = torch.randn(1, 10, self.vocab_size)
        target_logits = torch.randn(1, 10, self.vocab_size)

        loss = self.loss_fn(pred_logits, target_logits)

        # BLEU loss应该在合理范围内 (余弦相似度在[-1, 1])
        self.assertGreaterEqual(loss.item(), -1.0)
        self.assertLessEqual(loss.item(), 1.0)

    def test_identical_sequences(self):
        """测试相同序列"""
        logits = torch.randn(1, 10, self.vocab_size)

        loss = self.loss_fn(logits, logits)

        # 相同序列应该有高相似度
        self.assertGreater(loss.item(), 0.5)


class TestTopKFilter(unittest.TestCase):
    """测试TopKFilter"""

    def setUp(self):
        """测试前准备"""
        self.filter = TopKFilter()

    def test_filter_3d_basic(self):
        """测试3D过滤基础功能"""
        batch_size, seq_len, vocab_size = 2, 10, 1000
        top_k = 50

        logits = torch.randn(batch_size, seq_len, vocab_size)

        filtered_logits = self.filter.filter_3d(logits, top_k)

        # 检查形状
        self.assertEqual(filtered_logits.shape, logits.shape)

        # 检查每个位置只有top_k个非-inf值
        for b in range(batch_size):
            for s in range(seq_len):
                non_inf_count = torch.isfinite(filtered_logits[b, s]).sum()
                self.assertEqual(non_inf_count.item(), top_k)

    def test_filter_preserves_top_values(self):
        """测试过滤保留最高值"""
        logits = torch.randn(1, 5, 100)
        top_k = 10

        # 获取原始top-k值
        original_topk_values, original_topk_indices = torch.topk(logits, k=top_k, dim=-1)

        # 过滤
        filtered_logits = self.filter.filter_3d(logits, top_k)

        # 检查top-k值是否保留
        for b in range(logits.shape[0]):
            for s in range(logits.shape[1]):
                for k in range(top_k):
                    idx = original_topk_indices[b, s, k]
                    original_val = original_topk_values[b, s, k]
                    filtered_val = filtered_logits[b, s, idx]

                    self.assertAlmostEqual(
                        original_val.item(),
                        filtered_val.item(),
                        places=5
                    )

    def test_filter_return_mask(self):
        """测试返回mask"""
        logits = torch.randn(1, 5, 100)
        top_k = 20

        mask = self.filter.filter_3d(logits, top_k, return_mask=True)

        # 检查mask类型
        self.assertEqual(mask.dtype, torch.bool)

        # 检查mask中True的数量
        true_count = mask[0, 0].sum()
        self.assertEqual(true_count.item(), top_k)


class TestLossFunctions(unittest.TestCase):
    """测试LossFunctions集合"""

    def setUp(self):
        """测试前准备"""
        self.vocab_size = 100
        self.loss_fns = LossFunctions(vocab_size=self.vocab_size, device="cpu")

    def test_fluency_loss(self):
        """测试流畅度损失"""
        pred_logits = torch.randn(1, 10, self.vocab_size)
        target_logits = torch.randn(1, 10, self.vocab_size)

        loss = self.loss_fns.fluency_loss(pred_logits, target_logits)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(loss.item(), 0)

    def test_rejection_word_loss(self):
        """测试拒绝词损失"""
        pred_logits = torch.randn(1, 10, self.vocab_size)
        rejection_logits = torch.randn(1, 10, self.vocab_size)

        loss = self.loss_fns.rejection_word_loss(pred_logits, rejection_logits)

        self.assertIsInstance(loss, torch.Tensor)

    def test_cross_entropy_loss(self):
        """测试交叉熵损失"""
        batch_size, seq_len = 2, 10

        logits = torch.randn(batch_size, seq_len, self.vocab_size)
        target_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len))

        loss = self.loss_fns.cross_entropy_loss(logits, target_ids)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(loss.item(), 0)

    def test_cross_entropy_loss_with_ignore_index(self):
        """测试带ignore_index的交叉熵损失"""
        logits = torch.randn(2, 10, self.vocab_size)
        target_ids = torch.randint(0, self.vocab_size, (2, 10))

        # 设置一些位置为ignore_index
        target_ids[0, 5:] = -100

        loss = self.loss_fns.cross_entropy_loss(logits, target_ids, ignore_index=-100)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(loss.item(), 0)

    def test_apply_topk_filter(self):
        """测试应用Top-K过滤"""
        logits = torch.randn(1, 10, self.vocab_size)
        top_k = 20

        filtered_logits = self.loss_fns.apply_topk_filter(logits, top_k)

        # 检查过滤效果
        self.assertEqual(filtered_logits.shape, logits.shape)

        # 每个位置应该只有top_k个有限值
        non_inf_count = torch.isfinite(filtered_logits[0, 0]).sum()
        self.assertEqual(non_inf_count.item(), top_k)


class TestLossFunctionsIntegration(unittest.TestCase):
    """测试损失函数集成"""

    def test_combined_loss_calculation(self):
        """测试组合损失计算"""
        vocab_size = 100
        loss_fns = LossFunctions(vocab_size=vocab_size, device="cpu")

        # 创建测试数据
        pred_logits = torch.randn(1, 10, vocab_size)
        target_logits = torch.randn(1, 10, vocab_size)
        target_ids = torch.randint(0, vocab_size, (1, 10))
        rejection_logits = torch.randn(1, 10, vocab_size)

        # 计算各种损失
        ce_loss = loss_fns.cross_entropy_loss(pred_logits, target_ids)
        flu_loss = loss_fns.fluency_loss(pred_logits, target_logits)
        rej_loss = loss_fns.rejection_word_loss(pred_logits, rejection_logits)

        # 组合损失 (类似DyTA中的做法)
        total_loss = 100 * ce_loss + 1 * flu_loss - 10 * rej_loss

        self.assertIsInstance(total_loss, torch.Tensor)


if __name__ == '__main__':
    unittest.main()
