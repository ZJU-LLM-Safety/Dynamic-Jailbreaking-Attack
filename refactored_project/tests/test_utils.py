"""
工具模块单元测试

测试内容:
- TokenizationHelper功能
- I/O工具函数
- Logger配置
"""

import unittest
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import Mock, patch

from adversarial_attack.utils import (
    TokenizationHelper,
    save_jsonl,
    load_jsonl,
    save_json,
    load_json,
    ResultLogger,
)


class TestTokenizationHelper(unittest.TestCase):
    """测试TokenizationHelper"""

    def setUp(self):
        """测试前准备"""
        # 创建一个mock tokenizer
        self.mock_tokenizer = Mock()
        self.mock_tokenizer.vocab_size = 1000
        self.mock_tokenizer.pad_token = "[PAD]"
        self.mock_tokenizer.eos_token = "[EOS]"

        self.helper = TokenizationHelper(self.mock_tokenizer)

    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.helper.tokenizer)
        self.assertIsNotNone(self.helper.rejection_words)

    def test_has_rejection_words(self):
        """测试是否包含拒绝词"""
        # 默认拒绝词列表应该不为空
        self.assertGreater(len(self.helper.rejection_words), 0)

        # 应该包含常见拒绝词
        common_words = ["sorry", "cannot", "illegal"]
        for word in common_words:
            # 不区分大小写检查
            found = any(word.lower() in rw.lower() for rw in self.helper.rejection_words)
            self.assertTrue(found, f"Should contain '{word}' in rejection words")

    def test_default_rejection_words_list(self):
        """测试默认拒绝词列表"""
        from adversarial_attack.utils.tokenization import DEFAULT_REJECTION_WORDS

        self.assertIsInstance(DEFAULT_REJECTION_WORDS, list)
        self.assertGreater(len(DEFAULT_REJECTION_WORDS), 50)


class TestIOUtils(unittest.TestCase):
    """测试I/O工具函数"""

    def test_save_and_load_jsonl(self):
        """测试保存和加载JSONL"""
        data = [
            {"id": 1, "text": "hello"},
            {"id": 2, "text": "world"},
            {"id": 3, "text": "test"}
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            temp_path = f.name

        try:
            # 保存
            save_jsonl(data, temp_path)

            # 加载
            loaded_data = load_jsonl(temp_path)

            # 验证
            self.assertEqual(len(loaded_data), len(data))
            for original, loaded in zip(data, loaded_data):
                self.assertEqual(original, loaded)
        finally:
            os.unlink(temp_path)

    def test_save_and_load_json(self):
        """测试保存和加载JSON"""
        data = {
            "name": "test",
            "value": 123,
            "items": ["a", "b", "c"]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            # 保存
            save_json(data, temp_path)

            # 加载
            loaded_data = load_json(temp_path)

            # 验证
            self.assertEqual(loaded_data, data)
        finally:
            os.unlink(temp_path)

    def test_load_nonexistent_file(self):
        """测试加载不存在的文件"""
        with self.assertRaises(FileNotFoundError):
            load_json("nonexistent_file.json")

        with self.assertRaises(FileNotFoundError):
            load_jsonl("nonexistent_file.jsonl")

    def test_save_jsonl_with_path_object(self):
        """测试使用Path对象保存JSONL"""
        data = [{"test": "data"}]

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir) / "test.jsonl"

            save_jsonl(data, temp_path)

            self.assertTrue(temp_path.exists())

            loaded_data = load_jsonl(temp_path)
            self.assertEqual(loaded_data, data)


class TestResultLogger(unittest.TestCase):
    """测试ResultLogger"""

    def test_create_result_logger(self):
        """测试创建ResultLogger"""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "results.jsonl"

            logger = ResultLogger(save_path)

            self.assertEqual(logger.save_path, save_path)
            self.assertEqual(logger.results, [])

    def test_log_result(self):
        """测试记录结果"""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "results.jsonl"

            logger = ResultLogger(save_path)

            result = {"id": 1, "score": 0.95}
            logger.log(result)

            self.assertEqual(len(logger.results), 1)
            self.assertEqual(logger.results[0], result)

    def test_save_results(self):
        """测试保存结果"""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "results.jsonl"

            logger = ResultLogger(save_path)

            # 记录多个结果
            results = [
                {"id": 1, "score": 0.95},
                {"id": 2, "score": 0.87},
                {"id": 3, "score": 0.92}
            ]

            for result in results:
                logger.log(result)

            # 保存
            logger.save()

            # 验证文件存在
            self.assertTrue(save_path.exists())

            # 加载并验证
            loaded_results = load_jsonl(save_path)
            self.assertEqual(len(loaded_results), len(results))

    def test_auto_save(self):
        """测试自动保存"""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "results.jsonl"

            logger = ResultLogger(save_path, auto_save=True, save_frequency=2)

            # 记录结果
            logger.log({"id": 1})
            self.assertFalse(save_path.exists(), "Should not save yet")

            logger.log({"id": 2})
            self.assertTrue(save_path.exists(), "Should auto-save after 2 records")

    def test_get_results(self):
        """测试获取结果"""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "results.jsonl"

            logger = ResultLogger(save_path)

            results = [{"id": 1}, {"id": 2}]
            for result in results:
                logger.log(result)

            retrieved_results = logger.get_results()

            self.assertEqual(retrieved_results, results)


class TestLoggerSetup(unittest.TestCase):
    """测试Logger设置"""

    @patch('adversarial_attack.utils.logger.logger')
    def test_setup_logger_basic(self, mock_logger):
        """测试基础logger设置"""
        from adversarial_attack.utils.logger import setup_logger

        with tempfile.TemporaryDirectory() as tmpdir:
            setup_logger(
                log_dir=tmpdir,
                log_level="INFO",
                log_to_file=True,
                log_to_console=True
            )

            # 验证logger被配置
            # (实际调用会配置loguru,这里只是确保函数能运行)


if __name__ == '__main__':
    unittest.main()
