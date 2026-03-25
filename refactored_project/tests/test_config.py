"""
配置模块单元测试

测试内容:
- ModelConfig创建和验证
- AttackConfig创建和验证
- YAML/JSON配置加载
- 配置合并和更新
"""

import unittest
import tempfile
import os
import json
import yaml
from pathlib import Path

from adversarial_attack.config import (
    ModelConfig,
    AttackConfig,
    LoRAConfig,
    SuffixOptimizationConfig,
    ReferenceModelConfig,
    DeviceConfig,
    ModelType,
    AttackMethod,
)


class TestModelConfig(unittest.TestCase):
    """测试ModelConfig"""

    def test_create_basic_config(self):
        """测试创建基础配置"""
        config = ModelConfig(
            model_name_or_path="meta-llama/Llama-2-7b-chat-hf",
            device="cuda:0"
        )

        self.assertEqual(config.model_name_or_path, "meta-llama/Llama-2-7b-chat-hf")
        self.assertEqual(config.device, "cuda:0")

    def test_model_config_with_all_fields(self):
        """测试包含所有字段的配置"""
        config = ModelConfig(
            model_name_or_path="gpt-4",
            device="cuda:0",
            dtype="float16",
            client_name="openai",
            api_key="test-key",
            api_base="https://api.openai.com"
        )

        self.assertEqual(config.model_name_or_path, "gpt-4")
        self.assertEqual(config.client_name, "openai")
        self.assertEqual(config.api_key, "test-key")


class TestLoRAConfig(unittest.TestCase):
    """测试LoRAConfig"""

    def test_create_lora_config(self):
        """测试创建LoRA配置"""
        config = LoRAConfig(
            use_lora=True,
            lora_r=32,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"]
        )

        self.assertTrue(config.use_lora)
        self.assertEqual(config.lora_r, 32)
        self.assertEqual(config.lora_alpha, 16)
        self.assertEqual(len(config.target_modules), 2)

    def test_lora_config_defaults(self):
        """测试LoRA默认配置"""
        config = LoRAConfig()

        self.assertFalse(config.use_lora)
        self.assertEqual(config.lora_r, 32)


class TestSuffixOptimizationConfig(unittest.TestCase):
    """测试SuffixOptimizationConfig"""

    def test_create_suffix_config(self):
        """测试创建后缀优化配置"""
        config = SuffixOptimizationConfig(
            suffix_length=20,
            learning_rate=1.5,
            num_inner_iterations=400,
            suffix_topk=10
        )

        self.assertEqual(config.suffix_length, 20)
        self.assertEqual(config.learning_rate, 1.5)
        self.assertEqual(config.num_inner_iterations, 400)

    def test_suffix_config_validation(self):
        """测试配置验证"""
        # 正常情况
        config = SuffixOptimizationConfig(suffix_length=10)
        self.assertGreater(config.suffix_length, 0)


class TestAttackConfig(unittest.TestCase):
    """测试AttackConfig"""

    def test_create_basic_attack_config(self):
        """测试创建基础攻击配置"""
        config = AttackConfig(
            target_model=ModelConfig(
                model_name_or_path="meta-llama/Llama-2-7b-chat-hf"
            ),
            num_outer_iterations=20,
            num_inner_iterations=400
        )

        self.assertEqual(config.num_outer_iterations, 20)
        self.assertEqual(config.num_inner_iterations, 400)
        self.assertIsNotNone(config.target_model)

    def test_attack_config_with_all_components(self):
        """测试完整的攻击配置"""
        config = AttackConfig(
            target_model=ModelConfig(model_name_or_path="test-model"),
            lora_config=LoRAConfig(use_lora=True),
            suffix_config=SuffixOptimizationConfig(suffix_length=15),
            reference_model=ReferenceModelConfig(temperature=2.0),
            devices=DeviceConfig(
                target_device="cuda:0",
                reference_device="cuda:1",
                evaluator_device="cuda:2"
            )
        )

        self.assertIsNotNone(config.lora_config)
        self.assertIsNotNone(config.suffix_config)
        self.assertIsNotNone(config.reference_model)
        self.assertIsNotNone(config.devices)

    def test_attack_config_to_dict(self):
        """测试配置转换为字典"""
        config = AttackConfig(
            target_model=ModelConfig(model_name_or_path="test-model"),
            num_outer_iterations=10
        )

        config_dict = config.to_dict()
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict['num_outer_iterations'], 10)

    def test_attack_config_from_yaml(self):
        """测试从YAML加载配置"""
        # 创建临时YAML文件
        yaml_content = """
target_model:
  model_name_or_path: "meta-llama/Llama-2-7b-chat-hf"
  device: "cuda:0"

num_outer_iterations: 15
num_inner_iterations: 300
evaluator_type: "gptfuzzer"
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            config = AttackConfig.from_yaml(temp_path)
            self.assertEqual(config.num_outer_iterations, 15)
            self.assertEqual(config.num_inner_iterations, 300)
            self.assertEqual(config.evaluator_type, "gptfuzzer")
        finally:
            os.unlink(temp_path)

    def test_attack_config_from_json(self):
        """测试从JSON加载配置"""
        json_content = {
            "target_model": {
                "model_name_or_path": "test-model",
                "device": "cuda:0"
            },
            "num_outer_iterations": 25,
            "num_inner_iterations": 500
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(json_content, f)
            temp_path = f.name

        try:
            config = AttackConfig.from_json(temp_path)
            self.assertEqual(config.num_outer_iterations, 25)
            self.assertEqual(config.num_inner_iterations, 500)
        finally:
            os.unlink(temp_path)


class TestDeviceConfig(unittest.TestCase):
    """测试DeviceConfig"""

    def test_create_device_config(self):
        """测试创建设备配置"""
        config = DeviceConfig(
            target_device="cuda:0",
            reference_device="cuda:1",
            evaluator_device="cuda:2"
        )

        self.assertEqual(config.target_device, "cuda:0")
        self.assertEqual(config.reference_device, "cuda:1")
        self.assertEqual(config.evaluator_device, "cuda:2")

    def test_device_config_defaults(self):
        """测试设备配置默认值"""
        config = DeviceConfig()

        self.assertEqual(config.target_device, "cuda:0")


class TestEnums(unittest.TestCase):
    """测试枚举类型"""

    def test_model_type_enum(self):
        """测试ModelType枚举"""
        self.assertEqual(ModelType.LOCAL.value, "local")
        self.assertEqual(ModelType.API.value, "api")

    def test_attack_method_enum(self):
        """测试AttackMethod枚举"""
        self.assertEqual(AttackMethod.DYTA.value, "dyta")


if __name__ == '__main__':
    unittest.main()
