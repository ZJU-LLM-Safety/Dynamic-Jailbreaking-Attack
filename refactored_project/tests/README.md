# 单元测试文档 / Unit Tests Documentation

本目录包含adversarial_attack项目的完整单元测试套件。

This directory contains the complete unit test suite for the adversarial_attack project.

## 📁 测试文件结构 / Test Structure

```
tests/
├── __init__.py                 # 测试包初始化 / Test package init
├── test_config.py             # 配置模块测试 / Config module tests
├── test_models.py             # 模型模块测试 / Model module tests
├── test_evaluators.py         # 评估器模块测试 / Evaluator module tests
├── test_optimizers.py         # 优化器模块测试 / Optimizer module tests
├── test_utils.py              # 工具模块测试 / Utils module tests
├── pytest.ini                 # Pytest配置 / Pytest configuration
└── README.md                  # 本文档 / This document
```

## 🎯 测试覆盖范围 / Test Coverage

### 1. 配置模块测试 (test_config.py)
- ✅ ModelConfig 创建和验证
- ✅ LoRAConfig 创建和验证
- ✅ AttackConfig 完整配置
- ✅ YAML/JSON 配置文件加载
- ✅ 配置保存和导出

**测试类 / Test Classes:**
- `TestModelConfig` - 模型配置测试
- `TestLoRAConfig` - LoRA配置测试
- `TestAttackConfig` - 攻击配置测试

### 2. 模型模块测试 (test_models.py)
- ✅ ModelFactory 自动模型识别
- ✅ API模型提供商检测
- ✅ 本地模型创建
- ✅ API模型接口验证
- ✅ 模型信息获取

**测试类 / Test Classes:**
- `TestModelFactory` - 模型工厂测试
- `TestAPIModel` - API模型测试
- `TestLocalModel` - 本地模型测试

### 3. 评估器模块测试 (test_evaluators.py)
- ✅ AutoDAN 关键词匹配评估
- ✅ GPTFuzzer 评估器接口
- ✅ LlamaGuard3 评估器接口
- ✅ EvaluatorWrapper 评估器选择
- ✅ 批量评估和ASR计算

**测试类 / Test Classes:**
- `TestAutoDANEvaluator` - AutoDAN评估器测试
- `TestGPTFuzzerEvaluator` - GPTFuzzer评估器测试
- `TestLlamaGuard3Evaluator` - LlamaGuard3评估器测试
- `TestEvaluatorWrapper` - 评估器包装器测试

### 4. 优化器模块测试 (test_optimizers.py)
- ✅ SoftNLLLoss 损失计算
- ✅ BLEULoss 流畅度损失
- ✅ TopKFilter token过滤
- ✅ 交叉熵损失计算
- ✅ 拒绝词损失计算
- ✅ 流畅度损失计算

**测试类 / Test Classes:**
- `TestSoftNLLLoss` - 软NLL损失测试
- `TestBLEULoss` - BLEU损失测试
- `TestTopKFilter` - TopK过滤器测试
- `TestLossFunctions` - 损失函数测试

### 5. 工具模块测试 (test_utils.py)
- ✅ TokenizationHelper 拒绝词检测
- ✅ JSONL 文件保存和加载
- ✅ JSON 文件保存和加载
- ✅ ResultLogger 结果记录
- ✅ 自动保存功能

**测试类 / Test Classes:**
- `TestTokenizationHelper` - 分词辅助工具测试
- `TestIOUtils` - I/O工具测试
- `TestResultLogger` - 结果记录器测试
- `TestLoggerSetup` - Logger设置测试

## 🚀 运行测试 / Running Tests

### 方法1: 使用 unittest (推荐基础测试)

```bash
# 运行所有测试 / Run all tests
cd /data/home/Kedong/repos/Dynamic-Target-Prompt-Attacker/refactored_project
python -m unittest discover tests

# 运行特定测试文件 / Run specific test file
python -m unittest tests.test_config
python -m unittest tests.test_models
python -m unittest tests.test_evaluators
python -m unittest tests.test_optimizers
python -m unittest tests.test_utils

# 运行特定测试类 / Run specific test class
python -m unittest tests.test_config.TestModelConfig

# 运行特定测试方法 / Run specific test method
python -m unittest tests.test_config.TestModelConfig.test_create_basic_config

# 详细输出 / Verbose output
python -m unittest discover tests -v
```

### 方法2: 使用 pytest (推荐高级功能)

```bash
# 安装pytest (如果未安装) / Install pytest if not installed
pip install pytest pytest-cov

# 运行所有测试 / Run all tests
pytest tests/

# 详细输出 / Verbose output
pytest tests/ -v

# 运行特定测试文件 / Run specific test file
pytest tests/test_config.py
pytest tests/test_evaluators.py

# 显示测试覆盖率 / Show test coverage
pytest tests/ --cov=adversarial_attack --cov-report=html

# 只运行失败的测试 / Run only failed tests
pytest tests/ --lf

# 并行运行测试 (需要 pytest-xdist) / Run tests in parallel
pip install pytest-xdist
pytest tests/ -n auto
```

### 方法3: 使用测试脚本

```bash
# 运行测试脚本 / Run test script
python tests/run_tests.py
```

## 📊 测试统计 / Test Statistics

- **测试文件数 / Test Files**: 5
- **测试类数 / Test Classes**: 17
- **测试代码行数 / Lines of Test Code**: ~1,100
- **覆盖模块数 / Modules Covered**: 6 (config, models, evaluators, optimizers, utils, core)

## 🔧 测试环境要求 / Test Environment Requirements

### 必需依赖 / Required Dependencies

```bash
pip install torch transformers
pip install unittest  # Python标准库 / Python stdlib
```

### 可选依赖 (增强测试功能) / Optional Dependencies

```bash
pip install pytest pytest-cov pytest-xdist
pip install coverage
```

## 📝 测试设计原则 / Test Design Principles

### 1. 隔离性 / Isolation
- 使用 `unittest.mock.Mock` 模拟外部依赖
- 使用 `tempfile` 创建临时文件，测试后自动清理
- 每个测试方法相互独立

### 2. 全面性 / Comprehensiveness
- 测试正常情况 (Happy path)
- 测试边界情况 (Edge cases)
- 测试错误情况 (Error cases)
- 测试集成场景 (Integration scenarios)

### 3. 可读性 / Readability
- 清晰的测试方法命名 (`test_<功能>_<场景>`)
- 详细的文档字符串
- 使用注释解释复杂逻辑

### 4. 可维护性 / Maintainability
- 使用 `setUp()` 初始化测试数据
- 使用 `tearDown()` 清理测试环境
- 避免代码重复

## 🐛 调试测试 / Debugging Tests

### 打印调试信息 / Print Debug Info

```python
# 在测试中添加打印
def test_something(self):
    result = some_function()
    print(f"Debug: result = {result}")
    self.assertEqual(result, expected)
```

### 使用断点调试 / Use Breakpoints

```python
import pdb

def test_something(self):
    result = some_function()
    pdb.set_trace()  # 设置断点 / Set breakpoint
    self.assertEqual(result, expected)
```

### 只运行一个测试 / Run Single Test

```bash
# 只运行一个测试方便调试 / Run only one test for debugging
python -m unittest tests.test_config.TestModelConfig.test_create_basic_config -v
```

## ⚠️ 注意事项 / Important Notes

### 1. Mock对象的使用
由于某些测试需要真实模型文件或API密钥，我们使用Mock对象模拟:
- HuggingFace模型加载
- API调用
- GPU操作

### 2. 资源清理
测试使用 `tempfile` 模块创建临时文件，会自动清理，但如果测试中断可能留下临时文件。

### 3. GPU测试
某些测试涉及GPU操作，如果没有GPU可用，这些测试会被跳过或使用CPU模拟。

### 4. API密钥
API模型测试不需要真实API密钥，使用Mock模拟API响应。

## 📈 持续改进 / Continuous Improvement

### 待添加的测试 / Tests to Add

- ⚠️ 端到端集成测试 / End-to-end integration tests
- ⚠️ 性能基准测试 / Performance benchmark tests
- ⚠️ 压力测试 / Stress tests
- ⚠️ 更多边界情况 / More edge cases

### 提高覆盖率 / Improve Coverage

```bash
# 生成覆盖率报告 / Generate coverage report
coverage run -m pytest tests/
coverage report
coverage html  # 生成HTML报告 / Generate HTML report
```

## 🤝 贡献指南 / Contributing Guidelines

### 添加新测试 / Adding New Tests

1. 在相应的测试文件中添加测试方法
2. 遵循现有的命名约定
3. 添加清晰的文档字符串
4. 确保测试相互独立
5. 运行测试确保通过

### 测试命名约定 / Test Naming Convention

```python
def test_<function_name>_<scenario>(self):
    """测试<功能>在<场景>下的行为"""
    pass

# 示例 / Examples:
def test_create_config_basic(self):
    """测试基础配置创建"""

def test_create_config_with_lora(self):
    """测试带LoRA的配置创建"""

def test_create_config_invalid_device(self):
    """测试无效设备配置"""
```

## 📚 参考资料 / References

- [Python unittest 文档](https://docs.python.org/3/library/unittest.html)
- [pytest 文档](https://docs.pytest.org/)
- [unittest.mock 文档](https://docs.python.org/3/library/unittest.mock.html)

---

**最后更新 / Last Updated**: 2025-01-27
**版本 / Version**: 1.0.0
**状态 / Status**: ✅ 完成 / Complete
