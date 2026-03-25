#!/usr/bin/env python
"""
测试运行脚本 / Test Runner Script

方便的测试运行脚本，提供多种测试运行选项
Convenient test runner script with multiple testing options
"""

import sys
import os
import unittest
import argparse
from pathlib import Path


def run_all_tests(verbosity=2):
    """运行所有测试 / Run all tests"""
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(os.path.abspath(__file__))
    suite = loader.discover(start_dir, pattern='test_*.py')

    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)

    return result.wasSuccessful()


def run_specific_module(module_name, verbosity=2):
    """运行特定模块的测试 / Run tests for a specific module"""
    loader = unittest.TestLoader()

    # 导入测试模块 / Import test module
    module_path = f'tests.{module_name}'
    try:
        suite = loader.loadTestsFromName(module_path)
    except (ImportError, AttributeError) as e:
        print(f"错误: 无法加载测试模块 '{module_name}': {e}")
        print(f"Error: Cannot load test module '{module_name}': {e}")
        return False

    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)

    return result.wasSuccessful()


def list_test_modules():
    """列出所有可用的测试模块 / List all available test modules"""
    test_dir = Path(__file__).parent
    test_files = sorted(test_dir.glob('test_*.py'))

    print("可用的测试模块 / Available test modules:")
    print("-" * 50)
    for test_file in test_files:
        module_name = test_file.stem
        print(f"  - {module_name}")
    print("-" * 50)


def print_test_summary():
    """打印测试总结 / Print test summary"""
    test_files = {
        'test_config': '配置模块测试 / Config module tests',
        'test_models': '模型模块测试 / Model module tests',
        'test_evaluators': '评估器模块测试 / Evaluator module tests',
        'test_optimizers': '优化器模块测试 / Optimizer module tests',
        'test_utils': '工具模块测试 / Utils module tests',
    }

    print("\n" + "=" * 60)
    print("Adversarial Attack 单元测试套件 / Unit Test Suite")
    print("=" * 60)
    print("\n测试模块 / Test Modules:")
    for module, description in test_files.items():
        print(f"  ✓ {module:20s} - {description}")
    print("\n" + "=" * 60 + "\n")


def main():
    """主函数 / Main function"""
    parser = argparse.ArgumentParser(
        description='Adversarial Attack 测试运行器 / Test Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例 / Examples:
  %(prog)s                          # 运行所有测试 / Run all tests
  %(prog)s -m test_config           # 运行配置测试 / Run config tests
  %(prog)s -m test_evaluators -v 1  # 运行评估器测试(简化输出) / Run evaluator tests (minimal output)
  %(prog)s --list                   # 列出所有测试模块 / List all test modules
        """
    )

    parser.add_argument(
        '-m', '--module',
        type=str,
        help='运行特定测试模块 / Run specific test module (e.g., test_config)'
    )

    parser.add_argument(
        '-v', '--verbosity',
        type=int,
        choices=[0, 1, 2],
        default=2,
        help='输出详细程度 / Verbosity level (0=quiet, 1=normal, 2=verbose)'
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='列出所有可用的测试模块 / List all available test modules'
    )

    parser.add_argument(
        '--summary',
        action='store_true',
        help='显示测试总结 / Show test summary'
    )

    args = parser.parse_args()

    # 列出测试模块 / List test modules
    if args.list:
        list_test_modules()
        return 0

    # 显示测试总结 / Show test summary
    if args.summary:
        print_test_summary()
        return 0

    # 打印头部信息 / Print header
    print_test_summary()

    # 运行测试 / Run tests
    try:
        if args.module:
            print(f"运行测试模块: {args.module}")
            print(f"Running test module: {args.module}\n")
            success = run_specific_module(args.module, args.verbosity)
        else:
            print("运行所有测试...")
            print("Running all tests...\n")
            success = run_all_tests(args.verbosity)

        # 打印结果 / Print results
        print("\n" + "=" * 60)
        if success:
            print("✅ 所有测试通过! / All tests passed!")
        else:
            print("❌ 部分测试失败 / Some tests failed")
        print("=" * 60 + "\n")

        return 0 if success else 1

    except KeyboardInterrupt:
        print("\n\n测试被用户中断 / Tests interrupted by user")
        return 130


if __name__ == '__main__':
    # 确保可以导入adversarial_attack包 / Ensure adversarial_attack package can be imported
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    sys.exit(main())
