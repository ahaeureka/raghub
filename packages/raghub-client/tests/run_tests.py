"""
RAGHub Client 测试运行器
提供便捷的测试执行脚本
"""

import logging
import sys
from pathlib import Path

import pytest


def setup_logging():
    """设置测试日志"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("test_results.log")],
    )


def run_tests(test_pattern: str = None, verbose: bool = True):
    """运行测试"""
    setup_logging()

    # 测试参数
    args = ["-v"] if verbose else []

    if test_pattern:
        args.extend(["-k", test_pattern])

    # 添加异步支持
    args.extend(["--asyncio-mode=auto", "--tb=short", "--durations=10"])

    # 运行测试
    current_dir = Path(__file__).parent
    test_dir = str(current_dir)

    print("运行RAGHub Client测试...")
    print(f"测试目录: {test_dir}")
    print(f"测试参数: {' '.join(args)}")
    print("-" * 50)

    return pytest.main([test_dir] + args)


def run_specific_tests():
    """运行特定类型的测试"""
    test_suites = {
        "1": ("检索测试", "test_retrieval"),
        "2": ("聊天测试", "test_chat"),
        "3": ("文档管理测试", "test_documents"),
        "4": ("索引管理测试", "test_index"),
        "5": ("集成测试", "test_integration"),
        "6": ("全部测试", None),
    }

    print("请选择要运行的测试套件:")
    for key, (name, _) in test_suites.items():
        print(f"{key}. {name}")

    choice = input("\n请输入选择 (1-6): ").strip()

    if choice in test_suites:
        name, pattern = test_suites[choice]
        print(f"\n运行 {name}...")
        return run_tests(pattern)
    else:
        print("无效选择")
        return 1


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 命令行参数模式
        if sys.argv[1] == "--interactive":
            exit_code = run_specific_tests()
        else:
            exit_code = run_tests(sys.argv[1] if sys.argv[1] != "--all" else None)
    else:
        # 交互模式
        exit_code = run_specific_tests()

    sys.exit(exit_code)
