"""
RAGHub Client Tests - Pytest配置文件

提供客户端测试的fixtures和配置
"""

import asyncio
import logging
from typing import AsyncGenerator

import pytest
import pytest_asyncio

# 使用绝对导入避免模块路径冲突
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from base_test import BaseRAGTest
from config import TestConfig

logger = logging.getLogger(__name__)

# 不再在非顶级conftest中定义pytest_plugins
# pytest_plugins已移动到workspace级别的conftest.py


# 配置 asyncio 测试模式
@pytest.fixture(scope="session")
def event_loop_policy():
    """设置事件循环策略"""
    return asyncio.get_event_loop_policy()


@pytest.fixture(scope="session")
def event_loop(event_loop_policy):
    """创建 session 级别的事件循环"""
    loop = event_loop_policy.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def shared_index() -> AsyncGenerator[BaseRAGTest, None]:
    """
    共享索引 fixture (session 级别)
    在整个测试会话中只创建一次索引，所有测试共享使用
    """
    logger.info("🚀 初始化共享测试索引...")

    # 创建共享测试实例
    test_instance = BaseRAGTest()

    # 使用固定的索引名称以便共享
    test_instance.test_knowledge_id = f"{TestConfig.TEST_KNOWLEDGE_ID}_shared"

    try:
        # 创建索引和添加文档
        await test_instance.setup_test_environment()
        logger.info(f"✅ 共享索引创建成功: {test_instance.test_knowledge_id}")

        yield test_instance

    except Exception as e:
        logger.error(f"❌ 共享索引创建失败: {e}")
        raise
    finally:
        # 清理共享资源
        try:
            await test_instance.cleanup_test_environment()
            logger.info("🧹 共享索引清理完成")
        except Exception as e:
            logger.warning(f"⚠️ 共享索引清理失败: {e}")


@pytest_asyncio.fixture(scope="function")
async def isolated_index() -> AsyncGenerator[BaseRAGTest, None]:
    """
    隔离索引 fixture (function 级别)
    为每个测试函数创建独立的索引，适用于需要修改数据的测试
    """
    test_instance = BaseRAGTest()

    try:
        await test_instance.setup_test_environment()
        logger.info(f"📁 独立索引创建成功: {test_instance.test_knowledge_id}")

        yield test_instance

    except Exception as e:
        logger.error(f"❌ 独立索引创建失败: {e}")
        raise
    finally:
        try:
            await test_instance.cleanup_test_environment()
            logger.info(f"🗑️ 独立索引清理完成: {test_instance.test_knowledge_id}")
        except Exception as e:
            logger.warning(f"⚠️ 独立索引清理失败: {e}")


@pytest.fixture(scope="session", autouse=True)
def setup_test_logging():
    """设置测试日志格式"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("test_output.log", mode="w", encoding="utf-8")],
    )

    # 设置各个模块的日志级别
    logging.getLogger("raghub_client").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)


@pytest.fixture(autouse=True)
def test_info(request):
    """自动显示测试信息"""
    test_name = request.node.name
    test_file = request.node.parent.name
    logger.info(f"🧪 开始测试: {test_file}::{test_name}")

    yield

    logger.info(f"✅ 测试完成: {test_file}::{test_name}")


def pytest_configure(config):
    """pytest 配置"""
    # 添加自定义标记
    config.addinivalue_line("markers", "slow: 标记耗时较长的测试")
    config.addinivalue_line("markers", "integration: 标记集成测试")
    config.addinivalue_line("markers", "unit: 标记单元测试")


def pytest_collection_modifyitems(config, items):
    """修改测试收集行为"""
    # 为没有标记的测试添加默认标记
    for item in items:
        if not any(mark.name in ["slow", "integration", "unit"] for mark in item.iter_markers()):
            item.add_marker(pytest.mark.unit)


@pytest.fixture
def skip_if_no_server():
    """如果服务器不可用则跳过测试"""
    import httpx

    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{TestConfig.BASE_URL}/health")
            if response.status_code != 200:
                pytest.skip("RAGHub 服务器不可用")
    except Exception:
        pytest.skip("无法连接到 RAGHub 服务器")
