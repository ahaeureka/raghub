"""
RAGHub Workspace Level Pytest Configuration

提供workspace级别的pytest配置和通用fixtures
"""

import asyncio
import logging
import pytest

# 配置 pytest-asyncio (必须在顶级conftest中定义)
pytest_plugins = ("pytest_asyncio",)

# 设置全局日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("test_output.log", mode="w", encoding="utf-8")],
)

# 设置各个模块的日志级别
logging.getLogger("raghub_client").setLevel(logging.INFO)
logging.getLogger("raghub_core").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)


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


# Workspace level configuration
def pytest_configure(config):
    """Workspace级别的pytest配置"""
    # 添加workspace级别的标记
    config.addinivalue_line("markers", "workspace: 标记workspace级别的测试")
    config.addinivalue_line("markers", "slow: 标记耗时较长的测试")
    config.addinivalue_line("markers", "integration: 标记集成测试")
    config.addinivalue_line("markers", "unit: 标记单元测试")


def pytest_collection_modifyitems(config, items):
    """修改测试收集行为"""
    # 为没有标记的测试添加默认标记
    for item in items:
        if not any(mark.name in ["slow", "integration", "unit"] for mark in item.iter_markers()):
            item.add_marker(pytest.mark.unit)


@pytest.fixture(autouse=True)
def test_info(request):
    """自动显示测试信息"""
    test_name = request.node.name
    test_file = request.node.parent.name
    logger = logging.getLogger(__name__)
    logger.info(f"🧪 开始测试: {test_file}::{test_name}")

    yield

    logger.info(f"✅ 测试完成: {test_file}::{test_name}")
