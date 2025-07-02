"""
RAGHub Workspace Level Pytest Configuration

Provides workspace-level pytest configuration and common fixtures
"""

__package_name__ = "raghub_workspace"
__conftest_identifier__ = "raghub_workspace_conftest_20250702"

import asyncio
import logging

import pytest

# Configure pytest-asyncio (must be defined in top-level conftest)
# pytest_plugins = ("pytest_asyncio",)

# Set global logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("test_output.log", mode="w", encoding="utf-8")],
)

# Set log levels for various modules
logging.getLogger("raghub_client").setLevel(logging.INFO)
logging.getLogger("raghub_core").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)


@pytest.fixture(scope="session")
def event_loop_policy():
    """Set event loop policy"""
    return asyncio.get_event_loop_policy()


@pytest.fixture(scope="session")
def event_loop(event_loop_policy):
    """Create session-level event loop"""
    loop = event_loop_policy.new_event_loop()
    yield loop
    loop.close()


# Workspace level configuration
def pytest_configure(config):
    """Workspace-level pytest configuration"""
    # Add workspace-level markers
    config.addinivalue_line("markers", "workspace: marks workspace-level tests")
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "client: marks tests as client-specific tests")
    config.addinivalue_line("markers", "core: marks tests as core-specific tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection behavior"""
    # Add default marker for tests without specific markers
    valid_markers = ["slow", "integration", "unit", "client", "core", "workspace"]
    for item in items:
        if not any(mark.name in valid_markers for mark in item.iter_markers()):
            item.add_marker(pytest.mark.unit)


@pytest.fixture(autouse=True)
def test_info(request):
    """Automatically display test information"""
    test_name = request.node.name
    test_file = request.node.parent.name
    logger = logging.getLogger(__name__)
    logger.info(f"🧪 Starting test: {test_file}::{test_name}")

    yield

    logger.info(f"✅ Test completed: {test_file}::{test_name}")
