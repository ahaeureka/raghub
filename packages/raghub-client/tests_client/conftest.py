"""
RAGHub Client Tests Configuration

This conftest.py provides fixtures and configuration specific to raghub-client package tests.
Note: pytest_configure and pytest_collection_modifyitems hooks are handled
by the workspace-level conftest.py to avoid plugin conflicts.
"""

import logging
from typing import AsyncGenerator

import pytest
import pytest_asyncio

from .base_test import BaseRAGTest
from .config import TestConfig
from .server_manager import cleanup_server, ensure_server_running

__package_name__ = "raghub_client_tests"
__conftest_identifier__ = "raghub_client_conftest_20250703"


logger = logging.getLogger(__name__)


@pytest_asyncio.fixture(scope="session", autouse=True)
async def raghub_server():
    """
    Session-level fixture to start and stop RAGHub server automatically.

    This fixture ensures that:
    1. RAGHub server is started before any tests run (if AUTO_START_SERVER is True)
    2. Server is stopped after all tests complete (if we started it)
    3. Server health is verified before proceeding with tests

    Can be disabled by setting environment variable RAGHUB_AUTO_START_SERVER=false
    """
    if not TestConfig.AUTO_START_SERVER:
        logger.info("🔧 Server auto-start is disabled, assuming external server is running")

        # Still verify server is available
        from .server_manager import get_server_manager

        manager = await get_server_manager()
        if not await manager._is_server_healthy():
            pytest.fail(
                f"RAGHub server is not running at {TestConfig.BASE_URL}. "
                "Please start the server manually or set RAGHUB_AUTO_START_SERVER=true"
            )

        logger.info("✅ External RAGHub server is available")
        yield
        return

    logger.info("🚀 Starting RAGHub server for tests...")

    # Start server and wait for it to be ready
    await ensure_server_running()

    logger.info("✅ RAGHub server is ready for tests")

    yield

    # Cleanup after all tests complete
    logger.info("🧹 Stopping RAGHub server after tests...")
    await cleanup_server()
    logger.info("✅ RAGHub server stopped")


@pytest_asyncio.fixture(scope="session")
async def shared_index(raghub_server) -> AsyncGenerator[BaseRAGTest, None]:
    """
    Shared index fixture (session level)
    Creates a shared index for all tests in the session, suitable for read-only tests

    Args:
        raghub_server: Ensures server is running before creating the index
    """
    test_instance = BaseRAGTest(use_shared_index=True)

    try:
        await test_instance.setup_test_environment()
        logger.info(f"📚 Shared index created successfully: {test_instance.test_knowledge_id}")

        yield test_instance

    except Exception as e:
        logger.error(f"❌ Shared index creation failed: {e}")
        raise
    finally:
        # Clean up shared resources
        try:
            await test_instance.cleanup_test_environment()
            logger.info("🧹 Shared index cleanup completed")
        except Exception as e:
            logger.warning(f"⚠️ Shared index cleanup failed: {e}")


@pytest_asyncio.fixture(scope="function")
async def isolated_index(raghub_server) -> AsyncGenerator[BaseRAGTest, None]:
    """
    Isolated index fixture (function level)
    Creates independent index for each test function, suitable for tests that need to modify data

    Args:
        raghub_server: Ensures server is running before creating the index
    """
    test_instance = BaseRAGTest()

    try:
        await test_instance.setup_test_environment()
        logger.info(f"📁 Isolated index created successfully: {test_instance.test_knowledge_id}")

        yield test_instance

    except Exception as e:
        logger.error(f"❌ Isolated index creation failed: {e}")
        raise
    finally:
        try:
            await test_instance.cleanup_test_environment()
            logger.info(f"🗑️ Isolated index cleanup completed: {test_instance.test_knowledge_id}")
        except Exception as e:
            logger.warning(f"⚠️ Isolated index cleanup failed: {e}")


# Mode-specific fixtures
@pytest_asyncio.fixture(scope="function")
async def hipporag_index(raghub_server) -> AsyncGenerator[BaseRAGTest, None]:
    """
    HippoRAG mode isolated index fixture
    Creates independent index for HippoRAG mode tests

    Args:
        raghub_server: Ensures server is running before creating the index
    """
    test_instance = BaseRAGTest(rag_mode="hipporag")

    try:
        await test_instance.setup_test_environment()
        logger.info(f"🦛 HippoRAG index created successfully: {test_instance.test_knowledge_id}")

        yield test_instance

    except Exception as e:
        logger.error(f"❌ HippoRAG index creation failed: {e}")
        raise
    finally:
        try:
            await test_instance.cleanup_test_environment()
            logger.info(f"🗑️ HippoRAG index cleanup completed: {test_instance.test_knowledge_id}")
        except Exception as e:
            logger.warning(f"⚠️ HippoRAG index cleanup failed: {e}")


@pytest_asyncio.fixture(scope="function")
async def graphrag_index(raghub_server) -> AsyncGenerator[BaseRAGTest, None]:
    """
    GraphRAG mode isolated index fixture
    Creates independent index for GraphRAG mode tests

    Args:
        raghub_server: Ensures server is running before creating the index
    """
    test_instance = BaseRAGTest(rag_mode="graphrag")

    try:
        await test_instance.setup_test_environment()
        logger.info(f"📊 GraphRAG index created successfully: {test_instance.test_knowledge_id}")

        yield test_instance

    except Exception as e:
        logger.error(f"❌ GraphRAG index creation failed: {e}")
        raise
    finally:
        try:
            await test_instance.cleanup_test_environment()
            logger.info(f"🗑️ GraphRAG index cleanup completed: {test_instance.test_knowledge_id}")
        except Exception as e:
            logger.warning(f"⚠️ GraphRAG index cleanup failed: {e}")


@pytest_asyncio.fixture(scope="session")
async def hipporag_shared_index(raghub_server) -> AsyncGenerator[BaseRAGTest, None]:
    """
    HippoRAG mode shared index fixture (session level)
    Creates shared index for HippoRAG mode tests

    Args:
        raghub_server: Ensures server is running before creating the index
    """
    test_instance = BaseRAGTest(use_shared_index=True, rag_mode="hipporag")

    try:
        await test_instance.setup_test_environment()
        logger.info(f"🦛📚 HippoRAG shared index created successfully: {test_instance.test_knowledge_id}")

        yield test_instance

    except Exception as e:
        logger.error(f"❌ HippoRAG shared index creation failed: {e}")
        raise
    finally:
        try:
            await test_instance.cleanup_test_environment()
            logger.info("🧹 HippoRAG shared index cleanup completed")
        except Exception as e:
            logger.warning(f"⚠️ HippoRAG shared index cleanup failed: {e}")


@pytest_asyncio.fixture(scope="session")
async def graphrag_shared_index(raghub_server) -> AsyncGenerator[BaseRAGTest, None]:
    """
    GraphRAG mode shared index fixture (session level)
    Creates shared index for GraphRAG mode tests

    Args:
        raghub_server: Ensures server is running before creating the index
    """
    test_instance = BaseRAGTest(use_shared_index=True, rag_mode="graphrag")

    try:
        await test_instance.setup_test_environment()
        logger.info(f"📊📚 GraphRAG shared index created successfully: {test_instance.test_knowledge_id}")

        yield test_instance

    except Exception as e:
        logger.error(f"❌ GraphRAG shared index creation failed: {e}")
        raise
    finally:
        try:
            await test_instance.cleanup_test_environment()
            logger.info("🧹 GraphRAG shared index cleanup completed")
        except Exception as e:
            logger.warning(f"⚠️ GraphRAG shared index cleanup failed: {e}")


@pytest.fixture(scope="session", autouse=True)
def setup_test_logging():
    """Setup test logging format"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("test_output.log", mode="w", encoding="utf-8")],
    )

    # Set log levels for various modules
    logging.getLogger("raghub_client").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)


@pytest.fixture(autouse=True)
def test_info(request):
    """Automatically display test information"""
    test_name = request.node.name
    test_file = request.node.parent.name
    logger.info(f"🧪 Starting test: {test_file}::{test_name}")

    yield

    logger.info(f"✅ Test completed: {test_file}::{test_name}")


@pytest.fixture
def skip_if_no_server(raghub_server):
    """
    Skip test if server is not available

    Args:
        raghub_server: Dependency on server fixture to ensure it's running

    Note: This fixture now depends on raghub_server to ensure the server
    is automatically started, so it should never skip tests unless there's
    a startup failure.
    """
    # Since raghub_server fixture ensures server is running,
    # this fixture now mainly serves as a dependency marker
    pass


@pytest.fixture
def server_health_check():
    """
    Fixture to perform server health check before tests
    Can be used by individual tests that need to verify server state
    """
    import httpx

    async def check_health():
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{TestConfig.BASE_URL}/health")
                return response.status_code == 200
        except Exception:
            return False

    return check_health
