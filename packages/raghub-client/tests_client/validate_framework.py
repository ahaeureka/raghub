#!/usr/bin/env python3
"""
Test script to validate the mode-specific testing framework
"""

import asyncio
import sys
import os

# Add the package to the path
sys.path.insert(0, "/app/packages/raghub-client/tests_client")

from base_test import BaseRAGTest
from config import TestConfig


async def test_mode_specific_naming():
    """Test that mode-specific instances have correct naming"""

    print("🧪 Testing mode-specific index naming...")

    # Test default mode
    default_test = BaseRAGTest()
    print(f"Default mode:")
    print(f"  - RAG mode: {default_test.rag_mode}")
    print(f"  - Knowledge ID: {default_test.test_knowledge_id}")
    print(f"  - Index name: {default_test.test_index_name}")

    # Test HippoRAG mode
    hipporag_test = BaseRAGTest(rag_mode="hipporag")
    print(f"\nHippoRAG mode:")
    print(f"  - RAG mode: {hipporag_test.rag_mode}")
    print(f"  - Knowledge ID: {hipporag_test.test_knowledge_id}")
    print(f"  - Index name: {hipporag_test.test_index_name}")

    # Test GraphRAG mode
    graphrag_test = BaseRAGTest(rag_mode="graphrag")
    print(f"\nGraphRAG mode:")
    print(f"  - RAG mode: {graphrag_test.rag_mode}")
    print(f"  - Knowledge ID: {graphrag_test.test_knowledge_id}")
    print(f"  - Index name: {graphrag_test.test_index_name}")

    # Test shared indices
    hipporag_shared = BaseRAGTest(use_shared_index=True, rag_mode="hipporag")
    graphrag_shared = BaseRAGTest(use_shared_index=True, rag_mode="graphrag")

    print(f"\nShared indices:")
    print(f"  - HippoRAG shared: {hipporag_shared.test_knowledge_id}")
    print(f"  - GraphRAG shared: {graphrag_shared.test_knowledge_id}")

    # Verify naming conventions
    assert "hipporag" in hipporag_test.test_knowledge_id
    assert "graphrag" in graphrag_test.test_knowledge_id
    assert "hipporag" in hipporag_shared.test_knowledge_id
    assert "graphrag" in graphrag_shared.test_knowledge_id
    assert "shared" in hipporag_shared.test_knowledge_id
    assert "shared" in graphrag_shared.test_knowledge_id

    # Verify indices are different
    assert hipporag_test.test_knowledge_id != graphrag_test.test_knowledge_id
    assert hipporag_shared.test_knowledge_id != graphrag_shared.test_knowledge_id
    assert hipporag_test.test_knowledge_id != hipporag_shared.test_knowledge_id

    print("\n✅ All mode-specific naming tests passed!")


def test_config_methods():
    """Test configuration methods"""

    print("\n🧪 Testing configuration methods...")

    # Test server config path selection
    default_config = TestConfig.get_server_config_path()
    hipporag_config = TestConfig.get_server_config_path("hipporag")
    graphrag_config = TestConfig.get_server_config_path("graphrag")

    print(f"Server config paths:")
    print(f"  - Default: {default_config}")
    print(f"  - HippoRAG: {hipporag_config}")
    print(f"  - GraphRAG: {graphrag_config}")

    # Test mode checking
    supported_modes = TestConfig.get_supported_modes()
    print(f"\nSupported modes: {supported_modes}")

    # Test current mode settings
    current_mode = TestConfig.RAG_MODE
    print(f"Current RAG mode: {current_mode}")

    for mode in ["hipporag", "graphrag"]:
        should_test = TestConfig.should_test_mode(mode)
        print(f"Should test {mode}: {should_test}")

    print("\n✅ All configuration tests passed!")


def test_file_existence():
    """Test that required files exist"""

    print("\n🧪 Testing file existence...")

    required_files = [
        "/app/configs/test-hipporag.toml",
        "/app/configs/test-graphrag.toml",
        "/app/packages/raghub-client/tests_client/conftest.py",
        "/app/packages/raghub-client/tests_client/base_test.py",
        "/app/packages/raghub-client/tests_client/config.py",
        "/app/packages/raghub-client/tests_client/server_manager.py",
        "/app/packages/raghub-client/tests_client/test_mode_specific.py",
    ]

    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            missing_files.append(file_path)

    if missing_files:
        print(f"\n❌ Missing files: {missing_files}")
        return False
    else:
        print("\n✅ All required files exist!")
        return True


async def main():
    """Run all validation tests"""
    print("🚀 Validating mode-specific testing framework...\n")

    try:
        # Test file existence
        if not test_file_existence():
            return False

        # Test configuration methods
        test_config_methods()

        # Test mode-specific naming
        await test_mode_specific_naming()

        print("\n🎉 All validation tests passed!")
        print("\n📚 Mode-specific testing framework is ready to use!")
        print("\nUsage examples:")
        print("  # Test both modes")
        print("  export RAGHUB_TEST_RAG_MODE=both")
        print("  pytest tests_client/test_mode_specific.py -v")
        print()
        print("  # Test only HippoRAG")
        print("  export RAGHUB_TEST_RAG_MODE=hipporag")
        print("  pytest tests_client/test_mode_specific.py::TestHippoRAGOnly -v")
        print()
        print("  # Test only GraphRAG")
        print("  export RAGHUB_TEST_RAG_MODE=graphrag")
        print("  pytest tests_client/test_mode_specific.py::TestGraphRAGOnly -v")

        return True

    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
