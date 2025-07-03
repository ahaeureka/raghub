"""
RAGHub Client Mode-Specific Tests

This module demonstrates how to use the new mode-specific fixtures
to test HippoRAG and GraphRAG modes with separate indices.
"""

import logging

import pytest

from .base_test import BaseRAGTest
from .config import TestConfig

logger = logging.getLogger(__name__)


class TestModeSpecific:
    """Test class for mode-specific testing"""

    @pytest.mark.asyncio
    async def test_hipporag_mode_only(self, hipporag_index: BaseRAGTest):
        """Test that only uses HippoRAG mode with its own index"""
        logger.info("Testing HippoRAG mode with dedicated index")

        # Create a retrieval request
        query = TestConfig.TEST_QUERIES[0]
        request = hipporag_index.create_test_retrieval_request(query)

        # Perform retrieval
        response = await hipporag_index.client.rag_service_retrieve(request)

        # Validate response
        hipporag_index.assert_retrieval_response(response)

        # Verify the index name contains hipporag mode
        assert "hipporag" in hipporag_index.test_knowledge_id
        logger.info(f"✅ HippoRAG test completed with index: {hipporag_index.test_knowledge_id}")

    @pytest.mark.asyncio
    async def test_graphrag_mode_only(self, graphrag_index: BaseRAGTest):
        """Test that only uses GraphRAG mode with its own index"""
        logger.info("Testing GraphRAG mode with dedicated index")

        # Create a retrieval request
        query = TestConfig.TEST_QUERIES[1]
        request = graphrag_index.create_test_retrieval_request(query)

        # Perform retrieval
        response = await graphrag_index.client.rag_service_retrieve(request)

        # Validate response
        graphrag_index.assert_retrieval_response(response)

        # Verify the index name contains graphrag mode
        assert "graphrag" in graphrag_index.test_knowledge_id
        logger.info(f"✅ GraphRAG test completed with index: {graphrag_index.test_knowledge_id}")

    @pytest.mark.asyncio
    async def test_both_modes_comparison(self, hipporag_index: BaseRAGTest, graphrag_index: BaseRAGTest):
        """Test that compares both modes using separate indices"""
        logger.info("Testing both modes with separate indices")

        query = TestConfig.TEST_QUERIES[0]

        # Test HippoRAG mode
        hipporag_request = hipporag_index.create_test_retrieval_request(query)
        hipporag_response = await hipporag_index.client.rag_service_retrieve(hipporag_request)

        # Test GraphRAG mode
        graphrag_request = graphrag_index.create_test_retrieval_request(query)
        graphrag_response = await graphrag_index.client.rag_service_retrieve(graphrag_request)

        # Validate both responses
        hipporag_index.assert_retrieval_response(hipporag_response)
        graphrag_index.assert_retrieval_response(graphrag_response)

        # Verify indices are different
        assert hipporag_index.test_knowledge_id != graphrag_index.test_knowledge_id
        assert "hipporag" in hipporag_index.test_knowledge_id
        assert "graphrag" in graphrag_index.test_knowledge_id

        logger.info("✅ Both modes test completed:")
        logger.info(f"  HippoRAG index: {hipporag_index.test_knowledge_id}")
        logger.info(f"  GraphRAG index: {graphrag_index.test_knowledge_id}")

    @pytest.mark.asyncio
    async def test_shared_indices_different_modes(
        self, hipporag_shared_index: BaseRAGTest, graphrag_shared_index: BaseRAGTest
    ):
        """Test using shared indices for different modes"""
        logger.info("Testing shared indices for different modes")

        query = TestConfig.TEST_QUERIES[2]

        # Test HippoRAG shared index
        hipporag_request = hipporag_shared_index.create_test_retrieval_request(query)
        hipporag_response = await hipporag_shared_index.client.rag_service_retrieve(hipporag_request)

        # Test GraphRAG shared index
        graphrag_request = graphrag_shared_index.create_test_retrieval_request(query)
        graphrag_response = await graphrag_shared_index.client.rag_service_retrieve(graphrag_request)

        # Validate both responses
        hipporag_shared_index.assert_retrieval_response(hipporag_response)
        graphrag_shared_index.assert_retrieval_response(graphrag_response)

        # Verify indices are different and contain mode identifiers
        assert hipporag_shared_index.test_knowledge_id != graphrag_shared_index.test_knowledge_id
        assert "hipporag" in hipporag_shared_index.test_knowledge_id
        assert "graphrag" in graphrag_shared_index.test_knowledge_id
        assert "shared" in hipporag_shared_index.test_knowledge_id
        assert "shared" in graphrag_shared_index.test_knowledge_id

        logger.info("✅ Shared indices test completed:")
        logger.info(f"  HippoRAG shared index: {hipporag_shared_index.test_knowledge_id}")
        logger.info(f"  GraphRAG shared index: {graphrag_shared_index.test_knowledge_id}")


class TestModeConfiguration:
    """Test class for mode configuration validation"""

    @pytest.mark.asyncio
    async def test_mode_specific_index_naming(self, hipporag_index: BaseRAGTest):
        """Test that mode-specific indices have correct naming"""
        assert hipporag_index.rag_mode == "hipporag"
        assert "hipporag" in hipporag_index.test_knowledge_id
        assert "hipporag" in hipporag_index.test_index_name

        logger.info("✅ HippoRAG mode configuration validated:")
        logger.info(f"  Mode: {hipporag_index.rag_mode}")
        logger.info(f"  Knowledge ID: {hipporag_index.test_knowledge_id}")
        logger.info(f"  Index name: {hipporag_index.test_index_name}")

    @pytest.mark.asyncio
    async def test_default_mode_index_naming(self, isolated_index: BaseRAGTest):
        """Test that default mode indices have correct naming"""
        assert isolated_index.rag_mode == "default"
        assert "hipporag" not in isolated_index.test_knowledge_id
        assert "graphrag" not in isolated_index.test_knowledge_id

        logger.info("✅ Default mode configuration validated:")
        logger.info(f"  Mode: {isolated_index.rag_mode}")
        logger.info(f"  Knowledge ID: {isolated_index.test_knowledge_id}")
        logger.info(f"  Index name: {isolated_index.test_index_name}")


# Pytest marks for mode-specific testing
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.integration,
]


# Example of how to skip tests based on mode configuration
@pytest.mark.skipif(
    not TestConfig.should_test_mode("hipporag"), reason="HippoRAG mode not enabled in test configuration"
)
class TestHippoRAGOnly:
    """Test class that only runs when HippoRAG mode is enabled"""

    @pytest.mark.asyncio
    async def test_hipporag_specific_feature(self, hipporag_index: BaseRAGTest):
        """Test a feature specific to HippoRAG mode"""
        logger.info("Testing HippoRAG-specific feature")

        # This test would only run when RAGHUB_TEST_RAG_MODE is set to "hipporag" or "both"
        assert hipporag_index.rag_mode == "hipporag"

        # Add HippoRAG-specific test logic here
        query = "Test HippoRAG specific functionality"
        request = hipporag_index.create_test_retrieval_request(query)
        response = await hipporag_index.client.rag_service_retrieve(request)

        hipporag_index.assert_retrieval_response(response)
        logger.info("✅ HippoRAG-specific test completed")


@pytest.mark.skipif(
    not TestConfig.should_test_mode("graphrag"), reason="GraphRAG mode not enabled in test configuration"
)
class TestGraphRAGOnly:
    """Test class that only runs when GraphRAG mode is enabled"""

    @pytest.mark.asyncio
    async def test_graphrag_specific_feature(self, graphrag_index: BaseRAGTest):
        """Test a feature specific to GraphRAG mode"""
        logger.info("Testing GraphRAG-specific feature")

        # This test would only run when RAGHUB_TEST_RAG_MODE is set to "graphrag" or "both"
        assert graphrag_index.rag_mode == "graphrag"

        # Add GraphRAG-specific test logic here
        query = "Test GraphRAG specific functionality"
        request = graphrag_index.create_test_retrieval_request(query)
        response = await graphrag_index.client.rag_service_retrieve(request)

        graphrag_index.assert_retrieval_response(response)
        logger.info("✅ GraphRAG-specific test completed")
