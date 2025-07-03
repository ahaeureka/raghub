"""
RAGHub Client Base Test Class
Provides common functionality and utilities for testing
"""

import logging
import os

# Use absolute imports to avoid module path conflicts
import sys
import uuid
from typing import List, Optional

from raghub_client.rag_hub_client import RAGHubClient
from raghub_protos.models.chat_model import (
    ChatMessage,
    CreateChatCompletionRequest,
    CreateChatCompletionResponse,
    RetrievalSetting,
)
from raghub_protos.models.rag_model import (
    AddDocumentsRequest,
    CreateIndexRequest,
    DeleteDocumentsRequest,
    RAGDocument,
    RetrievalRequest,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from config import TestConfig

logger = logging.getLogger(__name__)


class BaseRAGTest:
    """RAG Test Base Class"""

    def __init__(self, use_shared_index: bool = False, rag_mode: Optional[str] = None):
        """
        Initialize RAG test instance

        Args:
            use_shared_index: Whether to use shared index across tests
            rag_mode: RAG mode ('hipporag', 'graphrag', or None for default)
        """
        self.client = RAGHubClient(**TestConfig.get_client_config())
        self.rag_mode = rag_mode or TestConfig.RAG_MODE or "default"

        # Create mode-specific index names
        print(f"Initializing BaseRAGTest with rag_mode={self.rag_mode}, use_shared_index={use_shared_index}")
        mode_suffix = f"_{self.rag_mode}" if self.rag_mode != "default" else ""

        if use_shared_index:
            # Use fixed shared index names with mode suffix
            self.test_knowledge_id = f"{TestConfig.TEST_KNOWLEDGE_ID}_shared_{mode_suffix}"
            self.test_index_name = f"{TestConfig.TEST_INDEX_NAME}_shared_{mode_suffix}"
        else:
            # Use unique index names with mode suffix
            unique_id = uuid.uuid4().hex[:8]
            self.test_knowledge_id = f"{TestConfig.TEST_KNOWLEDGE_ID}_{unique_id}{mode_suffix}"
            self.test_index_name = f"{TestConfig.TEST_INDEX_NAME}_{unique_id}{mode_suffix}"

        self.created_indices: List[str] = []
        self.added_document_ids: List[str] = []
        self._is_shared = use_shared_index

    async def setup_test_environment(self):
        """Setup test environment"""
        logger.info(f"Setting up test environment: knowledge_id={self.test_knowledge_id}, mode={self.rag_mode}")
        try:
            # Create test index
            await self.create_test_index()
            # Add test documents
            await self.add_test_documents()
        except Exception as e:
            logger.error(f"Failed to setup test environment: {e}")
            raise

    async def cleanup_test_environment(self):
        """Cleanup test environment"""
        if self._is_shared:
            # In shared index mode, only cleanup at the end
            logger.info(f"Shared index mode, skipping cleanup: knowledge_id={self.test_knowledge_id}")
            return

        logger.info(f"Cleaning up test environment: knowledge_id={self.test_knowledge_id}, mode={self.rag_mode}")
        try:
            # Delete added documents
            if self.added_document_ids:
                await self.delete_test_documents()
        except Exception as e:
            logger.warning(f"Failed to cleanup test environment: {e}")

    async def force_cleanup(self):
        """Force cleanup test environment (including shared index)"""
        logger.info(f"Force cleaning up test environment: knowledge_id={self.test_knowledge_id}, mode={self.rag_mode}")
        try:
            if self.added_document_ids:
                await self.delete_test_documents()
        except Exception as e:
            logger.warning(f"Failed to force cleanup test environment: {e}")

    async def create_test_index(self):
        """Create test index"""
        request = CreateIndexRequest(unique_name=self.test_knowledge_id)
        response = await self.client.rag_service_create_index(request)
        assert response.unique_name == self.test_knowledge_id
        self.created_indices.append(self.test_knowledge_id)
        logger.info(f"Test index created successfully: {self.test_knowledge_id} (mode: {self.rag_mode})")

    async def add_test_documents(self):
        """Add test documents"""
        documents = []
        for doc_data in TestConfig.TEST_DOCUMENTS:
            document = RAGDocument(
                content=doc_data["content"],
                title=doc_data["title"],
                metadata=doc_data["metadata"],
                type=doc_data["type"],
                source=doc_data["source"],
            )
            documents.append(document)

        request = AddDocumentsRequest(knowledge_id=self.test_knowledge_id, documents=documents)
        response = await self.client.rag_service_add_documents(request)

        assert response.documents is not None
        assert len(response.documents) == len(TestConfig.TEST_DOCUMENTS)
        logger.info(f"Test documents added successfully: {len(response.documents)} documents (mode: {self.rag_mode})")

        return response

    async def delete_test_documents(self):
        """Delete test documents"""
        if not self.added_document_ids:
            logger.info("No documents to delete")
            return

        request = DeleteDocumentsRequest(knowledge_id=self.test_knowledge_id, document_ids=self.added_document_ids)
        response = await self.client.rag_service_delete_documents(request)
        logger.info(
            f"Test documents deleted successfully: {len(response.deleted_ids)} documents (mode: {self.rag_mode})"
        )

        return response

    def create_test_retrieval_request(self, query: str, top_k: int = 5) -> RetrievalRequest:
        """Create test retrieval request"""
        retrieval_setting = RetrievalSetting(top_k=top_k)
        return RetrievalRequest(knowledge_id=self.test_knowledge_id, query=query, retrieval_setting=retrieval_setting)

    def create_test_chat_request(self, question: str, top_k: int = 5) -> CreateChatCompletionRequest:
        """Create test chat request"""
        messages = [ChatMessage(role="user", content=question)]
        retrieval_setting = RetrievalSetting(top_k=top_k)
        print(f"Creating chat request for question: {question}:{self.test_knowledge_id} with top_k={top_k}")
        return CreateChatCompletionRequest(
            knowledge_id=self.test_knowledge_id, messages=messages, retrieval_setting=retrieval_setting
        )

    def assert_retrieval_response(self, response, expected_min_records: int = 1):
        """Validate retrieval response"""
        assert response is not None
        assert response.error is None, f"Retrieval error: {response.error}"
        assert response.records is not None
        assert len(response.records) >= expected_min_records

        for record in response.records:
            assert record.content is not None and len(record.content) > 0
            assert record.score >= 0.0 and record.score <= 1.0
            assert record.title is not None

    def assert_chat_response(self, responses: List[CreateChatCompletionResponse], expected_min_responses: int = 1):
        """Validate chat response"""
        assert len(responses) >= expected_min_responses

        for response in responses:
            assert response is not None
            assert response.choices is not None and len(response.choices) > 0

            choice = response.choices[0]
            assert choice.message is not None
            assert choice.message.role == "assistant"
            if choice.index != "0":
                assert choice.message.content is not None and len(choice.message.content) > 0
