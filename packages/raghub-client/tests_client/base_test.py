"""
RAGHub Client 基础测试类
提供测试的通用功能和工具方法
"""

import logging
import os

# 使用绝对导入避免模块路径冲突
import sys
import uuid
from typing import List

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
    """RAG 测试基类"""

    def __init__(self, use_shared_index: bool = False):
        self.client = RAGHubClient(**TestConfig.get_client_config())

        if use_shared_index:
            # 使用固定的共享索引名称
            self.test_knowledge_id = f"{TestConfig.TEST_KNOWLEDGE_ID}_shared"
            self.test_index_name = f"{TestConfig.TEST_INDEX_NAME}_shared"
        else:
            # 使用唯一的索引名称
            self.test_knowledge_id = f"{TestConfig.TEST_KNOWLEDGE_ID}_{uuid.uuid4().hex[:8]}"
            self.test_index_name = f"{TestConfig.TEST_INDEX_NAME}_{uuid.uuid4().hex[:8]}"

        self.created_indices: List[str] = []
        self.added_document_ids: List[str] = []
        self._is_shared = use_shared_index

    async def setup_test_environment(self):
        """设置测试环境"""
        logger.info(f"设置测试环境: knowledge_id={self.test_knowledge_id}")
        try:
            # 创建测试索引
            await self.create_test_index()
            # 添加测试文档
            await self.add_test_documents()
        except Exception as e:
            logger.error(f"设置测试环境失败: {e}")
            raise

    async def cleanup_test_environment(self):
        """清理测试环境"""
        if self._is_shared:
            # 共享索引模式下，只在最后清理
            logger.info(f"共享索引模式，跳过清理: knowledge_id={self.test_knowledge_id}")
            return

        logger.info(f"清理测试环境: knowledge_id={self.test_knowledge_id}")
        try:
            # 删除添加的文档
            if self.added_document_ids:
                await self.delete_test_documents()
        except Exception as e:
            logger.warning(f"清理测试环境失败: {e}")

    async def force_cleanup(self):
        """强制清理测试环境（包括共享索引）"""
        logger.info(f"强制清理测试环境: knowledge_id={self.test_knowledge_id}")
        try:
            if self.added_document_ids:
                await self.delete_test_documents()
        except Exception as e:
            logger.warning(f"强制清理测试环境失败: {e}")

    async def create_test_index(self):
        """创建测试索引"""
        request = CreateIndexRequest(unique_name=self.test_knowledge_id)
        response = await self.client.rag_service_create_index(request)
        assert response.unique_name == self.test_knowledge_id
        self.created_indices.append(self.test_knowledge_id)
        logger.info(f"创建测试索引成功: {self.test_knowledge_id}")

    async def add_test_documents(self):
        """添加测试文档"""
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
        logger.info(f"添加测试文档成功: {len(response.documents)} 个文档")

        return response

    async def delete_test_documents(self):
        """删除测试文档"""
        if not self.added_document_ids:
            logger.info("没有需要删除的文档")
            return

        request = DeleteDocumentsRequest(knowledge_id=self.test_knowledge_id, document_ids=self.added_document_ids)
        response = await self.client.rag_service_delete_documents(request)
        logger.info(f"删除测试文档成功: {len(response.deleted_ids)} 个文档")

        return response

    def create_test_retrieval_request(self, query: str, top_k: int = 5) -> RetrievalRequest:
        """创建测试检索请求"""
        retrieval_setting = RetrievalSetting(top_k=top_k)
        return RetrievalRequest(knowledge_id=self.test_knowledge_id, query=query, retrieval_setting=retrieval_setting)

    def create_test_chat_request(self, question: str, top_k: int = 5) -> CreateChatCompletionRequest:
        """创建测试聊天请求"""
        messages = [ChatMessage(role="user", content=question)]
        retrieval_setting = RetrievalSetting(top_k=top_k)

        return CreateChatCompletionRequest(
            knowledge_id=self.test_knowledge_id, messages=messages, retrieval_setting=retrieval_setting
        )

    def assert_retrieval_response(self, response, expected_min_records: int = 1):
        """验证检索响应"""
        assert response is not None
        assert response.error is None, f"检索出错: {response.error}"
        assert response.records is not None
        assert len(response.records) >= expected_min_records

        for record in response.records:
            assert record.content is not None and len(record.content) > 0
            assert record.score >= 0.0 and record.score <= 1.0
            assert record.title is not None

    def assert_chat_response(self, responses: List[CreateChatCompletionResponse], expected_min_responses: int = 1):
        """验证聊天响应"""
        assert len(responses) >= expected_min_responses

        for response in responses:
            assert response is not None
            assert response.choices is not None and len(response.choices) > 0

            choice = response.choices[0]
            assert choice.message is not None
            assert choice.message.role == "assistant"
            if choice.index != "0":
                assert choice.message.content is not None and len(choice.message.content) > 0
