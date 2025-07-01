"""
RAGHub Client 文档管理测试
测试文档的添加和删除功能
"""

import logging

import pytest
from raghub_protos.models.rag_model import RAGDocument

# 使用绝对导入避免模块路径冲突
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from base_test import BaseRAGTest
from config import TestConfig

logger = logging.getLogger(__name__)


class TestDocumentService:
    """文档管理服务测试类"""

    @pytest.fixture(scope="class")
    async def rag_test(self):
        """测试环境fixture（不预先添加文档）"""
        test_instance = BaseRAGTest()
        # 只创建索引，不添加文档
        await test_instance.create_test_index()
        yield test_instance
        await test_instance.cleanup_test_environment()

    @pytest.mark.asyncio
    async def test_add_documents_basic(self, rag_test: BaseRAGTest):
        """测试基本文档添加功能"""
        # 准备测试文档
        documents = []
        for doc_data in TestConfig.TEST_DOCUMENTS[:2]:  # 只使用前2个文档
            document = RAGDocument(
                content=doc_data["content"],
                title=doc_data["title"],
                metadata=doc_data["metadata"],
                type=doc_data["type"],
                source=doc_data["source"],
            )
            documents.append(document)

        # 添加文档
        from raghub_protos.models.rag_model import AddDocumentsRequest

        request = AddDocumentsRequest(knowledge_id=rag_test.test_knowledge_id, documents=documents)

        response = await rag_test.client.rag_service_add_documents(request)

        # 验证响应
        assert response.error is None, f"添加文档出错: {response.error}"
        assert response.documents is not None
        assert len(response.documents) == len(documents)

        logger.info(f"成功添加 {len(response.documents)} 个文档")

        # 保存文档ID用于后续清理
        for i, doc in enumerate(response.documents):
            rag_test.added_document_ids.append(f"doc_{i}")

    @pytest.mark.asyncio
    async def test_add_documents_single(self, rag_test: BaseRAGTest):
        """测试添加单个文档"""
        document = RAGDocument(
            content="单元测试文档内容：这是一个用于测试的文档。",
            title="测试文档",
            metadata={"category": "test", "author": "test_user"},
            type="text",
            source="test",
        )

        from raghub_protos.models.rag_model import AddDocumentsRequest

        request = AddDocumentsRequest(knowledge_id=rag_test.test_knowledge_id, documents=[document])

        response = await rag_test.client.rag_service_add_documents(request)

        assert response.error is None
        assert response.documents is not None
        assert len(response.documents) == 1

        added_doc = response.documents[0]
        assert added_doc.content == document.content
        assert added_doc.title == document.title

        logger.info("成功添加单个测试文档")

    @pytest.mark.asyncio
    async def test_add_documents_with_metadata(self, rag_test: BaseRAGTest):
        """测试添加带复杂元数据的文档"""
        document = RAGDocument(
            content="包含复杂元数据的测试文档。",
            title="复杂元数据测试",
            metadata={
                "category": "advanced_test",
                "tags": ["test", "metadata", "complex"],
                "version": "1.0",
                "priority": "high",
                "created_by": "automated_test",
            },
            type="text",
            source="test_suite",
        )

        from raghub_protos.models.rag_model import AddDocumentsRequest

        request = AddDocumentsRequest(knowledge_id=rag_test.test_knowledge_id, documents=[document])

        response = await rag_test.client.rag_service_add_documents(request)

        assert response.error is None
        assert response.documents is not None
        assert len(response.documents) == 1

        added_doc = response.documents[0]
        assert added_doc.metadata is not None

        logger.info("成功添加带复杂元数据的文档")

    @pytest.mark.asyncio
    async def test_add_empty_documents(self, rag_test: BaseRAGTest):
        """测试添加空文档列表"""
        from raghub_protos.models.rag_model import AddDocumentsRequest

        request = AddDocumentsRequest(knowledge_id=rag_test.test_knowledge_id, documents=[])

        response = await rag_test.client.rag_service_add_documents(request)

        # 空文档列表应该正常处理
        assert response.documents is not None
        assert len(response.documents) == 0

        logger.info("空文档列表处理正常")

    @pytest.mark.asyncio
    async def test_delete_documents(self, rag_test: BaseRAGTest):
        """测试文档删除功能"""
        # 首先添加一些文档
        documents = []
        test_doc_ids = []

        for i, doc_data in enumerate(TestConfig.TEST_DOCUMENTS[:2]):
            document = RAGDocument(
                content=doc_data["content"],
                title=doc_data["title"],
                metadata=doc_data["metadata"],
                type=doc_data["type"],
                source=doc_data["source"],
            )
            documents.append(document)
            test_doc_ids.append(f"delete_test_doc_{i}")

        # 添加文档
        from raghub_protos.models.rag_model import AddDocumentsRequest

        add_request = AddDocumentsRequest(knowledge_id=rag_test.test_knowledge_id, documents=documents)

        add_response = await rag_test.client.rag_service_add_documents(add_request)
        assert add_response.error is None

        # 删除文档
        from raghub_protos.models.rag_model import DeleteDocumentsRequest

        delete_request = DeleteDocumentsRequest(knowledge_id=rag_test.test_knowledge_id, document_ids=test_doc_ids)

        delete_response = await rag_test.client.rag_service_delete_documents(delete_request)

        assert delete_response.error is None
        assert delete_response.deleted_ids is not None

        logger.info(f"成功删除 {len(delete_response.deleted_ids)} 个文档")

    @pytest.mark.asyncio
    async def test_delete_nonexistent_documents(self, rag_test: BaseRAGTest):
        """测试删除不存在的文档"""
        nonexistent_ids = ["nonexistent_1", "nonexistent_2"]

        from raghub_protos.models.rag_model import DeleteDocumentsRequest

        request = DeleteDocumentsRequest(knowledge_id=rag_test.test_knowledge_id, document_ids=nonexistent_ids)

        response = await rag_test.client.rag_service_delete_documents(request)

        # 删除不存在的文档应该正常处理（可能返回空列表）
        assert response.deleted_ids is not None

        logger.info(f"删除不存在文档的响应: {len(response.deleted_ids)} 个已删除")

    @pytest.mark.asyncio
    async def test_add_documents_to_nonexistent_knowledge_id(self, rag_test: BaseRAGTest):
        """测试向不存在的知识库添加文档"""
        document = RAGDocument(
            content="测试文档内容", title="测试文档", metadata={"test": "true"}, type="text", source="test"
        )

        from raghub_protos.models.rag_model import AddDocumentsRequest

        request = AddDocumentsRequest(knowledge_id="nonexistent_knowledge_id", documents=[document])

        response = await rag_test.client.rag_service_add_documents(request)

        # 应该返回错误或者自动创建知识库
        if response.error:
            logger.info(f"向不存在知识库添加文档的预期错误: {response.error.error_msg}")
        else:
            logger.info("系统自动处理了不存在的知识库")

    @pytest.mark.asyncio
    async def test_document_content_validation(self, rag_test: BaseRAGTest):
        """测试文档内容验证"""
        # 测试各种内容长度的文档
        test_cases = [
            ("短文档", "短"),
            ("中等长度文档", "这是一个中等长度的测试文档，包含了一些基本的信息用于验证系统的处理能力。"),
            ("长文档", "这是一个很长的测试文档。" * 100),  # 重复100次
        ]

        for title, content in test_cases:
            document = RAGDocument(
                content=content, title=title, metadata={"length": str(len(content))}, type="text", source="test"
            )

            from raghub_protos.models.rag_model import AddDocumentsRequest

            request = AddDocumentsRequest(knowledge_id=rag_test.test_knowledge_id, documents=[document])

            response = await rag_test.client.rag_service_add_documents(request)

            assert response.error is None, f"添加{title}失败: {response.error}"
            assert len(response.documents) == 1

            logger.info(f"成功添加{title}，长度: {len(content)} 字符")
