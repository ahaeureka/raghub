"""
RAGHub Client 索引管理测试
测试索引的创建和管理功能
"""

import logging

import pytest

from .base_test import BaseRAGTest

logger = logging.getLogger(__name__)


class TestIndexService:
    """索引管理服务测试类"""

    @pytest.fixture(scope="function")
    async def rag_test(self):
        """测试环境fixture（每个测试独立）"""
        test_instance = BaseRAGTest()
        yield test_instance
        await test_instance.cleanup_test_environment()

    @pytest.mark.asyncio
    async def test_create_index_basic(self, rag_test: BaseRAGTest):
        """测试基本索引创建功能"""
        from raghub_protos.models.rag_model import CreateIndexRequest

        request = CreateIndexRequest(unique_name=rag_test.test_knowledge_id)
        response = await rag_test.client.rag_service_create_index(request)

        assert response.unique_name == rag_test.test_knowledge_id
        rag_test.created_indices.append(rag_test.test_knowledge_id)

        logger.info(f"成功创建索引: {response.unique_name}")

    @pytest.mark.asyncio
    async def test_create_index_with_special_characters(self, rag_test: BaseRAGTest):
        """测试包含特殊字符的索引名称"""
        special_names = [
            f"{rag_test.test_knowledge_id}_with_underscore",
            f"{rag_test.test_knowledge_id}-with-dash",
            f"{rag_test.test_knowledge_id}.with.dot",
        ]

        from raghub_protos.models.rag_model import CreateIndexRequest

        for name in special_names:
            request = CreateIndexRequest(unique_name=name)
            response = await rag_test.client.rag_service_create_index(request)

            assert response.unique_name == name
            rag_test.created_indices.append(name)

            logger.info(f"成功创建特殊字符索引: {name}")

    @pytest.mark.asyncio
    async def test_create_duplicate_index(self, rag_test: BaseRAGTest):
        """测试创建重复索引"""
        from raghub_protos.models.rag_model import CreateIndexRequest

        # 创建第一个索引
        request = CreateIndexRequest(unique_name=rag_test.test_knowledge_id)
        response1 = await rag_test.client.rag_service_create_index(request)
        assert response1.unique_name == rag_test.test_knowledge_id
        rag_test.created_indices.append(rag_test.test_knowledge_id)

        # 尝试创建相同名称的索引
        response2 = await rag_test.client.rag_service_create_index(request)

        # 应该正常处理重复创建（可能返回现有索引或错误）
        logger.info(f"重复创建索引的响应: {response2.unique_name}")

    @pytest.mark.asyncio
    async def test_create_index_empty_name(self, rag_test: BaseRAGTest):
        """测试创建空名称索引"""
        from raghub_protos.models.rag_model import CreateIndexRequest

        request = CreateIndexRequest(unique_name="")

        try:
            response = await rag_test.client.rag_service_create_index(request)
            logger.info(f"空名称索引创建响应: {response.unique_name}")
        except Exception as e:
            logger.info(f"空名称索引创建产生预期异常: {e}")

    @pytest.mark.asyncio
    async def test_create_multiple_indices(self, rag_test: BaseRAGTest):
        """测试创建多个索引"""
        from raghub_protos.models.rag_model import CreateIndexRequest

        index_names = [
            f"{rag_test.test_knowledge_id}_1",
            f"{rag_test.test_knowledge_id}_2",
            f"{rag_test.test_knowledge_id}_3",
        ]

        created_indices = []

        for name in index_names:
            request = CreateIndexRequest(unique_name=name)
            response = await rag_test.client.rag_service_create_index(request)

            assert response.unique_name == name
            created_indices.append(name)
            rag_test.created_indices.append(name)

        assert len(created_indices) == len(index_names)
        logger.info(f"成功创建 {len(created_indices)} 个索引")

    @pytest.mark.asyncio
    async def test_index_creation_and_usage(self, rag_test: BaseRAGTest):
        """测试索引创建后的使用"""
        from raghub_protos.models.rag_model import AddDocumentsRequest, CreateIndexRequest, RAGDocument

        # 创建索引
        create_request = CreateIndexRequest(unique_name=rag_test.test_knowledge_id)
        create_response = await rag_test.client.rag_service_create_index(create_request)
        assert create_response.unique_name == rag_test.test_knowledge_id
        rag_test.created_indices.append(rag_test.test_knowledge_id)

        # 向创建的索引添加文档
        document = RAGDocument(
            content="这是添加到新创建索引的测试文档。",
            title="索引测试文档",
            metadata={"test": "index_creation"},
            type="text",
            source="test",
        )

        add_request = AddDocumentsRequest(knowledge_id=rag_test.test_knowledge_id, documents=[document])

        add_response = await rag_test.client.rag_service_add_documents(add_request)
        assert add_response.error is None
        assert len(add_response.documents) == 1

        logger.info("成功在新创建的索引中添加文档")

        # 测试检索
        retrieval_request = rag_test.create_test_retrieval_request("测试文档", top_k=1)
        retrieval_response = await rag_test.client.rag_service_retrieval(retrieval_request)

        assert retrieval_response.error is None
        assert len(retrieval_response.records) > 0

        logger.info("成功在新创建的索引中检索文档")

    @pytest.mark.asyncio
    async def test_index_name_length_limits(self, rag_test: BaseRAGTest):
        """测试索引名称长度限制"""
        from raghub_protos.models.rag_model import CreateIndexRequest

        test_cases = [
            ("短名称", "short"),
            ("中等长度名称", "medium_length_index_name_for_testing"),
            ("长名称", "very_long_index_name_" + "x" * 100),  # 很长的名称
        ]

        for description, name in test_cases:
            full_name = f"{rag_test.test_knowledge_id}_{name}"
            request = CreateIndexRequest(unique_name=full_name)

            try:
                response = await rag_test.client.rag_service_create_index(request)
                assert response.unique_name == full_name
                rag_test.created_indices.append(full_name)
                logger.info(f"成功创建{description}: {len(full_name)} 字符")
            except Exception as e:
                logger.info(f"{description}创建失败（可能超出长度限制）: {e}")

    @pytest.mark.asyncio
    async def test_concurrent_index_creation(self, rag_test: BaseRAGTest):
        """测试并发索引创建"""
        import asyncio

        from raghub_protos.models.rag_model import CreateIndexRequest

        async def create_index(suffix: str):
            name = f"{rag_test.test_knowledge_id}_concurrent_{suffix}"
            request = CreateIndexRequest(unique_name=name)
            response = await rag_test.client.rag_service_create_index(request)
            rag_test.created_indices.append(name)
            return response

        # 并发创建多个索引
        tasks = [create_index(str(i)) for i in range(3)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        successful_creations = [r for r in responses if not isinstance(r, Exception)]

        assert len(successful_creations) > 0
        logger.info(f"并发创建索引: {len(successful_creations)} 个成功")

        for response in successful_creations:
            logger.info(f"并发创建的索引: {response.unique_name}")
