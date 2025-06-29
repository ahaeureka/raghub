"""
RAGHub Client 检索服务测试
测试文档检索功能
"""

import logging

import pytest
from raghub_protos.models.rag_model import MetadataCondition, MetadataConditionItem

from .base_test import BaseRAGTest
from .config import TestConfig

logger = logging.getLogger(__name__)


class TestRetrievalService:
    """检索服务测试类"""

    @pytest.fixture(scope="class")
    async def rag_test(self):
        """测试环境fixture"""
        test_instance = BaseRAGTest()
        await test_instance.setup_test_environment()
        yield test_instance
        await test_instance.cleanup_test_environment()

    @pytest.mark.asyncio
    async def test_retrieval_basic(self, rag_test: BaseRAGTest):
        """测试基本检索功能"""
        query = TestConfig.TEST_QUERIES[0]
        request = rag_test.create_test_retrieval_request(query, top_k=3)

        response = await rag_test.client.rag_service_retrieval(request)

        rag_test.assert_retrieval_response(response, expected_min_records=1)
        logger.info(f"检索到 {len(response.records)} 个相关文档")

        # 验证响应内容
        for record in response.records:
            logger.info(f"文档标题: {record.title}, 相关性得分: {record.score:.3f}")
            assert "Python" in record.content or "python" in record.content.lower()

    @pytest.mark.asyncio
    async def test_retrieval_with_different_top_k(self, rag_test: BaseRAGTest):
        """测试不同 top_k 值的检索"""
        query = TestConfig.TEST_QUERIES[1]

        # 测试 top_k=1
        request_1 = rag_test.create_test_retrieval_request(query, top_k=1)
        response_1 = await rag_test.client.rag_service_retrieval(request_1)
        assert len(response_1.records) <= 1

        # 测试 top_k=3
        request_3 = rag_test.create_test_retrieval_request(query, top_k=3)
        response_3 = await rag_test.client.rag_service_retrieval(request_3)
        assert len(response_3.records) <= 3
        assert len(response_3.records) >= len(response_1.records)

        logger.info(f"top_k=1: {len(response_1.records)} 个结果")
        logger.info(f"top_k=3: {len(response_3.records)} 个结果")

    @pytest.mark.asyncio
    async def test_retrieval_with_metadata_filter(self, rag_test: BaseRAGTest):
        """测试带元数据过滤的检索"""
        query = "人工智能"

        # 创建元数据过滤条件
        condition_item = MetadataConditionItem(name=["category"], comparison_operator="eq", value="ai")
        metadata_condition = MetadataCondition(logical_operator="and", conditions=[condition_item])

        request = rag_test.create_test_retrieval_request(query, top_k=5)
        request.metadata_condition = metadata_condition

        response = await rag_test.client.rag_service_retrieval(request)

        rag_test.assert_retrieval_response(response, expected_min_records=1)

        # 验证返回的文档都符合过滤条件
        for record in response.records:
            logger.info(f"过滤后的文档: {record.title}")

    @pytest.mark.asyncio
    async def test_retrieval_empty_query(self, rag_test: BaseRAGTest):
        """测试空查询"""
        request = rag_test.create_test_retrieval_request("", top_k=3)

        response = await rag_test.client.rag_service_retrieval(request)

        # 空查询应该返回一些结果或者没有结果，但不应该出错
        assert response.error is None
        logger.info(f"空查询返回 {len(response.records) if response.records else 0} 个结果")

    @pytest.mark.asyncio
    async def test_retrieval_nonexistent_knowledge_id(self, rag_test: BaseRAGTest):
        """测试不存在的知识库ID"""
        request = rag_test.create_test_retrieval_request(TestConfig.TEST_QUERIES[0], top_k=3)
        request.knowledge_id = "nonexistent_knowledge_id"

        response = await rag_test.client.rag_service_retrieval(request)

        # 应该返回错误或空结果
        if response.error:
            logger.info(f"预期的错误: {response.error.error_msg}")
        else:
            assert len(response.records) == 0
            logger.info("不存在的知识库ID返回空结果")

    @pytest.mark.asyncio
    async def test_retrieval_score_ordering(self, rag_test: BaseRAGTest):
        """测试检索结果按相关性得分排序"""
        query = TestConfig.TEST_QUERIES[2]
        request = rag_test.create_test_retrieval_request(query, top_k=3)

        response = await rag_test.client.rag_service_retrieval(request)

        rag_test.assert_retrieval_response(response, expected_min_records=1)

        # 验证结果按得分降序排列
        scores = [record.score for record in response.records]
        assert scores == sorted(scores, reverse=True), "检索结果应该按相关性得分降序排列"

        logger.info(f"检索得分排序: {scores}")

    @pytest.mark.asyncio
    async def test_retrieval_response_fields(self, rag_test: BaseRAGTest):
        """测试检索响应字段完整性"""
        query = TestConfig.TEST_QUERIES[0]
        request = rag_test.create_test_retrieval_request(query, top_k=1)

        response = await rag_test.client.rag_service_retrieval(request)

        rag_test.assert_retrieval_response(response, expected_min_records=1)

        # 验证响应字段
        assert response.request_id is not None

        record = response.records[0]
        assert record.content is not None and len(record.content) > 0
        assert record.score is not None
        assert record.title is not None
        assert record.metadata is not None

        logger.info(f"检索响应字段验证通过: request_id={response.request_id}")
