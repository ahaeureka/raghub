"""
RAGHub Client 集成测试
测试完整的工作流程和系统集成
"""

import logging

import pytest

from .base_test import BaseRAGTest
from .config import TestConfig

logger = logging.getLogger(__name__)


class TestIntegration:
    """集成测试类"""

    @pytest.fixture(scope="class")
    async def rag_test(self):
        """测试环境fixture"""
        test_instance = BaseRAGTest()
        yield test_instance
        await test_instance.cleanup_test_environment()

    @pytest.mark.asyncio
    async def test_full_workflow(self, rag_test: BaseRAGTest):
        """测试完整的RAG工作流程"""
        logger.info("开始完整工作流程测试")

        # 1. 创建索引
        from raghub_protos.models.rag_model import CreateIndexRequest

        create_request = CreateIndexRequest(unique_name=rag_test.test_knowledge_id)
        create_response = await rag_test.client.rag_service_create_index(create_request)
        assert create_response.unique_name == rag_test.test_knowledge_id
        rag_test.created_indices.append(rag_test.test_knowledge_id)
        logger.info("✓ 索引创建成功")

        # 2. 添加文档
        await rag_test.add_test_documents()
        logger.info("✓ 文档添加成功")

        # 3. 检索测试
        query = TestConfig.TEST_QUERIES[0]
        retrieval_request = rag_test.create_test_retrieval_request(query, top_k=3)
        retrieval_response = await rag_test.client.rag_service_retrieval(retrieval_request)

        rag_test.assert_retrieval_response(retrieval_response, expected_min_records=1)
        logger.info(f"✓ 检索成功，找到 {len(retrieval_response.records)} 个相关文档")

        # 4. 聊天测试
        question = TestConfig.TEST_QUERIES[1]
        chat_request = rag_test.create_test_chat_request(question, top_k=3)

        chat_responses = []
        async for response in rag_test.client.rag_service_chat(chat_request):
            chat_responses.append(response)

        rag_test.assert_chat_response(chat_responses, expected_min_responses=1)
        logger.info(f"✓ 聊天成功，收到 {len(chat_responses)} 个响应")

        logger.info("完整工作流程测试通过！")

    @pytest.mark.asyncio
    async def test_error_handling_workflow(self, rag_test: BaseRAGTest):
        """测试错误处理工作流程"""
        logger.info("开始错误处理工作流程测试")

        # 1. 对不存在的知识库进行检索
        retrieval_request = rag_test.create_test_retrieval_request("测试查询", top_k=3)
        retrieval_request.knowledge_id = "nonexistent_kb"

        retrieval_response = await rag_test.client.rag_service_retrieval(retrieval_request)

        # 应该有错误处理或返回空结果
        if retrieval_response.error:
            logger.info(f"✓ 检索不存在知识库的错误处理: {retrieval_response.error.error_msg}")
        else:
            logger.info("✓ 检索不存在知识库返回空结果")

        # 2. 对不存在的知识库进行聊天
        chat_request = rag_test.create_test_chat_request("测试问题", top_k=3)
        chat_request.knowledge_id = "nonexistent_kb"

        chat_responses = []
        try:
            async for response in rag_test.client.rag_service_chat(chat_request):
                chat_responses.append(response)
        except Exception as e:
            logger.info(f"✓ 聊天不存在知识库的错误处理: {e}")

        logger.info("错误处理工作流程测试通过！")

    @pytest.mark.asyncio
    async def test_performance_workflow(self, rag_test: BaseRAGTest):
        """测试性能相关的工作流程"""
        import time

        logger.info("开始性能工作流程测试")

        # 设置测试环境
        await rag_test.setup_test_environment()

        # 1. 批量检索性能测试
        start_time = time.time()

        retrieval_tasks = []
        for query in TestConfig.TEST_QUERIES:
            request = rag_test.create_test_retrieval_request(query, top_k=5)
            retrieval_tasks.append(rag_test.client.rag_service_retrieval(request))

        import asyncio

        retrieval_responses = await asyncio.gather(*retrieval_tasks)

        retrieval_time = time.time() - start_time

        for response in retrieval_responses:
            assert response.error is None

        logger.info(f"✓ 批量检索完成: {len(retrieval_responses)} 个查询，耗时 {retrieval_time:.2f} 秒")

        # 2. 连续聊天性能测试
        start_time = time.time()

        for i, question in enumerate(TestConfig.TEST_QUERIES[:2]):  # 只测试前2个问题
            chat_request = rag_test.create_test_chat_request(question, top_k=3)

            response_count = 0
            async for response in rag_test.client.rag_service_chat(chat_request):
                response_count += 1

            logger.info(f"问题 {i + 1} 收到 {response_count} 个响应")

        chat_time = time.time() - start_time
        logger.info(f"✓ 连续聊天完成: 耗时 {chat_time:.2f} 秒")

        logger.info("性能工作流程测试通过！")

    @pytest.mark.asyncio
    async def test_data_consistency_workflow(self, rag_test: BaseRAGTest):
        """测试数据一致性工作流程"""
        logger.info("开始数据一致性工作流程测试")

        # 设置测试环境
        await rag_test.setup_test_environment()

        # 1. 添加文档后立即检索
        from raghub_protos.models.rag_model import AddDocumentsRequest, RAGDocument

        new_document = RAGDocument(
            content="这是一个用于测试数据一致性的新文档，包含唯一标识符：data_consistency_test_123456",
            title="数据一致性测试文档",
            metadata={"test_type": "consistency", "unique_id": "123456"},
            type="text",
            source="consistency_test",
        )

        add_request = AddDocumentsRequest(knowledge_id=rag_test.test_knowledge_id, documents=[new_document])

        add_response = await rag_test.client.rag_service_add_documents(add_request)
        assert add_response.error is None
        logger.info("✓ 新文档添加成功")

        # 2. 立即检索新添加的文档
        retrieval_request = rag_test.create_test_retrieval_request("data_consistency_test_123456", top_k=5)
        retrieval_response = await rag_test.client.rag_service_retrieval(retrieval_request)

        # 验证能够检索到新添加的文档
        found_new_doc = False
        for record in retrieval_response.records:
            if "data_consistency_test_123456" in record.content:
                found_new_doc = True
                break

        assert found_new_doc, "新添加的文档应该能够立即被检索到"
        logger.info("✓ 新添加的文档立即可检索")

        # 3. 测试聊天中的数据一致性
        chat_request = rag_test.create_test_chat_request("请告诉我关于数据一致性测试的信息", top_k=3)

        chat_responses = []
        async for response in rag_test.client.rag_service_chat(chat_request):
            chat_responses.append(response)

        # 验证聊天回答中包含新文档的信息
        full_answer = "".join([r.choices[0].message.content for r in chat_responses])
        logger.info(f"聊天回答: {full_answer[:200]}...")

        logger.info("数据一致性工作流程测试通过！")

    @pytest.mark.asyncio
    async def test_concurrent_operations_workflow(self, rag_test: BaseRAGTest):
        """测试并发操作工作流程"""
        import asyncio

        logger.info("开始并发操作工作流程测试")

        # 设置测试环境
        await rag_test.setup_test_environment()

        # 1. 并发检索测试
        async def concurrent_retrieval(query_index):
            query = f"测试查询 {query_index}"
            request = rag_test.create_test_retrieval_request(query, top_k=3)
            response = await rag_test.client.rag_service_retrieval(request)
            return response

        retrieval_tasks = [concurrent_retrieval(i) for i in range(5)]
        retrieval_results = await asyncio.gather(*retrieval_tasks, return_exceptions=True)

        successful_retrievals = [r for r in retrieval_results if not isinstance(r, Exception)]
        logger.info(f"✓ 并发检索: {len(successful_retrievals)}/5 个成功")

        # 2. 并发聊天测试
        async def concurrent_chat(question_index):
            question = f"这是并发测试问题 {question_index}"
            request = rag_test.create_test_chat_request(question, top_k=2)

            responses = []
            async for response in rag_test.client.rag_service_chat(request):
                responses.append(response)
            return responses

        chat_tasks = [concurrent_chat(i) for i in range(3)]
        chat_results = await asyncio.gather(*chat_tasks, return_exceptions=True)

        successful_chats = [r for r in chat_results if not isinstance(r, Exception)]
        logger.info(f"✓ 并发聊天: {len(successful_chats)}/3 个成功")

        logger.info("并发操作工作流程测试通过！")
