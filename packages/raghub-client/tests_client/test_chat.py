"""
RAGHub Client 聊天服务测试
测试基于知识库的问答功能
"""

import logging
import os

# 使用绝对导入避免模块路径冲突
import sys
from typing import List

import pytest
from raghub_protos.models.chat_model import CreateChatCompletionResponse

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from base_test import BaseRAGTest
from config import TestConfig

logger = logging.getLogger(__name__)


class TestChatService:
    """聊天服务测试类"""

    @pytest.mark.asyncio
    async def test_chat_basic(self, shared_index: BaseRAGTest):
        """测试基本聊天功能"""
        question = TestConfig.TEST_QUERIES[0]
        request = shared_index.create_test_chat_request(question, top_k=3)

        responses: List[CreateChatCompletionResponse] = []
        async for response in shared_index.client.rag_service_chat(request):
            responses.append(response)
            logger.info(f"收到聊天响应: {response.choices[0].message.content[:100]}...")

        shared_index.assert_chat_response(responses, expected_min_responses=1)

        # 验证最终回答包含相关内容
        final_content = "".join([r.choices[0].message.content for r in responses])
        assert len(final_content) > 0
        logger.info(f"聊天回答总长度: {len(final_content)} 字符")

    @pytest.mark.asyncio
    async def test_chat_streaming(self, shared_index: BaseRAGTest):
        """测试流式聊天响应"""
        question = TestConfig.TEST_QUERIES[1]
        request = shared_index.create_test_chat_request(question, top_k=2)

        response_count = 0
        total_tokens = 0

        async for response in shared_index.client.rag_service_chat(request):
            response_count += 1
            assert response.choices is not None and len(response.choices) > 0

            choice = response.choices[0]
            assert choice.message.role == "assistant"
            assert choice.message.content is not None

            if response.usage:
                total_tokens += response.usage.total_tokens

            logger.info(f"流式响应 {response_count}: {choice.message.content[:50]}...")

        assert response_count > 0
        logger.info(f"收到 {response_count} 个流式响应，总token数: {total_tokens}")

    @pytest.mark.asyncio
    async def test_chat_with_context(self, shared_index: BaseRAGTest):
        """测试基于上下文的聊天"""
        question = "请详细解释深度学习的原理"
        request = shared_index.create_test_chat_request(question, top_k=5)

        responses: List[CreateChatCompletionResponse] = []
        async for response in shared_index.client.rag_service_chat(request):
            responses.append(response)

        shared_index.assert_chat_response(responses, expected_min_responses=1)

        # 验证回答内容相关性
        full_answer = "".join([r.choices[0].message.content for r in responses])
        assert "深度学习" in full_answer or "神经网络" in full_answer
        logger.info("上下文聊天回答包含相关概念")

    @pytest.mark.asyncio
    async def test_chat_empty_question(self, shared_index: BaseRAGTest):
        """测试空问题"""
        request = shared_index.create_test_chat_request("", top_k=3)

        responses: List[CreateChatCompletionResponse] = []
        try:
            async for response in shared_index.client.rag_service_chat(request):
                responses.append(response)
        except Exception as e:
            logger.info(f"空问题产生预期异常: {e}")
            return

        # 如果没有异常，应该返回合理的回答
        if responses:
            shared_index.assert_chat_response(responses, expected_min_responses=1)
            logger.info("空问题返回了默认回答")

    @pytest.mark.asyncio
    async def test_chat_nonexistent_knowledge_id(self, shared_index: BaseRAGTest):
        """测试不存在的知识库ID"""
        question = TestConfig.TEST_QUERIES[0]
        request = shared_index.create_test_chat_request(question, top_k=3)
        request.knowledge_id = "nonexistent_knowledge_id"

        responses: List[CreateChatCompletionResponse] = []
        try:
            async for response in shared_index.client.rag_service_chat(request):
                responses.append(response)
        except Exception as e:
            logger.info(f"不存在的知识库ID产生预期异常: {e}")
            return

        # 如果没有异常，应该有合理的错误处理
        logger.info("不存在的知识库ID被正常处理")

    @pytest.mark.asyncio
    async def test_chat_complex_question(self, shared_index: BaseRAGTest):
        """测试复杂问题"""
        question = "比较Python和机器学习的关系，并说明如何使用Python进行深度学习开发？"
        request = shared_index.create_test_chat_request(question, top_k=3)

        responses: List[CreateChatCompletionResponse] = []
        async for response in shared_index.client.rag_service_chat(request):
            responses.append(response)

        shared_index.assert_chat_response(responses, expected_min_responses=1)

        # 验证复杂问题的回答质量
        full_answer = "".join([r.choices[0].message.content for r in responses])
        assert len(full_answer) > 50  # 复杂问题应该有较长的回答

        logger.info(f"复杂问题回答长度: {len(full_answer)} 字符")

    @pytest.mark.asyncio
    async def test_chat_response_structure(self, shared_index: BaseRAGTest):
        """测试聊天响应结构"""
        question = TestConfig.TEST_QUERIES[2]
        request = shared_index.create_test_chat_request(question, top_k=2)

        response_received = False
        async for response in shared_index.client.rag_service_chat(request):
            response_received = True

            # 验证响应结构
            assert response.choices is not None
            assert len(response.choices) > 0

            choice = response.choices[0]
            assert choice.index is not None
            assert choice.message is not None
            assert choice.message.role == "assistant"
            assert choice.message.content is not None

            if response.usage:
                assert response.usage.total_tokens >= 0

            logger.info(f"响应结构验证通过: index={choice.index}")

        assert response_received, "应该至少收到一个响应"
