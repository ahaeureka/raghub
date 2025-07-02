#!/usr/bin/env python3
"""
RAGHub Client 测试示例
演示如何手动运行单个测试场景
"""

import asyncio
import logging
import os
import sys

# 添加路径以便导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from raghub_client.rag_hub_client import RAGHubClient
from raghub_protos.models.chat_model import ChatMessage, CreateChatCompletionRequest
from raghub_protos.models.rag_model import (
    AddDocumentsRequest,
    CreateIndexRequest,
    RAGDocument,
    RetrievalRequest,
    RetrievalSetting,
)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_basic_workflow():
    """演示基本的RAG工作流程"""
    # 配置客户端
    base_url = os.getenv("RAGHUB_TEST_BASE_URL", "http://localhost:8000")
    api_key = os.getenv("RAGHUB_TEST_API_KEY", "")

    client = RAGHubClient(base_url=base_url, api_key=api_key, timeout=30.0)

    # 生成唯一的知识库ID
    import uuid

    knowledge_id = f"demo_kb_{uuid.uuid4().hex[:8]}"

    logger.info(f"开始演示RAG工作流程，知识库ID: {knowledge_id}")

    try:
        # 1. 创建索引
        logger.info("1. 创建索引...")
        create_request = CreateIndexRequest(unique_name=knowledge_id)
        create_response = await client.rag_service_create_index(create_request)
        logger.info(f"   索引创建成功: {create_response.unique_name}")

        # 2. 添加文档
        logger.info("2. 添加测试文档...")
        documents = [
            RAGDocument(
                content="Python是一种高级编程语言，以其简洁和可读性而闻名。它广泛用于Web开发、数据科学、人工智能等领域。",
                title="Python编程语言介绍",
                metadata={"category": "programming", "language": "python", "difficulty": "beginner"},
                type="text",
                source="demo",
            ),
            RAGDocument(
                content="机器学习是人工智能的一个分支，它使计算机能够在没有明确编程的情况下学习和改进。常见的机器学习算法包括监督学习、无监督学习和强化学习。",
                title="机器学习基础",
                metadata={"category": "ai", "topic": "machine_learning", "difficulty": "intermediate"},
                type="text",
                source="demo",
            ),
            RAGDocument(
                content="深度学习是机器学习的一个子集，它使用多层神经网络来模拟人脑的学习过程。深度学习在图像识别、自然语言处理等领域取得了突破性进展。",
                title="深度学习概述",
                metadata={"category": "ai", "topic": "deep_learning", "difficulty": "advanced"},
                type="text",
                source="demo",
            ),
        ]

        add_request = AddDocumentsRequest(knowledge_id=knowledge_id, documents=documents)
        add_response = await client.rag_service_add_documents(add_request)
        logger.info(f"   成功添加 {len(add_response.documents)} 个文档")

        # 3. 测试检索
        logger.info("3. 测试文档检索...")
        retrieval_setting = RetrievalSetting(top_k=3)
        retrieval_request = RetrievalRequest(
            knowledge_id=knowledge_id, query="什么是Python编程语言？", retrieval_setting=retrieval_setting
        )

        retrieval_response = await client.rag_service_retrieval(retrieval_request)
        logger.info(f"   检索到 {len(retrieval_response.records)} 个相关文档:")

        for i, record in enumerate(retrieval_response.records, 1):
            logger.info(f"   文档{i}: {record.title}")
            logger.info(f"          相关性得分: {record.score:.3f}")
            logger.info(f"          内容摘要: {record.content[:100]}...")

        # 4. 测试聊天
        logger.info("4. 测试智能问答...")
        messages = [ChatMessage(role="user", content="请比较Python和机器学习的关系")]
        chat_request = CreateChatCompletionRequest(
            knowledge_id=knowledge_id, messages=messages, retrieval_setting=retrieval_setting
        )

        logger.info("   AI回答:")
        full_answer = ""
        async for response in client.rag_service_chat(chat_request):
            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                full_answer += content
                print(content, end="", flush=True)

        print()  # 换行
        logger.info(f"   完整回答长度: {len(full_answer)} 字符")

        logger.info("✅ RAG工作流程演示完成！")

    except Exception as e:
        logger.error(f"❌ 演示过程中发生错误: {e}")
        raise


async def demo_error_handling():
    """演示错误处理"""
    base_url = os.getenv("RAGHUB_TEST_BASE_URL", "http://localhost:8000")
    client = RAGHubClient(base_url=base_url, timeout=30.0)

    logger.info("演示错误处理场景...")

    # 测试不存在的知识库
    try:
        retrieval_setting = RetrievalSetting(top_k=3)
        retrieval_request = RetrievalRequest(
            knowledge_id="nonexistent_kb", query="测试查询", retrieval_setting=retrieval_setting
        )

        response = await client.rag_service_retrieval(retrieval_request)
        if response.error:
            logger.info(f"预期的错误处理: {response.error.error_msg}")
        else:
            logger.info("系统正常处理了不存在的知识库")

    except Exception as e:
        logger.info(f"捕获到异常（正常）: {e}")


async def main():
    """主函数"""
    print("=" * 60)
    print("RAGHub Client 测试演示")
    print("=" * 60)

    # 检查环境配置
    base_url = os.getenv("RAGHUB_TEST_BASE_URL")
    if not base_url:
        print("❌ 请设置环境变量 RAGHUB_TEST_BASE_URL")
        print("例如: export RAGHUB_TEST_BASE_URL='http://localhost:8000'")
        return

    print(f"服务器地址: {base_url}")
    print("-" * 60)

    try:
        # 运行基本工作流程演示
        await demo_basic_workflow()
        print()

        # 运行错误处理演示
        await demo_error_handling()

    except Exception as e:
        logger.error(f"演示失败: {e}")
        print("\n请确保RAGHub服务器正在运行并可访问。")

    print("=" * 60)
    print("演示结束")


if __name__ == "__main__":
    asyncio.run(main())
