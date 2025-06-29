"""
RAGHub Client 测试配置文件
包含测试服务器配置和通用设置
"""

import os
from typing import Any, Dict


class TestConfig:
    """测试配置类"""

    # 服务器配置
    BASE_URL = os.getenv("RAGHUB_TEST_BASE_URL", "http://localhost:8000")
    API_KEY = os.getenv("RAGHUB_TEST_API_KEY", "")
    TIMEOUT = float(os.getenv("RAGHUB_TEST_TIMEOUT", "60.0"))

    # 测试数据配置
    TEST_KNOWLEDGE_ID = "test_knowledge_base"
    TEST_INDEX_NAME = "test_index"

    # 测试文档
    TEST_DOCUMENTS = [
        {
            "content": "Python是一种高级编程语言，广泛用于Web开发、数据科学和人工智能。",
            "title": "Python简介",
            "metadata": {"category": "programming", "language": "python"},
            "type": "text",
            "source": "knowledge_base",
        },
        {
            "content": "机器学习是人工智能的一个分支，使计算机能够在不被明确编程的情况下学习。",
            "title": "机器学习概述",
            "metadata": {"category": "ai", "topic": "machine_learning"},
            "type": "text",
            "source": "knowledge_base",
        },
        {
            "content": "深度学习是机器学习的一个子集，使用神经网络来模仿人脑的学习过程。",
            "title": "深度学习介绍",
            "metadata": {"category": "ai", "topic": "deep_learning"},
            "type": "text",
            "source": "knowledge_base",
        },
    ]

    # 测试查询
    TEST_QUERIES = ["什么是Python？", "机器学习和深度学习的区别是什么？", "如何开始学习人工智能？"]

    # 测试聊天消息
    TEST_CHAT_MESSAGES = [{"role": "user", "content": "请解释一下Python编程语言的特点"}]

    @classmethod
    def get_client_config(cls) -> Dict[str, Any]:
        """获取客户端配置"""
        return {"base_url": cls.BASE_URL, "api_key": cls.API_KEY, "timeout": cls.TIMEOUT}
