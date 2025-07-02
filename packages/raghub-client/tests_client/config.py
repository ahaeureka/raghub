"""
RAGHub Client Test Configuration
Contains test server configuration and common settings
"""

import os
from typing import Any, Dict


class TestConfig:
    """Test configuration class"""

    # Server configuration
    BASE_URL = os.getenv("RAGHUB_TEST_BASE_URL", "http://localhost:8000")
    API_KEY = os.getenv("RAGHUB_TEST_API_KEY", "")
    TIMEOUT = float(os.getenv("RAGHUB_TEST_TIMEOUT", "60.0"))

    # Server startup configuration
    AUTO_START_SERVER = os.getenv("RAGHUB_AUTO_START_SERVER", "true").lower() == "true"
    SERVER_CONFIG_PATH = os.getenv("RAGHUB_SERVER_CONFIG", "configs/test.toml")
    SERVER_STARTUP_TIMEOUT = int(os.getenv("RAGHUB_SERVER_STARTUP_TIMEOUT", "60"))

    # Test data configuration
    TEST_KNOWLEDGE_ID = "test_knowledge_base"
    TEST_INDEX_NAME = "test_index"

    # Test documents
    TEST_DOCUMENTS = [
        {
            "content": (
                "Python is a high-level programming language widely used for "
                "web development, data science, and artificial intelligence."
            ),
            "title": "Python Introduction",
            "metadata": {"category": "programming", "language": "python"},
            "type": "text",
            "source": "knowledge_base",
        },
        {
            "content": (
                "Machine learning is a branch of artificial intelligence that "
                "enables computers to learn without being explicitly programmed."
            ),
            "title": "Machine Learning Overview",
            "metadata": {"category": "ai", "topic": "machine_learning"},
            "type": "text",
            "source": "knowledge_base",
        },
        {
            "content": (
                "Deep learning is a subset of machine learning that uses neural "
                "networks to mimic the learning process of the human brain."
            ),
            "title": "Deep Learning Introduction",
            "metadata": {"category": "ai", "topic": "deep_learning"},
            "type": "text",
            "source": "knowledge_base",
        },
    ]

    # Test queries
    TEST_QUERIES = [
        "What is Python?",
        "What is the difference between machine learning and deep learning?",
        "How to start learning artificial intelligence?",
    ]

    # Test chat messages
    TEST_CHAT_MESSAGES = [
        {
            "role": "user",
            "content": "Please explain the characteristics of Python programming language",
        }
    ]

    @classmethod
    def get_client_config(cls) -> Dict[str, Any]:
        """Get client configuration"""
        return {"base_url": cls.BASE_URL, "api_key": cls.API_KEY, "timeout": cls.TIMEOUT}
