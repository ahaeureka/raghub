"""
RAGHub Core Tests - Pytest配置文件

提供共享的fixtures和测试配置
"""

import pytest

# 唯一标识 - 用于区分不同包的conftest
__package_name__ = "raghub_core_tests"


@pytest.fixture(scope="session")
def qdrant_available():
    """检查qdrant-client是否可用"""
    try:
        import qdrant_client  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.fixture
def sample_metadata_condition():
    """提供样例元数据条件"""
    return {
        "logical_operator": "and",
        "conditions": [
            {"name": ["category"], "comparison_operator": "is", "value": "programming"},
            {"name": ["priority"], "comparison_operator": ">=", "value": "3"},
        ],
    }


@pytest.fixture
def sample_or_condition():
    """提供样例OR条件"""
    return {
        "logical_operator": "or",
        "conditions": [
            {"name": ["category"], "comparison_operator": "is", "value": "programming"},
            {"name": ["category"], "comparison_operator": "is", "value": "tutorial"},
        ],
    }


@pytest.fixture
def empty_condition():
    """提供空条件"""
    return {
        "logical_operator": "and",
        "conditions": [],
    }


@pytest.fixture
def complex_condition():
    """提供复杂条件"""
    return {
        "logical_operator": "and",
        "conditions": [
            {"name": ["category"], "comparison_operator": "is", "value": "programming"},
            {"name": ["tags"], "comparison_operator": "contains", "value": "python"},
            {"name": ["priority"], "comparison_operator": ">=", "value": "3"},
            {"name": ["status"], "comparison_operator": "is not", "value": "archived"},
            {"name": ["created_date"], "comparison_operator": "after", "value": "2024-01-01"},
        ],
    }


@pytest.fixture
def sample_documents():
    """提供测试用的文档样例"""
    from raghub_core.schemas.document import Document

    return [
        Document(
            content="Python is a programming language",
            metadata={"category": "programming", "priority": 5, "tags": ["python", "tutorial"]},
            uid="doc1",
            summary="Python programming tutorial",
        ),
        Document(
            content="Machine learning with Python",
            metadata={"category": "machine-learning", "priority": 3, "tags": ["ai", "ml", "python"]},
            uid="doc2",
            summary="ML guide using Python",
        ),
        Document(
            content="JavaScript fundamentals",
            metadata={"category": "programming", "priority": 4, "tags": ["javascript", "beginner"]},
            uid="doc3",
            summary="Basic JavaScript concepts",
        ),
    ]


@pytest.fixture(scope="session")
def langchain_qdrant_available():
    """检查langchain-qdrant是否可用"""
    try:
        import langchain_qdrant  # noqa: F401

        return True
    except ImportError:
        return False
