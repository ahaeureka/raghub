"""
RAGHub Core Tests Configuration

This conftest.py provides fixtures and configuration specific to raghub-core package tests.
"""

__package_name__ = "raghub_core_tests"
__conftest_identifier__ = "raghub_core_conftest_20250702"

from pathlib import Path

import pytest


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


@pytest.fixture(scope="session")
def test_data_dir():
    """测试数据目录"""
    return Path(__file__).parent / "test_data"


@pytest.fixture(scope="session")
def sample_graph_data():
    """示例图数据"""
    return {
        "entities": [
            {"name": "Python", "type": "programming_language", "description": "A high-level programming language"},
            {"name": "Java", "type": "programming_language", "description": "An object-oriented programming language"},
            {"name": "Machine Learning", "type": "field", "description": "A subset of artificial intelligence"},
        ],
        "relations": [
            {
                "source": "Python",
                "target": "Machine Learning",
                "relation": "used_in",
                "description": "Python is commonly used in ML",
            },
            {
                "source": "Java",
                "target": "Machine Learning",
                "relation": "used_in",
                "description": "Java can be used in ML applications",
            },
        ],
    }


@pytest.fixture
def mock_config():
    """模拟配置"""
    return {
        "llm": {"model": "mock_model", "temperature": 0.7, "max_tokens": 1000},
        "embedding": {"model": "mock_embedding", "dimension": 768},
        "graph": {"max_entities_per_doc": 10, "min_entity_confidence": 0.5, "community_detection_algorithm": "leiden"},
    }
