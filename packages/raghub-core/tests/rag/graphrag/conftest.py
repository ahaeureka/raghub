"""
GraphRAG 测试配置和工具
"""

from pathlib import Path

import pytest


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
