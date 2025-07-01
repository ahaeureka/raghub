"""
Elasticsearch元数据查询测试用例
"""

from typing import List, Optional

import numpy as np
import pytest
from raghub_core.embedding.base_embedding import BaseEmbedding
from raghub_ext.storage_ext.elasticsearch_vector import ElasticsearchVectorStorage


class MockEmbedding(BaseEmbedding):
    """模拟的嵌入模型"""

    name = "mock_embedding"

    def __init__(self):
        super().__init__()

    @property
    def embedding_dim(self):
        return 768

    def encode_query(self, query: str, instruction: Optional[str] = None) -> np.ndarray:
        return np.array([0.1] * 768, dtype=np.float32)

    def encode(self, texts: List[str], instruction: Optional[str] = None) -> np.ndarray:
        return np.array([[0.1] * 768 for _ in texts], dtype=np.float32)

    async def aencode_query(self, query: List[str], instruction: Optional[str] = None) -> np.ndarray:
        if isinstance(query, list):
            return np.array([[0.1] * 768 for _ in query], dtype=np.float32)
        else:
            return np.array([0.1] * 768, dtype=np.float32)

    async def aencode(self, texts: List[str], instruction: Optional[str] = None) -> np.ndarray:
        return np.array([[0.1] * 768 for _ in texts], dtype=np.float32)


class TestElasticsearchMetadata:
    """Elasticsearch元数据查询测试类"""

    @pytest.fixture
    def es_storage(self):
        """创建ElasticsearchVectorStorage实例"""
        embedder = MockEmbedding()
        return ElasticsearchVectorStorage(
            embedder=embedder, host="localhost", port=9200, username="elastic", password="password"
        )

    def test_is_metadata_condition_format_true(self, es_storage):
        """测试识别MetadataCondition格式 - 正确格式"""
        metadata_filter = {
            "logical_operator": "and",
            "conditions": [{"name": ["category"], "comparison_operator": "eq", "value": "programming"}],
        }

        result = es_storage._is_metadata_condition_format(metadata_filter)
        assert result is True

    def test_is_metadata_condition_format_false_simple(self, es_storage):
        """测试识别MetadataCondition格式 - 简单格式"""
        metadata_filter = {"category": "programming", "status": "published"}

        result = es_storage._is_metadata_condition_format(metadata_filter)
        assert result is False

    def test_is_metadata_condition_format_false_empty_conditions(self, es_storage):
        """测试识别MetadataCondition格式 - 空条件列表"""
        metadata_filter = {"logical_operator": "and", "conditions": []}

        result = es_storage._is_metadata_condition_format(metadata_filter)
        assert result is False

    def test_is_metadata_condition_format_false_missing_fields(self, es_storage):
        """测试识别MetadataCondition格式 - 缺少必需字段"""
        metadata_filter = {
            "logical_operator": "and",
            "conditions": [{"comparison_operator": "eq", "value": "programming"}],
        }

        result = es_storage._is_metadata_condition_format(metadata_filter)
        assert result is False

    def test_build_simple_metadata_query(self, es_storage):
        """测试构建简单元数据查询"""
        metadata_filter = {"category": "programming", "tags": ["python", "tutorial"]}

        result = es_storage._build_simple_metadata_query(metadata_filter)
        expected = {
            "bool": {
                "must": [
                    {"term": {"metadata.category.keyword": "programming"}},
                    {"terms": {"metadata.tags.keyword": ["python", "tutorial"]}},
                ]
            }
        }

        assert result == expected

    def test_build_metadata_query_simple_format(self, es_storage):
        """测试_build_metadata_query方法 - 简单格式"""
        metadata_filter = {"category": "programming", "status": "published"}

        result = es_storage._build_metadata_query(metadata_filter)
        expected = {
            "bool": {
                "must": [
                    {"term": {"metadata.category.keyword": "programming"}},
                    {"term": {"metadata.status.keyword": "published"}},
                ]
            }
        }

        assert result == expected

    def test_build_metadata_query_condition_format(self, es_storage):
        """测试_build_metadata_query方法 - MetadataCondition格式"""
        metadata_filter = {
            "logical_operator": "and",
            "conditions": [
                {"name": ["category"], "comparison_operator": "eq", "value": "programming"},
                {"name": ["rating"], "comparison_operator": "gt", "value": "4.0"},
            ],
        }

        result = es_storage._build_metadata_query(metadata_filter)
        expected = {
            "bool": {
                "must": [
                    {"term": {"metadata.category.keyword": "programming"}},
                    {"range": {"metadata.rating": {"gt": 4.0}}},
                ]
            }
        }

        assert result == expected

    def test_build_metadata_query_condition_format_or(self, es_storage):
        """测试_build_metadata_query方法 - MetadataCondition格式使用OR逻辑"""
        metadata_filter = {
            "logical_operator": "or",
            "conditions": [
                {"name": ["category"], "comparison_operator": "eq", "value": "programming"},
                {"name": ["category"], "comparison_operator": "eq", "value": "technology"},
            ],
        }

        result = es_storage._build_metadata_query(metadata_filter)
        expected = {
            "bool": {
                "should": [
                    {"term": {"metadata.category.keyword": "programming"}},
                    {"term": {"metadata.category.keyword": "technology"}},
                ],
                "minimum_should_match": 1,
            }
        }

        assert result == expected

    def test_build_metadata_query_condition_format_in_operator(self, es_storage):
        """测试_build_metadata_query方法 - 使用in操作符"""
        metadata_filter = {
            "logical_operator": "and",
            "conditions": [
                {"name": ["category"], "comparison_operator": "in", "value": "programming,technology,science"}
            ],
        }

        result = es_storage._build_metadata_query(metadata_filter)
        expected = {"terms": {"metadata.category.keyword": ["programming", "technology", "science"]}}

        assert result == expected

    def test_build_metadata_query_condition_format_multi_fields(self, es_storage):
        """测试_build_metadata_query方法 - 多字段条件"""
        metadata_filter = {
            "logical_operator": "and",
            "conditions": [{"name": ["title", "description"], "comparison_operator": "contains", "value": "python"}],
        }

        result = es_storage._build_metadata_query(metadata_filter)
        expected = {
            "bool": {
                "should": [
                    {"wildcard": {"metadata.title.keyword": "*python*"}},
                    {"wildcard": {"metadata.description.keyword": "*python*"}},
                ],
                "minimum_should_match": 1,
            }
        }

        assert result == expected

    def test_build_metadata_query_empty_conditions(self, es_storage):
        """测试_build_metadata_query方法 - 空条件"""
        metadata_filter = {"logical_operator": "and", "conditions": []}

        result = es_storage._build_metadata_query(metadata_filter)
        expected = {"match_all": {}}

        assert result == expected

    def test_build_metadata_query_complex_condition(self, es_storage):
        """测试_build_metadata_query方法 - 复杂条件"""
        metadata_filter = {
            "logical_operator": "and",
            "conditions": [
                {"name": ["category"], "comparison_operator": "eq", "value": "programming"},
                {"name": ["rating"], "comparison_operator": "ge", "value": "4.0"},
                {"name": ["status"], "comparison_operator": "ne", "value": "draft"},
                {"name": ["tags"], "comparison_operator": "in", "value": "python,tutorial,beginner"},
            ],
        }

        result = es_storage._build_metadata_query(metadata_filter)
        expected = {
            "bool": {
                "must": [
                    {"term": {"metadata.category.keyword": "programming"}},
                    {"range": {"metadata.rating": {"gte": 4.0}}},
                    {"bool": {"must_not": [{"term": {"metadata.status.keyword": "draft"}}]}},
                    {"terms": {"metadata.tags.keyword": ["python", "tutorial", "beginner"]}},
                ]
            }
        }

        assert result == expected
