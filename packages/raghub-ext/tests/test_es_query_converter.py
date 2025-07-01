"""
Elasticsearch查询转换器测试用例
"""

import pytest
from raghub_ext.storage_ext.utils.es_query_converter import ESQueryConverter


class TestESQueryConverter:
    """Elasticsearch查询转换器测试类"""

    @pytest.fixture
    def converter(self):
        """创建转换器实例"""
        return ESQueryConverter()

    def test_empty_condition(self, converter):
        """测试空条件"""
        result = converter.convert_metadata_condition({})
        expected = {"match_all": {}}
        assert result == expected

    def test_missing_conditions(self, converter):
        """测试缺少conditions字段"""
        condition_dict = {"logical_operator": "and"}
        result = converter.convert_metadata_condition(condition_dict)
        expected = {"match_all": {}}
        assert result == expected

    def test_single_contains_condition(self, converter):
        """测试单个包含条件"""
        condition_dict = {
            "logical_operator": "and",
            "conditions": [{"name": ["category"], "comparison_operator": "contains", "value": "programming"}],
        }
        result = converter.convert_metadata_condition(condition_dict)
        expected = {"wildcard": {"category.keyword": "*programming*"}}
        assert result == expected

    def test_single_equals_condition(self, converter):
        """测试单个等于条件"""
        condition_dict = {
            "logical_operator": "and",
            "conditions": [{"name": ["status"], "comparison_operator": "is", "value": "published"}],
        }
        result = converter.convert_metadata_condition(condition_dict)
        expected = {"term": {"status.keyword": "published"}}
        assert result == expected

    def test_multiple_field_names(self, converter):
        """测试多个字段名的条件"""
        condition_dict = {
            "logical_operator": "and",
            "conditions": [{"name": ["category", "tag"], "comparison_operator": "contains", "value": "python"}],
        }
        result = converter.convert_metadata_condition(condition_dict)
        expected = {
            "bool": {
                "should": [{"wildcard": {"category.keyword": "*python*"}}, {"wildcard": {"tag.keyword": "*python*"}}],
                "minimum_should_match": 1,
            }
        }
        assert result == expected

    def test_and_logical_operator(self, converter):
        """测试AND逻辑操作符"""
        condition_dict = {
            "logical_operator": "and",
            "conditions": [
                {"name": ["category"], "comparison_operator": "is", "value": "programming"},
                {"name": ["status"], "comparison_operator": "is", "value": "published"},
            ],
        }
        result = converter.convert_metadata_condition(condition_dict)
        expected = {
            "bool": {"must": [{"term": {"category.keyword": "programming"}}, {"term": {"status.keyword": "published"}}]}
        }
        assert result == expected

    def test_or_logical_operator(self, converter):
        """测试OR逻辑操作符"""
        condition_dict = {
            "logical_operator": "or",
            "conditions": [
                {"name": ["category"], "comparison_operator": "is", "value": "programming"},
                {"name": ["category"], "comparison_operator": "is", "value": "technology"},
            ],
        }
        result = converter.convert_metadata_condition(condition_dict)
        expected = {
            "bool": {
                "should": [{"term": {"category.keyword": "programming"}}, {"term": {"category.keyword": "technology"}}],
                "minimum_should_match": 1,
            }
        }
        assert result == expected

    def test_not_contains_condition(self, converter):
        """测试不包含条件"""
        condition_dict = {
            "logical_operator": "and",
            "conditions": [{"name": ["description"], "comparison_operator": "not contains", "value": "deprecated"}],
        }
        result = converter.convert_metadata_condition(condition_dict)
        expected = {"bool": {"must_not": [{"wildcard": {"description.keyword": "*deprecated*"}}]}}
        assert result == expected

    def test_starts_with_condition(self, converter):
        """测试以...开头条件"""
        condition_dict = {
            "logical_operator": "and",
            "conditions": [{"name": ["title"], "comparison_operator": "start with", "value": "How to"}],
        }
        result = converter.convert_metadata_condition(condition_dict)
        expected = {"prefix": {"title.keyword": "How to"}}
        assert result == expected

    def test_ends_with_condition(self, converter):
        """测试以...结尾条件"""
        condition_dict = {
            "logical_operator": "and",
            "conditions": [{"name": ["filename"], "comparison_operator": "end with", "value": ".pdf"}],
        }
        result = converter.convert_metadata_condition(condition_dict)
        expected = {"wildcard": {"filename.keyword": "*.pdf"}}
        assert result == expected

    def test_numeric_greater_than_condition(self, converter):
        """测试数值大于条件"""
        condition_dict = {
            "logical_operator": "and",
            "conditions": [{"name": ["score"], "comparison_operator": ">", "value": "85.5"}],
        }
        result = converter.convert_metadata_condition(condition_dict)
        expected = {"range": {"score": {"gt": 85.5}}}
        assert result == expected

    def test_numeric_less_than_condition(self, converter):
        """测试数值小于条件"""
        condition_dict = {
            "logical_operator": "and",
            "conditions": [{"name": ["age"], "comparison_operator": "<", "value": "30"}],
        }
        result = converter.convert_metadata_condition(condition_dict)
        expected = {"range": {"age": {"lt": 30}}}
        assert result == expected

    def test_numeric_equals_condition(self, converter):
        """测试数值等于条件"""
        condition_dict = {
            "logical_operator": "and",
            "conditions": [{"name": ["priority"], "comparison_operator": "=", "value": "1"}],
        }
        result = converter.convert_metadata_condition(condition_dict)
        expected = {"term": {"priority": 1}}
        assert result == expected

    def test_date_before_condition(self, converter):
        """测试日期早于条件"""
        condition_dict = {
            "logical_operator": "and",
            "conditions": [{"name": ["created_date"], "comparison_operator": "before", "value": "2024-01-01"}],
        }
        result = converter.convert_metadata_condition(condition_dict)
        expected = {"range": {"created_date": {"lt": "2024-01-01"}}}
        assert result == expected

    def test_date_after_condition(self, converter):
        """测试日期晚于条件"""
        condition_dict = {
            "logical_operator": "and",
            "conditions": [{"name": ["updated_date"], "comparison_operator": "after", "value": "2024-01-01"}],
        }
        result = converter.convert_metadata_condition(condition_dict)
        expected = {"range": {"updated_date": {"gt": "2024-01-01"}}}
        assert result == expected

    def test_empty_condition_check(self, converter):
        """测试字段为空条件"""
        condition_dict = {
            "logical_operator": "and",
            "conditions": [{"name": ["description"], "comparison_operator": "empty"}],
        }
        result = converter.convert_metadata_condition(condition_dict)
        expected = {
            "bool": {
                "should": [
                    {"bool": {"must_not": [{"exists": {"field": "description"}}]}},
                    {"term": {"description.keyword": ""}},
                ],
                "minimum_should_match": 1,
            }
        }
        assert result == expected

    def test_not_empty_condition_check(self, converter):
        """测试字段非空条件"""
        condition_dict = {
            "logical_operator": "and",
            "conditions": [{"name": ["description"], "comparison_operator": "not empty"}],
        }
        result = converter.convert_metadata_condition(condition_dict)
        expected = {
            "bool": {
                "must": [{"exists": {"field": "description"}}],
                "must_not": [{"term": {"description.keyword": ""}}],
            }
        }
        assert result == expected

    def test_unsupported_operator(self, converter):
        """测试不支持的操作符"""
        condition_dict = {
            "logical_operator": "and",
            "conditions": [{"name": ["category"], "comparison_operator": "unsupported_op", "value": "test"}],
        }
        result = converter.convert_metadata_condition(condition_dict)
        expected = {"match_all": {}}
        assert result == expected

    def test_invalid_condition_missing_name(self, converter):
        """测试无效条件：缺少字段名"""
        condition_dict = {"logical_operator": "and", "conditions": [{"comparison_operator": "is", "value": "test"}]}
        result = converter.convert_metadata_condition(condition_dict)
        expected = {"match_all": {}}
        assert result == expected

    def test_invalid_condition_missing_operator(self, converter):
        """测试无效条件：缺少操作符"""
        condition_dict = {"logical_operator": "and", "conditions": [{"name": ["category"], "value": "test"}]}
        result = converter.convert_metadata_condition(condition_dict)
        expected = {"match_all": {}}
        assert result == expected

    def test_complex_nested_condition(self, converter):
        """测试复杂嵌套条件"""
        condition_dict = {
            "logical_operator": "and",
            "conditions": [
                {"name": ["category"], "comparison_operator": "is", "value": "programming"},
                {"name": ["rating", "score"], "comparison_operator": "≥", "value": "4.0"},
                {"name": ["status"], "comparison_operator": "is not", "value": "draft"},
            ],
        }
        result = converter.convert_metadata_condition(condition_dict)
        expected = {
            "bool": {
                "must": [
                    {"term": {"category.keyword": "programming"}},
                    {
                        "bool": {
                            "should": [{"range": {"rating": {"gte": 4.0}}}, {"range": {"score": {"gte": 4.0}}}],
                            "minimum_should_match": 1,
                        }
                    },
                    {"bool": {"must_not": [{"term": {"status.keyword": "draft"}}]}},
                ]
            }
        }
        assert result == expected

    def test_build_full_es_query(self, converter):
        """测试构建完整ES查询"""
        metadata_condition = {
            "logical_operator": "and",
            "conditions": [{"name": ["category"], "comparison_operator": "is", "value": "programming"}],
        }

        result = converter.build_es_query(
            metadata_condition=metadata_condition, query_text="python tutorial", size=20, from_=10
        )

        expected = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": "python tutorial",
                                "fields": ["content^2", "title^1.5", "metadata.*"],
                            }
                        },
                        {"term": {"category.keyword": "programming"}},
                    ]
                }
            },
            "size": 20,
            "from": 10,
            "_source": {"excludes": ["vector"]},
        }

        assert result == expected

    def test_build_es_query_text_only(self, converter):
        """测试只有文本查询的ES查询"""
        result = converter.build_es_query(query_text="machine learning", size=5)

        expected = {
            "query": {"multi_match": {"query": "machine learning", "fields": ["content^2", "title^1.5", "metadata.*"]}},
            "size": 5,
            "from": 0,
            "_source": {"excludes": ["vector"]},
        }

        assert result == expected

    def test_build_es_query_metadata_only(self, converter):
        """测试只有元数据查询的ES查询"""
        metadata_condition = {
            "logical_operator": "or",
            "conditions": [
                {"name": ["type"], "comparison_operator": "is", "value": "article"},
                {"name": ["type"], "comparison_operator": "is", "value": "tutorial"},
            ],
        }

        result = converter.build_es_query(metadata_condition=metadata_condition)

        expected = {
            "query": {
                "bool": {
                    "should": [{"term": {"type.keyword": "article"}}, {"term": {"type.keyword": "tutorial"}}],
                    "minimum_should_match": 1,
                }
            },
            "size": 10,
            "from": 0,
            "_source": {"excludes": ["vector"]},
        }

        assert result == expected

    def test_build_es_query_empty(self, converter):
        """测试空查询"""
        result = converter.build_es_query()

        expected = {"query": {"match_all": {}}, "size": 10, "from": 0, "_source": {"excludes": ["vector"]}}

        assert result == expected

    def test_is_numeric(self, converter):
        """测试数字判断"""
        assert converter._is_numeric("123") is True
        assert converter._is_numeric("123.45") is True
        assert converter._is_numeric("-123") is True
        assert converter._is_numeric("abc") is False
        assert converter._is_numeric("") is False

    def test_to_number(self, converter):
        """测试数字转换"""
        assert converter._to_number("123") == 123
        assert converter._to_number("123.45") == 123.45
        assert converter._to_number("-123") == -123
        assert converter._to_number("abc") == 0

    def test_is_date(self, converter):
        """测试日期判断"""
        assert converter._is_date("2024-01-01") is True
        assert converter._is_date("2024-01-01 10:30:00") is True
        assert converter._is_date("2024/01/01") is True
        assert converter._is_date("01/01/2024") is True
        assert converter._is_date("not-a-date") is False
        assert converter._is_date("") is False

    def test_eq_operator(self, converter):
        """测试eq操作符"""
        condition_dict = {
            "logical_operator": "and",
            "conditions": [{"name": ["status"], "comparison_operator": "eq", "value": "active"}],
        }
        result = converter.convert_metadata_condition(condition_dict)
        expected = {"term": {"status.keyword": "active"}}
        assert result == expected

    def test_ne_operator(self, converter):
        """测试ne操作符"""
        condition_dict = {
            "logical_operator": "and",
            "conditions": [{"name": ["status"], "comparison_operator": "ne", "value": "inactive"}],
        }
        result = converter.convert_metadata_condition(condition_dict)
        expected = {"bool": {"must_not": [{"term": {"status.keyword": "inactive"}}]}}
        assert result == expected

    def test_lt_operator(self, converter):
        """测试lt操作符"""
        condition_dict = {
            "logical_operator": "and",
            "conditions": [{"name": ["score"], "comparison_operator": "lt", "value": "80"}],
        }
        result = converter.convert_metadata_condition(condition_dict)
        expected = {"range": {"score": {"lt": 80}}}
        assert result == expected

    def test_gt_operator(self, converter):
        """测试gt操作符"""
        condition_dict = {
            "logical_operator": "and",
            "conditions": [{"name": ["rating"], "comparison_operator": "gt", "value": "4.5"}],
        }
        result = converter.convert_metadata_condition(condition_dict)
        expected = {"range": {"rating": {"gt": 4.5}}}
        assert result == expected

    def test_le_operator(self, converter):
        """测试le操作符"""
        condition_dict = {
            "logical_operator": "and",
            "conditions": [{"name": ["age"], "comparison_operator": "le", "value": "65"}],
        }
        result = converter.convert_metadata_condition(condition_dict)
        expected = {"range": {"age": {"lte": 65}}}
        assert result == expected

    def test_ge_operator(self, converter):
        """测试ge操作符"""
        condition_dict = {
            "logical_operator": "and",
            "conditions": [{"name": ["price"], "comparison_operator": "ge", "value": "100.0"}],
        }
        result = converter.convert_metadata_condition(condition_dict)
        expected = {"range": {"price": {"gte": 100.0}}}
        assert result == expected

    def test_in_operator_numeric(self, converter):
        """测试in操作符（数值）"""
        condition_dict = {
            "logical_operator": "and",
            "conditions": [{"name": ["category_id"], "comparison_operator": "in", "value": "1,2,3,4"}],
        }
        result = converter.convert_metadata_condition(condition_dict)
        expected = {"terms": {"category_id": [1, 2, 3, 4]}}
        assert result == expected

    def test_in_operator_text(self, converter):
        """测试in操作符（文本）"""
        condition_dict = {
            "logical_operator": "and",
            "conditions": [{"name": ["status"], "comparison_operator": "in", "value": "active,pending,review"}],
        }
        result = converter.convert_metadata_condition(condition_dict)
        expected = {"terms": {"status.keyword": ["active", "pending", "review"]}}
        assert result == expected

    def test_notin_operator(self, converter):
        """测试notin操作符"""
        condition_dict = {
            "logical_operator": "and",
            "conditions": [{"name": ["status"], "comparison_operator": "notin", "value": "deleted,archived"}],
        }
        result = converter.convert_metadata_condition(condition_dict)
        expected = {"bool": {"must_not": [{"terms": {"status.keyword": ["deleted", "archived"]}}]}}
        assert result == expected

    def test_in_operator_single_value(self, converter):
        """测试in操作符（单个值）"""
        condition_dict = {
            "logical_operator": "and",
            "conditions": [{"name": ["priority"], "comparison_operator": "in", "value": "high"}],
        }
        result = converter.convert_metadata_condition(condition_dict)
        expected = {"terms": {"priority.keyword": ["high"]}}
        assert result == expected

    def test_mixed_new_operators(self, converter):
        """测试混合使用新操作符"""
        condition_dict = {
            "logical_operator": "and",
            "conditions": [
                {"name": ["status"], "comparison_operator": "in", "value": "active,published"},
                {"name": ["score"], "comparison_operator": "ge", "value": "85"},
                {"name": ["category"], "comparison_operator": "ne", "value": "spam"},
            ],
        }
        result = converter.convert_metadata_condition(condition_dict)
        expected = {
            "bool": {
                "must": [
                    {"terms": {"status.keyword": ["active", "published"]}},
                    {"range": {"score": {"gte": 85}}},
                    {"bool": {"must_not": [{"term": {"category.keyword": "spam"}}]}},
                ]
            }
        }
        assert result == expected
