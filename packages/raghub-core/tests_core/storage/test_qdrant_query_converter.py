from __future__ import annotations

import pytest
from raghub_core.utils.qdrant_query_converter import QdrantQueryConverter, create_qdrant_filter

try:
    from qdrant_client.models import Filter

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False


# 本地fixture定义（从conftest_core.py移动过来）
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


class TestQdrantQueryConverter:
    """Qdrant查询转换器测试类"""

    @pytest.fixture
    def converter(self):
        """创建转换器实例"""
        return QdrantQueryConverter()

    @pytest.mark.skipif(not QDRANT_AVAILABLE, reason="qdrant-client not installed")
    def test_basic_equality_condition(self, converter):
        """测试基本等值查询条件"""
        condition = {
            "logical_operator": "and",
            "conditions": [{"name": ["category"], "comparison_operator": "is", "value": "programming"}],
        }

        filter_result = converter.convert_metadata_condition(condition)

        assert filter_result is not None
        assert isinstance(filter_result, Filter)
        assert hasattr(filter_result, "must")

    @pytest.mark.skipif(not QDRANT_AVAILABLE, reason="qdrant-client not installed")
    def test_contains_condition(self, converter):
        """测试包含查询条件"""
        condition = {
            "logical_operator": "and",
            "conditions": [{"name": ["content"], "comparison_operator": "contains", "value": "python"}],
        }

        filter_result = converter.convert_metadata_condition(condition)

        assert filter_result is not None
        assert isinstance(filter_result, Filter)

    @pytest.mark.skipif(not QDRANT_AVAILABLE, reason="qdrant-client not installed")
    def test_numeric_comparison_conditions(self, converter):
        """测试数值比较条件"""
        test_cases = [
            (">", "5"),
            ("<", "10"),
            (">=", "3"),
            ("<=", "8"),
            ("=", "7"),
            ("!=", "0"),
        ]

        for operator, value in test_cases:
            condition = {
                "logical_operator": "and",
                "conditions": [{"name": ["priority"], "comparison_operator": operator, "value": value}],
            }

            filter_result = converter.convert_metadata_condition(condition)
            assert filter_result is not None, f"Failed for operator: {operator}"

    @pytest.mark.skipif(not QDRANT_AVAILABLE, reason="qdrant-client not installed")
    def test_and_logic_operator(self, converter, sample_metadata_condition):
        """测试AND逻辑操作符"""
        filter_result = converter.convert_metadata_condition(sample_metadata_condition)

        assert filter_result is not None
        assert isinstance(filter_result, Filter)
        assert hasattr(filter_result, "must")
        assert len(filter_result.must) == 2

    @pytest.mark.skipif(not QDRANT_AVAILABLE, reason="qdrant-client not installed")
    def test_or_logic_operator(self, converter, sample_or_condition):
        """测试OR逻辑操作符"""
        filter_result = converter.convert_metadata_condition(sample_or_condition)

        assert filter_result is not None
        assert isinstance(filter_result, Filter)
        assert hasattr(filter_result, "should")
        assert len(filter_result.should) == 2

    @pytest.mark.skipif(not QDRANT_AVAILABLE, reason="qdrant-client not installed")
    def test_negation_conditions(self, converter):
        """测试否定查询条件"""
        test_cases = [
            ("is not", "archived"),
            ("not contains", "deprecated"),
            ("≠", "deleted"),
        ]

        for operator, value in test_cases:
            condition = {
                "logical_operator": "and",
                "conditions": [{"name": ["status"], "comparison_operator": operator, "value": value}],
            }

            filter_result = converter.convert_metadata_condition(condition)
            assert filter_result is not None, f"Failed for negation operator: {operator}"

    @pytest.mark.skipif(not QDRANT_AVAILABLE, reason="qdrant-client not installed")
    def test_empty_and_null_conditions(self, converter):
        """测试空值和NULL查询条件"""
        test_cases = [
            ("empty", ""),
            ("not empty", ""),
            ("is null", ""),
            ("is not null", ""),
        ]

        for operator, value in test_cases:
            condition = {
                "logical_operator": "and",
                "conditions": [{"name": ["description"], "comparison_operator": operator, "value": value}],
            }

            filter_result = converter.convert_metadata_condition(condition)
            assert filter_result is not None, f"Failed for null/empty operator: {operator}"

    @pytest.mark.skipif(not QDRANT_AVAILABLE, reason="qdrant-client not installed")
    def test_date_conditions(self, converter):
        """测试日期查询条件"""
        test_cases = [
            ("after", "2024-01-01"),
            ("before", "2024-12-31"),
        ]

        for operator, value in test_cases:
            condition = {
                "logical_operator": "and",
                "conditions": [{"name": ["created_date"], "comparison_operator": operator, "value": value}],
            }

            filter_result = converter.convert_metadata_condition(condition)
            assert filter_result is not None, f"Failed for date operator: {operator}"

    @pytest.mark.skipif(not QDRANT_AVAILABLE, reason="qdrant-client not installed")
    def test_string_pattern_conditions(self, converter):
        """测试字符串模式查询条件"""
        test_cases = [
            ("start with", "prefix_"),
            ("end with", "_suffix"),
        ]

        for operator, value in test_cases:
            condition = {
                "logical_operator": "and",
                "conditions": [{"name": ["filename"], "comparison_operator": operator, "value": value}],
            }

            filter_result = converter.convert_metadata_condition(condition)
            assert filter_result is not None, f"Failed for string pattern operator: {operator}"

    @pytest.mark.skipif(not QDRANT_AVAILABLE, reason="qdrant-client not installed")
    def test_multiple_field_names(self, converter):
        """测试多字段名查询"""
        condition = {
            "logical_operator": "and",
            "conditions": [{"name": ["category", "tag", "type"], "comparison_operator": "is", "value": "python"}],
        }

        filter_result = converter.convert_metadata_condition(condition)

        assert filter_result is not None
        assert isinstance(filter_result, Filter)
        # 多字段名应该生成OR条件组合
        assert hasattr(filter_result, "must")

    def test_edge_cases_empty_conditions(self, converter, empty_condition):
        """测试边界情况：空条件列表"""
        filter_result = converter.convert_metadata_condition(empty_condition)
        assert filter_result is None

    def test_edge_cases_none_condition(self, converter):
        """测试边界情况：None条件"""
        filter_result = converter.convert_metadata_condition(None)
        assert filter_result is None

    def test_edge_cases_invalid_operator(self, converter):
        """测试边界情况：无效操作符"""
        condition = {
            "logical_operator": "and",
            "conditions": [{"name": ["field"], "comparison_operator": "invalid_operator", "value": "value"}],
        }

        filter_result = converter.convert_metadata_condition(condition)
        assert filter_result is None

    def test_edge_cases_missing_fields(self, converter):
        """测试边界情况：缺少必需字段"""
        # 缺少name字段
        condition1 = {
            "logical_operator": "and",
            "conditions": [{"comparison_operator": "is", "value": "value"}],
        }

        filter_result1 = converter.convert_metadata_condition(condition1)
        assert filter_result1 is None

        # 缺少comparison_operator字段
        condition2 = {
            "logical_operator": "and",
            "conditions": [{"name": ["field"], "value": "value"}],
        }

        filter_result2 = converter.convert_metadata_condition(condition2)
        assert filter_result2 is None

    def test_edge_cases_empty_field_names(self, converter):
        """测试边界情况：空字段名列表"""
        condition = {
            "logical_operator": "and",
            "conditions": [{"name": [], "comparison_operator": "is", "value": "value"}],
        }

        filter_result = converter.convert_metadata_condition(condition)
        assert filter_result is None

    @pytest.mark.skipif(not QDRANT_AVAILABLE, reason="qdrant-client not installed")
    def test_complex_nested_conditions(self, converter, complex_condition):
        """测试复杂嵌套条件"""
        filter_result = converter.convert_metadata_condition(complex_condition)

        assert filter_result is not None
        assert isinstance(filter_result, Filter)
        assert hasattr(filter_result, "must")
        assert len(filter_result.must) == 5

    @pytest.mark.skipif(not QDRANT_AVAILABLE, reason="qdrant-client not installed")
    def test_convenience_function(self):
        """测试便捷函数"""
        condition = {
            "logical_operator": "and",
            "conditions": [{"name": ["category"], "comparison_operator": "is", "value": "test"}],
        }

        filter_result = create_qdrant_filter(condition)

        assert filter_result is not None
        assert isinstance(filter_result, Filter)

    def test_convenience_function_with_none(self):
        """测试便捷函数处理None情况"""
        filter_result = create_qdrant_filter(None)
        assert filter_result is None


class TestQdrantQueryConverterParametrized:
    """使用参数化测试的Qdrant查询转换器测试"""

    @pytest.fixture
    def converter(self):
        """创建转换器实例"""
        return QdrantQueryConverter()

    @pytest.mark.skipif(not QDRANT_AVAILABLE, reason="qdrant-client not installed")
    @pytest.mark.parametrize(
        "operator,value,expected_type",
        [
            ("is", "value", Filter),
            ("contains", "substring", Filter),
            (">", "5", Filter),
            (">=", "3", Filter),
            ("<", "10", Filter),
            ("<=", "8", Filter),
            ("=", "7", Filter),
            ("≠", "0", Filter),
            ("is not", "archived", Filter),
            ("not contains", "deprecated", Filter),
            ("empty", "", Filter),
            ("not empty", "", Filter),
            ("start with", "prefix", Filter),
            ("end with", "suffix", Filter),
            ("after", "2024-01-01", Filter),
            ("before", "2024-12-31", Filter),
        ],
    )
    def test_all_operators(self, converter, operator, value, expected_type):
        """参数化测试所有支持的操作符"""
        condition = {
            "logical_operator": "and",
            "conditions": [{"name": ["field"], "comparison_operator": operator, "value": value}],
        }

        filter_result = converter.convert_metadata_condition(condition)

        assert filter_result is not None
        assert isinstance(filter_result, expected_type)

    @pytest.mark.parametrize(
        "logical_op,expected_attr",
        [
            ("and", "must"),
            ("or", "should"),
        ],
    )
    @pytest.mark.skipif(not QDRANT_AVAILABLE, reason="qdrant-client not installed")
    def test_logical_operators(self, converter, logical_op, expected_attr):
        """参数化测试逻辑操作符"""
        condition = {
            "logical_operator": logical_op,
            "conditions": [
                {"name": ["field1"], "comparison_operator": "is", "value": "value1"},
                {"name": ["field2"], "comparison_operator": "is", "value": "value2"},
            ],
        }

        filter_result = converter.convert_metadata_condition(condition)

        assert filter_result is not None
        assert isinstance(filter_result, Filter)
        assert hasattr(filter_result, expected_attr)
        assert len(getattr(filter_result, expected_attr)) == 2

    @pytest.mark.parametrize(
        "invalid_input",
        [
            None,
            {},
            {"logical_operator": "and"},
            {"conditions": []},
            {"logical_operator": "invalid", "conditions": []},
            {"logical_operator": "and", "conditions": [{}]},
            {"logical_operator": "and", "conditions": [{"name": []}]},
            {"logical_operator": "and", "conditions": [{"name": ["field"]}]},
            {"logical_operator": "and", "conditions": [{"comparison_operator": "is"}]},
        ],
    )
    def test_invalid_inputs(self, converter, invalid_input):
        """参数化测试无效输入"""
        filter_result = converter.convert_metadata_condition(invalid_input)
        assert filter_result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
