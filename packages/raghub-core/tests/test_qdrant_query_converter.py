"""
Qdrant查询转换器测试用例
"""

import pytest
from raghub_core.utils.qdrant_query_converter import QdrantQueryConverter


class TestQdrantQueryConverter:
    """Qdrant查询转换器测试类"""

    @pytest.fixture
    def converter(self):
        """创建转换器实例"""
        return QdrantQueryConverter()

    def test_empty_condition(self, converter):
        """测试空条件"""
        result = converter.convert_metadata_condition({})
        assert result is None

    def test_missing_conditions(self, converter):
        """测试缺少conditions字段"""
        condition_dict = {"logical_operator": "and"}
        result = converter.convert_metadata_condition(condition_dict)
        assert result is None

    def test_eq_operator(self, converter):
        """测试eq操作符"""
        condition_dict = {
            "logical_operator": "and",
            "conditions": [{"name": ["status"], "comparison_operator": "eq", "value": "active"}],
        }
        filter_result = converter.convert_metadata_condition(condition_dict)
        assert filter_result is not None
        # 检查Filter的must条件
        assert len(filter_result.must) == 1
        field_condition = filter_result.must[0]
        assert field_condition.key == "status"
        assert field_condition.match.value == "active"

    def test_ne_operator(self, converter):
        """测试ne操作符"""
        condition_dict = {
            "logical_operator": "and",
            "conditions": [{"name": ["status"], "comparison_operator": "ne", "value": "inactive"}],
        }
        filter_result = converter.convert_metadata_condition(condition_dict)
        assert filter_result is not None
        # 检查Filter的must条件
        assert len(filter_result.must) == 1
        field_condition = filter_result.must[0]
        assert field_condition.key == "status"
        # ne操作符使用MatchExcept
        assert hasattr(field_condition.match, "except_")
        assert "inactive" in getattr(field_condition.match, "except_")

    def test_lt_operator(self, converter):
        """测试lt操作符"""
        condition_dict = {
            "logical_operator": "and",
            "conditions": [{"name": ["score"], "comparison_operator": "lt", "value": "80"}],
        }
        filter_result = converter.convert_metadata_condition(condition_dict)
        assert filter_result is not None
        # 检查Filter的must条件
        assert len(filter_result.must) == 1
        field_condition = filter_result.must[0]
        assert field_condition.key == "score"
        assert field_condition.range.lt == 80

    def test_gt_operator(self, converter):
        """测试gt操作符"""
        condition_dict = {
            "logical_operator": "and",
            "conditions": [{"name": ["rating"], "comparison_operator": "gt", "value": "4.5"}],
        }
        filter_result = converter.convert_metadata_condition(condition_dict)
        assert filter_result is not None
        # 检查Filter的must条件
        assert len(filter_result.must) == 1
        field_condition = filter_result.must[0]
        assert field_condition.key == "rating"
        assert field_condition.range.gt == 4.5

    def test_le_operator(self, converter):
        """测试le操作符"""
        condition_dict = {
            "logical_operator": "and",
            "conditions": [{"name": ["age"], "comparison_operator": "le", "value": "65"}],
        }
        filter_result = converter.convert_metadata_condition(condition_dict)
        assert filter_result is not None
        # 检查Filter的must条件
        assert len(filter_result.must) == 1
        field_condition = filter_result.must[0]
        assert field_condition.key == "age"
        assert field_condition.range.lte == 65

    def test_ge_operator(self, converter):
        """测试ge操作符"""
        condition_dict = {
            "logical_operator": "and",
            "conditions": [{"name": ["price"], "comparison_operator": "ge", "value": "100.0"}],
        }
        filter_result = converter.convert_metadata_condition(condition_dict)
        assert filter_result is not None
        # 检查Filter的must条件
        assert len(filter_result.must) == 1
        field_condition = filter_result.must[0]
        assert field_condition.key == "price"
        assert field_condition.range.gte == 100.0

    def test_in_operator_comma_separated(self, converter):
        """测试in操作符（逗号分隔）"""
        condition_dict = {
            "logical_operator": "and",
            "conditions": [{"name": ["category"], "comparison_operator": "in", "value": "tech,science,programming"}],
        }
        filter_result = converter.convert_metadata_condition(condition_dict)
        assert filter_result is not None
        # 检查Filter的must条件
        assert len(filter_result.must) == 1
        field_condition = filter_result.must[0]
        assert field_condition.key == "category"
        # in操作符使用MatchAny
        assert hasattr(field_condition.match, "any")
        assert "tech" in field_condition.match.any
        assert "science" in field_condition.match.any
        assert "programming" in field_condition.match.any

    def test_in_operator_json_array(self, converter):
        """测试in操作符（JSON数组格式）"""
        condition_dict = {
            "logical_operator": "and",
            "conditions": [
                {"name": ["priority"], "comparison_operator": "in", "value": '["high", "medium", "urgent"]'}
            ],
        }
        filter_result = converter.convert_metadata_condition(condition_dict)
        assert filter_result is not None
        # 检查Filter的must条件
        assert len(filter_result.must) == 1
        field_condition = filter_result.must[0]
        assert field_condition.key == "priority"
        # in操作符使用MatchAny
        assert hasattr(field_condition.match, "any")
        assert "high" in field_condition.match.any
        assert "medium" in field_condition.match.any
        assert "urgent" in field_condition.match.any

    def test_notin_operator(self, converter):
        """测试notin操作符"""
        condition_dict = {
            "logical_operator": "and",
            "conditions": [{"name": ["status"], "comparison_operator": "notin", "value": "deleted,archived,banned"}],
        }
        filter_result = converter.convert_metadata_condition(condition_dict)
        assert filter_result is not None
        # 检查Filter的must条件
        assert len(filter_result.must) == 1
        field_condition = filter_result.must[0]
        assert field_condition.key == "status"
        # notin操作符使用MatchExcept
        assert hasattr(field_condition.match, "except_")
        assert "deleted" in getattr(field_condition.match, "except_")
        assert "archived" in getattr(field_condition.match, "except_")
        assert "banned" in getattr(field_condition.match, "except_")

    def test_in_operator_numeric_values(self, converter):
        """测试in操作符（数值）"""
        condition_dict = {
            "logical_operator": "and",
            "conditions": [{"name": ["category_id"], "comparison_operator": "in", "value": "1,2,3,10"}],
        }
        filter_result = converter.convert_metadata_condition(condition_dict)
        assert filter_result is not None
        # 检查Filter的must条件
        assert len(filter_result.must) == 1
        field_condition = filter_result.must[0]
        assert field_condition.key == "category_id"
        # 数值应该被正确转换
        assert 1 in field_condition.match.any
        assert 2 in field_condition.match.any
        assert 3 in field_condition.match.any
        assert 10 in field_condition.match.any

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
        filter_result = converter.convert_metadata_condition(condition_dict)
        assert filter_result is not None
        # 应该有3个must条件
        assert len(filter_result.must) == 3

        # 检查每个条件
        conditions = {cond.key: cond for cond in filter_result.must}

        # 检查status in条件
        assert "status" in conditions
        assert hasattr(conditions["status"].match, "any")
        assert "active" in conditions["status"].match.any
        assert "published" in conditions["status"].match.any

        # 检查score ge条件
        assert "score" in conditions
        assert conditions["score"].range.gte == 85

        # 检查category ne条件
        assert "category" in conditions
        assert hasattr(conditions["category"].match, "except_")
        assert "spam" in getattr(conditions["category"].match, "except_")

    def test_or_logical_operator_new_ops(self, converter):
        """测试OR逻辑操作符与新操作符"""
        condition_dict = {
            "logical_operator": "or",
            "conditions": [
                {"name": ["priority"], "comparison_operator": "in", "value": "high,urgent"},
                {"name": ["score"], "comparison_operator": "gt", "value": "90"},
            ],
        }
        filter_result = converter.convert_metadata_condition(condition_dict)
        assert filter_result is not None
        # 应该有should条件而不是must
        assert hasattr(filter_result, "should")
        assert filter_result.should is not None
        assert len(filter_result.should) == 2

    def test_parse_value_list_comma_separated(self, converter):
        """测试解析逗号分隔的值列表"""
        values = converter._parse_value_list("apple,banana,cherry")
        assert values == ["apple", "banana", "cherry"]

    def test_parse_value_list_json_array(self, converter):
        """测试解析JSON数组格式的值列表"""
        values = converter._parse_value_list('["apple", "banana", "cherry"]')
        assert values == ["apple", "banana", "cherry"]

    def test_parse_value_list_single_value(self, converter):
        """测试解析单个值"""
        values = converter._parse_value_list("single_value")
        assert values == ["single_value"]

    def test_parse_value_list_empty(self, converter):
        """测试解析空值"""
        values = converter._parse_value_list("")
        assert values == []

    def test_parse_value_list_with_quotes(self, converter):
        """测试解析带引号的值"""
        values = converter._parse_value_list('"apple", "banana", "cherry"')
        assert values == ["apple", "banana", "cherry"]

    def test_in_operator_empty_value(self, converter):
        """测试in操作符空值"""
        condition_dict = {
            "logical_operator": "and",
            "conditions": [{"name": ["category"], "comparison_operator": "in", "value": ""}],
        }
        filter_result = converter.convert_metadata_condition(condition_dict)
        # 应该返回None或空Filter，因为没有有效条件
        assert filter_result is None or (hasattr(filter_result, "must") and len(filter_result.must) == 0)

    def test_numeric_operators_with_text_values(self, converter):
        """测试数值操作符使用文本值（应该失败）"""
        condition_dict = {
            "logical_operator": "and",
            "conditions": [{"name": ["score"], "comparison_operator": "gt", "value": "not_a_number"}],
        }
        filter_result = converter.convert_metadata_condition(condition_dict)
        # 应该返回None，因为条件无效
        assert filter_result is None or (hasattr(filter_result, "must") and len(filter_result.must) == 0)
