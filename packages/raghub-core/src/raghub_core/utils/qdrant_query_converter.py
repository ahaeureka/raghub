from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

from loguru import logger

if TYPE_CHECKING:
    from qdrant_client.models import FieldCondition, Filter, IsEmptyCondition, IsNullCondition


class QdrantQueryConverter:
    """将MetadataCondition转换为Qdrant查询过滤器的转换器"""

    def __init__(self):
        """初始化转换器"""
        try:
            from qdrant_client.models import (
                FieldCondition,
                Filter,
                GeoBoundingBox,
                GeoRadius,
                HasIdCondition,
                IsEmptyCondition,
                IsNullCondition,
                MatchAny,
                MatchExcept,
                MatchText,
                MatchValue,
                NestedCondition,
                PayloadField,
                Range,
            )

            self._models = {
                "Filter": Filter,
                "FieldCondition": FieldCondition,
                "MatchValue": MatchValue,
                "MatchText": MatchText,
                "MatchAny": MatchAny,
                "MatchExcept": MatchExcept,
                "Range": Range,
                "GeoBoundingBox": GeoBoundingBox,
                "GeoRadius": GeoRadius,
                "IsEmptyCondition": IsEmptyCondition,
                "IsNullCondition": IsNullCondition,
                "HasIdCondition": HasIdCondition,
                "NestedCondition": NestedCondition,
                "PayloadField": PayloadField,
            }
        except ImportError:
            raise ImportError("Qdrant client is not installed. Please install it using `pip install qdrant-client`.")

    def convert_metadata_condition(self, metadata_condition: dict) -> Optional["Filter"]:
        """
        将MetadataCondition转换为Qdrant Filter对象

        Args:
            metadata_condition: 包含logical_operator和conditions的字典

        Returns:
            Qdrant Filter对象，如果无有效条件则返回None
        """
        if not metadata_condition or not metadata_condition.get("conditions"):
            return None

        logical_operator = metadata_condition.get("logical_operator", "and").lower()
        conditions = metadata_condition.get("conditions", [])

        if not conditions:
            return None

        # 转换每个条件
        qdrant_conditions = []
        for condition in conditions:
            qdrant_condition = self._convert_single_condition(condition)
            if qdrant_condition:
                qdrant_conditions.append(qdrant_condition)

        if not qdrant_conditions:
            return None

        # 根据逻辑操作符构建Filter
        Filter = self._models["Filter"]

        if logical_operator == "and":
            return Filter(must=qdrant_conditions)
        elif logical_operator == "or":
            return Filter(should=qdrant_conditions)
        else:
            logger.warning(f"Unknown logical operator: {logical_operator}, using 'and' as default")
            return Filter(must=qdrant_conditions)

    def _convert_single_condition(
        self, condition: dict
    ) -> Optional[Union["FieldCondition", "IsEmptyCondition", "IsNullCondition"]]:
        """
        转换单个MetadataConditionItem为Qdrant条件

        Args:
            condition: 包含name, comparison_operator, value的字典

        Returns:
            Qdrant条件对象
        """
        names = condition.get("name", [])
        comparison_operator = condition.get("comparison_operator", "")
        value = condition.get("value", "")

        if not names or not comparison_operator:
            logger.warning(f"Invalid condition: {condition}")
            return None

        # 处理多个字段名（通常只有一个）
        if len(names) == 1:
            field_name = names[0]
            return self._create_field_condition(field_name, comparison_operator, value)
        else:
            # 如果有多个字段名，创建OR条件
            conditions = []
            for field_name in names:
                field_condition = self._create_field_condition(field_name, comparison_operator, value)
                if field_condition:
                    conditions.append(field_condition)

            if not conditions:
                return None
            elif len(conditions) == 1:
                return conditions[0]
            else:
                # 返回OR条件组合
                Filter = self._models["Filter"]
                return Filter(should=conditions)

    def _create_field_condition(
        self, field_name: str, operator: str, value: str
    ) -> Optional[Union["FieldCondition", "IsEmptyCondition", "IsNullCondition"]]:
        """
        为单个字段创建Qdrant条件

        Args:
            field_name: 字段名
            operator: 比较操作符
            value: 比较值

        Returns:
            Qdrant条件对象
        """
        FieldCondition = self._models["FieldCondition"]
        MatchValue = self._models["MatchValue"]
        MatchText = self._models["MatchText"]
        MatchExcept = self._models["MatchExcept"]
        Range = self._models["Range"]
        IsEmptyCondition = self._models["IsEmptyCondition"]
        IsNullCondition = self._models["IsNullCondition"]

        operator = operator.lower().strip()

        try:
            # 处理不需要值的操作符
            if operator in ["empty", "is empty"]:
                PayloadField = self._models["PayloadField"]
                return IsEmptyCondition(is_empty=PayloadField(key=field_name))
            elif operator in ["not empty", "is not empty"]:
                PayloadField = self._models["PayloadField"]
                return IsEmptyCondition(is_empty=PayloadField(key=field_name))  # 需要在外层用must_not包装
            elif operator in ["null", "is null"]:
                PayloadField = self._models["PayloadField"]
                return IsNullCondition(is_null=PayloadField(key=field_name))
            elif operator in ["not null", "is not null"]:
                PayloadField = self._models["PayloadField"]
                return IsNullCondition(is_null=PayloadField(key=field_name))  # 需要在外层用must_not包装

            # 处理需要值的操作符
            if not value and operator not in ["empty", "not empty", "null", "not null"]:
                logger.warning(f"Value required for operator '{operator}' but not provided")
                return None

            # 尝试转换值的类型
            typed_value = self._convert_value_type(value)

            if operator in ["contains"]:
                return FieldCondition(key=field_name, match=MatchText(text=value))
            elif operator in ["not contains"]:
                # 使用MatchExcept或在外层包装must_not
                return FieldCondition(key=field_name, match=MatchText(text=value))  # 需要must_not包装
            elif operator in ["start with", "starts with"]:
                return FieldCondition(key=field_name, match=MatchText(text=f"{value}*"))
            elif operator in ["end with", "ends with"]:
                return FieldCondition(key=field_name, match=MatchText(text=f"*{value}"))
            elif operator in ["is", "=", "equals", "eq"]:
                return FieldCondition(key=field_name, match=MatchValue(value=typed_value))
            elif operator in ["is not", "≠", "!=", "not equals", "ne"]:
                value_list = [typed_value] if not isinstance(typed_value, list) else typed_value

                return FieldCondition(key=field_name, match=MatchExcept(**{"except": value_list}))
            elif operator in [">", "greater than", "gt"]:
                if isinstance(typed_value, (int, float)):
                    return FieldCondition(key=field_name, range=Range(gt=typed_value))
                else:
                    logger.warning(f"Range operator '>' requires numeric value, got: {typed_value}")
                    return None
            elif operator in ["<", "less than", "lt"]:
                if isinstance(typed_value, (int, float)):
                    return FieldCondition(key=field_name, range=Range(lt=typed_value))
                else:
                    logger.warning(f"Range operator '<' requires numeric value, got: {typed_value}")
                    return None
            elif operator in [">=", "≥", "greater than or equal to", "ge"]:
                if isinstance(typed_value, (int, float)):
                    return FieldCondition(key=field_name, range=Range(gte=typed_value))
                else:
                    logger.warning(f"Range operator '>=' requires numeric value, got: {typed_value}")
                    return None
            elif operator in ["<=", "≤", "less than or equal to", "le"]:
                if isinstance(typed_value, (int, float)):
                    return FieldCondition(key=field_name, range=Range(lte=typed_value))
                else:
                    logger.warning(f"Range operator '<=' requires numeric value, got: {typed_value}")
                    return None
            elif operator in ["in"]:
                # 处理IN操作符
                values = self._parse_value_list(value)
                if not values:
                    logger.warning(f"IN operator requires at least one value, got: {value}")
                    return None

                # 转换值类型
                converted_values = [self._convert_value_type(v) for v in values]
                MatchAny = self._models["MatchAny"]
                return FieldCondition(key=field_name, match=MatchAny(any=converted_values))
            elif operator in ["notin", "not in"]:
                # 处理NOT IN操作符
                values = self._parse_value_list(value)
                if not values:
                    logger.warning(f"NOT IN operator requires at least one value, got: {value}")
                    return None

                # 转换值类型
                converted_values = [self._convert_value_type(v) for v in values]
                return FieldCondition(key=field_name, match=MatchExcept(**{"except": converted_values}))
            elif operator in ["before"]:
                # 对于日期比较，尝试转换为时间戳进行数值比较
                if self._is_numeric_value(value):
                    return FieldCondition(key=field_name, range=Range(lt=typed_value))
                else:
                    # 尝试转换日期字符串为时间戳
                    timestamp = self._convert_date_to_timestamp(value)
                    if timestamp is not None:
                        return FieldCondition(key=field_name, range=Range(lt=timestamp))
                    else:
                        # 如果转换失败，使用字符串比较
                        logger.warning(f"Cannot convert '{value}' to timestamp for date comparison")
                        return FieldCondition(key=field_name, match=MatchValue(value=value))
            elif operator in ["after"]:
                # 对于日期比较，尝试转换为时间戳进行数值比较
                if self._is_numeric_value(value):
                    return FieldCondition(key=field_name, range=Range(gt=typed_value))
                else:
                    # 尝试转换日期字符串为时间戳
                    timestamp = self._convert_date_to_timestamp(value)
                    if timestamp is not None:
                        return FieldCondition(key=field_name, range=Range(gt=timestamp))
                    else:
                        # 如果转换失败，使用字符串比较
                        logger.warning(f"Cannot convert '{value}' to timestamp for date comparison")
                        return FieldCondition(key=field_name, match=MatchValue(value=value))
            else:
                logger.warning(f"Unsupported operator: {operator}")
                return None

        except Exception as e:
            import traceback

            traceback.print_exc()
            logger.error(f"Error creating field condition for {field_name} {operator} {value}: {e}")
            return None

    def _convert_date_to_timestamp(self, date_str: str) -> Optional[float]:
        """
        将日期字符串转换为时间戳

        Args:
            date_str: 日期字符串（支持多种格式）

        Returns:
            时间戳（float）或None（如果转换失败）
        """
        import datetime

        date_formats = [
            "%Y-%m-%d",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%SZ",
        ]

        for fmt in date_formats:
            try:
                dt = datetime.datetime.strptime(date_str, fmt)
                return dt.timestamp()
            except ValueError:
                continue

        return None

    def _is_numeric_value(self, value: str) -> bool:
        """
        检查值是否为数值类型（可以转换为float）

        Args:
            value: 要检查的值

        Returns:
            True如果是数值，False否则
        """
        if not isinstance(value, str):
            return isinstance(value, (int, float))

        try:
            float(value)
            return True
        except ValueError:
            return False

    def _convert_value_type(self, value: str) -> Union[str, int, float, bool]:
        """
        尝试将字符串值转换为合适的数据类型

        Args:
            value: 字符串值

        Returns:
            转换后的值
        """
        if not isinstance(value, str):
            return value

        # 尝试转换为布尔值
        if value.lower() in ["true", "false"]:
            return value.lower() == "true"

        # 尝试转换为整数
        try:
            if "." not in value:
                return int(value)
        except ValueError:
            pass

        # 尝试转换为浮点数
        try:
            return float(value)
        except ValueError:
            pass

        # 保持为字符串
        return value

    def build_filter_with_negation(self, condition: dict, negate: bool = False) -> Optional["Filter"]:
        """
        构建包含否定逻辑的Filter

        Args:
            condition: 条件字典
            negate: 是否需要否定

        Returns:
            Qdrant Filter对象
        """
        if not condition:
            return None

        Filter = self._models["Filter"]

        # 处理单个条件项
        if "comparison_operator" in condition:
            base_condition = self._convert_single_condition(condition)
            if not base_condition:
                return None

            operator = condition.get("comparison_operator", "").lower().strip()

            # 检查是否是否定操作符
            negation_operators = [
                "not contains",
                "is not",
                "!=",
                "≠",
                "not equals",
                "not empty",
                "is not empty",
                "not null",
                "is not null",
            ]
            if operator in negation_operators:
                # 这些操作符本身就是否定的，需要用must_not包装原始条件
                if operator in ["not contains"]:
                    # 重新创建原始条件
                    field_name = condition.get("name", [""])[0]
                    value = condition.get("value", "")
                    FieldCondition = self._models["FieldCondition"]
                    MatchText = self._models["MatchText"]
                    original_condition = FieldCondition(key=field_name, match=MatchText(text=value))
                    return Filter(must_not=[original_condition])
                elif operator in ["not empty", "is not empty"]:
                    field_name = condition.get("name", [""])[0]
                    IsEmptyCondition = self._models["IsEmptyCondition"]
                    PayloadField = self._models["PayloadField"]
                    original_condition = IsEmptyCondition(is_empty=PayloadField(key=field_name))
                    return Filter(must_not=[original_condition])
                elif operator in ["not null", "is not null"]:
                    field_name = condition.get("name", [""])[0]
                    IsNullCondition = self._models["IsNullCondition"]
                    PayloadField = self._models["PayloadField"]
                    original_condition = IsNullCondition(is_null=PayloadField(key=field_name))
                    return Filter(must_not=[original_condition])
                else:
                    return Filter(must=[base_condition])
            else:
                if negate:
                    return Filter(must_not=[base_condition])
                else:
                    return Filter(must=[base_condition])

        # 处理复合条件
        elif "conditions" in condition:
            base_filter = self.convert_metadata_condition(condition)
            if not base_filter:
                return None

            if negate:
                return Filter(must_not=[base_filter])
            else:
                return base_filter

        return None

    def _parse_value_list(self, value: str) -> list:
        """
        解析值列表（支持逗号分隔或JSON数组格式）

        Args:
            value: 值字符串，可以是逗号分隔的值或JSON数组

        Returns:
            解析后的值列表
        """
        if not value:
            return []

        # 去除首尾空格
        value = value.strip()

        # 尝试解析JSON数组格式
        if value.startswith("[") and value.endswith("]"):
            try:
                import json

                return json.loads(value)
            except (json.JSONDecodeError, ValueError):
                # 如果JSON解析失败，去掉方括号按逗号分隔处理
                value = value[1:-1]

        # 按逗号分隔处理
        if "," in value:
            return [v.strip().strip('"').strip("'") for v in value.split(",") if v.strip()]
        else:
            return [value.strip().strip('"').strip("'")]


def create_qdrant_filter(metadata_condition: dict) -> Optional["Filter"]:
    """
    便捷函数：创建Qdrant过滤器

    Args:
        metadata_condition: MetadataCondition字典

    Returns:
        Qdrant Filter对象或None
    """
    converter = QdrantQueryConverter()
    return converter.convert_metadata_condition(metadata_condition)


# 示例用法和测试
if __name__ == "__main__":
    # 示例1: 简单条件
    example_condition_1 = {
        "logical_operator": "and",
        "conditions": [
            {"name": ["category"], "comparison_operator": "is", "value": "programming"},
            {"name": ["tags"], "comparison_operator": "contains", "value": "python"},
        ],
    }

    # 示例2: 复杂条件
    example_condition_2 = {
        "logical_operator": "or",
        "conditions": [
            {"name": ["priority"], "comparison_operator": ">", "value": "5"},
            {"name": ["status"], "comparison_operator": "is not", "value": "archived"},
            {"name": ["created_date"], "comparison_operator": "after", "value": "2024-01-01"},
        ],
    }

    # 示例3: 空值检查
    example_condition_3 = {
        "logical_operator": "and",
        "conditions": [
            {"name": ["description"], "comparison_operator": "not empty", "value": ""},
            {"name": ["deleted_at"], "comparison_operator": "is null", "value": ""},
        ],
    }

    converter = QdrantQueryConverter()

    print("示例1 - 简单AND条件:")
    filter1 = converter.convert_metadata_condition(example_condition_1)
    print(f"Generated filter: {filter1}")

    print("\n示例2 - 复杂OR条件:")
    filter2 = converter.convert_metadata_condition(example_condition_2)
    print(f"Generated filter: {filter2}")

    print("\n示例3 - 空值检查:")
    filter3 = converter.convert_metadata_condition(example_condition_3)
    print(f"Generated filter: {filter3}")
