"""
Elasticsearch查询转换器

将MetadataCondition转换为Elasticsearch查询语句
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from loguru import logger


class ESQueryConverter:
    """将MetadataCondition转换为Elasticsearch查询的转换器"""

    def __init__(self):
        """初始化转换器"""
        self.operator_mapping = {
            # 文本操作符
            "contains": self._create_contains_query,
            "not contains": self._create_not_contains_query,
            "start with": self._create_starts_with_query,
            "end with": self._create_ends_with_query,
            "is": self._create_equals_query,
            "is not": self._create_not_equals_query,
            # 存在性操作符
            "empty": self._create_empty_query,
            "not empty": self._create_not_empty_query,
            # 比较操作符
            "=": self._create_equals_query,
            "≠": self._create_not_equals_query,
            ">": self._create_greater_than_query,
            "<": self._create_less_than_query,
            "≥": self._create_greater_equal_query,
            "≤": self._create_less_equal_query,
            # 新增操作符
            "eq": self._create_equals_query,
            "ne": self._create_not_equals_query,
            "lt": self._create_less_than_query,
            "gt": self._create_greater_than_query,
            "le": self._create_less_equal_query,
            "ge": self._create_greater_equal_query,
            "in": self._create_in_query,
            "notin": self._create_not_in_query,
            # 日期操作符
            "before": self._create_before_query,
            "after": self._create_after_query,
        }

    def convert_metadata_condition(self, condition_dict: Dict[str, Any], field_prefix: str = "") -> Dict[str, Any]:
        """
        将MetadataCondition字典转换为Elasticsearch查询

        Args:
            condition_dict: MetadataCondition的字典形式
            field_prefix: 字段名前缀，例如 "metadata." 用于Elasticsearch

        Returns:
            Elasticsearch查询字典
        """
        if not condition_dict or "conditions" not in condition_dict:
            return {"match_all": {}}

        logical_operator = condition_dict.get("logical_operator", "and").lower()
        conditions = condition_dict.get("conditions", [])

        if not conditions:
            return {"match_all": {}}

        # 转换所有条件
        converted_conditions = []
        for condition in conditions:
            converted_condition = self._convert_single_condition(condition, field_prefix)
            if converted_condition:
                converted_conditions.append(converted_condition)

        if not converted_conditions:
            return {"match_all": {}}

        # 根据逻辑操作符组合条件
        if len(converted_conditions) == 1:
            return converted_conditions[0]

        if logical_operator == "and":
            return {"bool": {"must": converted_conditions}}
        elif logical_operator == "or":
            return {"bool": {"should": converted_conditions, "minimum_should_match": 1}}
        else:
            logger.warning(f"Unknown logical operator: {logical_operator}, defaulting to 'and'")
            return {"bool": {"must": converted_conditions}}

    def _convert_single_condition(self, condition: Dict[str, Any], field_prefix: str = "") -> Optional[Dict[str, Any]]:
        """
        转换单个条件

        Args:
            condition: MetadataConditionItem的字典形式
            field_prefix: 字段名前缀

        Returns:
            Elasticsearch查询条件
        """
        try:
            field_names = condition.get("name", [])
            operator = condition.get("comparison_operator", "")
            value = condition.get("value", "")

            if not field_names or not operator:
                logger.warning("Invalid condition: missing field names or operator")
                return None

            # 获取操作符对应的转换函数
            converter_func = self.operator_mapping.get(operator)
            if not converter_func:
                logger.warning(f"Unsupported operator: {operator}")
                return None

            # 如果有多个字段名，使用OR逻辑组合
            if len(field_names) == 1:
                field_name = self._build_field_name(field_names[0], field_prefix)
                return converter_func(field_name, value)
            else:
                field_conditions = []
                for field_name in field_names:
                    full_field_name = self._build_field_name(field_name, field_prefix)
                    field_condition = converter_func(full_field_name, value)
                    if field_condition:
                        field_conditions.append(field_condition)

                if field_conditions:
                    return {"bool": {"should": field_conditions, "minimum_should_match": 1}}

        except Exception as e:
            logger.error(f"Error converting condition: {e}")

        return None

    def _build_field_name(self, field_name: str, field_prefix: str = "") -> str:
        """
        构建完整的字段名

        Args:
            field_name: 原始字段名
            field_prefix: 字段前缀

        Returns:
            完整的字段名
        """
        if field_prefix:
            return f"{field_prefix}{field_name}"
        return field_name

    def _create_contains_query(self, field_name: str, value: str) -> Dict[str, Any]:
        """创建包含查询"""
        return {"wildcard": {f"{field_name}.keyword": f"*{value}*"}}

    def _create_not_contains_query(self, field_name: str, value: str) -> Dict[str, Any]:
        """创建不包含查询"""
        return {"bool": {"must_not": [{"wildcard": {f"{field_name}.keyword": f"*{value}*"}}]}}

    def _create_starts_with_query(self, field_name: str, value: str) -> Dict[str, Any]:
        """创建以...开头查询"""
        return {"prefix": {f"{field_name}.keyword": value}}

    def _create_ends_with_query(self, field_name: str, value: str) -> Dict[str, Any]:
        """创建以...结尾查询"""
        return {"wildcard": {f"{field_name}.keyword": f"*{value}"}}

    def _create_equals_query(self, field_name: str, value: str) -> Dict[str, Any]:
        """创建等于查询"""
        # 尝试转换为数字
        if self._is_numeric(value):
            return {"term": {field_name: self._to_number(value)}}
        else:
            return {"term": {f"{field_name}.keyword": value}}

    def _create_not_equals_query(self, field_name: str, value: str) -> Dict[str, Any]:
        """创建不等于查询"""
        if self._is_numeric(value):
            return {"bool": {"must_not": [{"term": {field_name: self._to_number(value)}}]}}
        else:
            return {"bool": {"must_not": [{"term": {f"{field_name}.keyword": value}}]}}

    def _create_greater_than_query(self, field_name: str, value: str) -> Dict[str, Any]:
        """创建大于查询"""
        if self._is_numeric(value):
            return {"range": {field_name: {"gt": self._to_number(value)}}}
        elif self._is_date(value):
            return {"range": {field_name: {"gt": value}}}
        else:
            return {"range": {f"{field_name}.keyword": {"gt": value}}}

    def _create_less_than_query(self, field_name: str, value: str) -> Dict[str, Any]:
        """创建小于查询"""
        if self._is_numeric(value):
            return {"range": {field_name: {"lt": self._to_number(value)}}}
        elif self._is_date(value):
            return {"range": {field_name: {"lt": value}}}
        else:
            return {"range": {f"{field_name}.keyword": {"lt": value}}}

    def _create_greater_equal_query(self, field_name: str, value: str) -> Dict[str, Any]:
        """创建大于等于查询"""
        if self._is_numeric(value):
            return {"range": {field_name: {"gte": self._to_number(value)}}}
        elif self._is_date(value):
            return {"range": {field_name: {"gte": value}}}
        else:
            return {"range": {f"{field_name}.keyword": {"gte": value}}}

    def _create_less_equal_query(self, field_name: str, value: str) -> Dict[str, Any]:
        """创建小于等于查询"""
        if self._is_numeric(value):
            return {"range": {field_name: {"lte": self._to_number(value)}}}
        elif self._is_date(value):
            return {"range": {field_name: {"lte": value}}}
        else:
            return {"range": {f"{field_name}.keyword": {"lte": value}}}

    def _create_empty_query(self, field_name: str, value: str = "") -> Dict[str, Any]:
        """创建为空查询"""
        return {
            "bool": {
                "should": [
                    {"bool": {"must_not": [{"exists": {"field": field_name}}]}},
                    {"term": {f"{field_name}.keyword": ""}},
                ],
                "minimum_should_match": 1,
            }
        }

    def _create_not_empty_query(self, field_name: str, value: str = "") -> Dict[str, Any]:
        """创建非空查询"""
        return {
            "bool": {"must": [{"exists": {"field": field_name}}], "must_not": [{"term": {f"{field_name}.keyword": ""}}]}
        }

    def _create_before_query(self, field_name: str, value: str) -> Dict[str, Any]:
        """创建早于日期查询"""
        return {"range": {field_name: {"lt": value}}}

    def _create_after_query(self, field_name: str, value: str) -> Dict[str, Any]:
        """创建晚于日期查询"""
        return {"range": {field_name: {"gt": value}}}

    def _create_in_query(self, field_name: str, value: str) -> Dict[str, Any]:
        """创建IN查询（包含在值列表中）"""
        # 支持逗号分隔的值列表
        if "," in value:
            values = [v.strip() for v in value.split(",")]
        else:
            values = [value]

        # 尝试转换数值类型
        converted_values: List[str | float | str] = []
        for v in values:
            if self._is_numeric(v):
                converted_values.append(self._to_number(v))
            else:
                converted_values.append(v)

        # 如果所有值都是数值类型，使用数值字段；否则使用keyword字段
        if all(isinstance(v, (int, float)) for v in converted_values):
            return {"terms": {field_name: converted_values}}
        else:
            # 对于文本值，使用keyword字段
            return {"terms": {f"{field_name}.keyword": [str(v) for v in converted_values]}}

    def _create_not_in_query(self, field_name: str, value: str) -> Dict[str, Any]:
        """创建NOT IN查询（不包含在值列表中）"""
        # 复用in查询逻辑，然后用must_not包装
        in_query = self._create_in_query(field_name, value)

        return {"bool": {"must_not": [in_query]}}

    def _is_numeric(self, value: str) -> bool:
        """判断字符串是否为数字"""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False

    def _to_number(self, value: str) -> Union[int, float]:
        """将字符串转换为数字"""
        try:
            if "." in value:
                return float(value)
            else:
                return int(value)
        except (ValueError, TypeError):
            return 0

    def _is_date(self, value: str) -> bool:
        """判断字符串是否为日期格式"""
        date_formats = [
            "%Y-%m-%d",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y/%m/%d",
            "%d/%m/%Y",
        ]

        for fmt in date_formats:
            try:
                datetime.strptime(value, fmt)
                return True
            except ValueError:
                continue

        return False

    def build_es_query(
        self,
        metadata_condition: Optional[Dict[str, Any]] = None,
        query_text: Optional[str] = None,
        vector_query: Optional[Dict[str, Any]] = None,
        size: int = 10,
        from_: int = 0,
    ) -> Dict[str, Any]:
        """
        构建完整的Elasticsearch查询

        Args:
            metadata_condition: 元数据过滤条件
            query_text: 文本查询
            vector_query: 向量查询（如果支持）
            size: 返回结果数量
            from_: 查询起始位置

        Returns:
            完整的Elasticsearch查询字典
        """
        query_parts = []

        # 添加文本查询
        if query_text:
            query_parts.append(
                {"multi_match": {"query": query_text, "fields": ["content^2", "title^1.5", "metadata.*"]}}
            )

        # 添加元数据过滤
        if metadata_condition:
            metadata_query = self.convert_metadata_condition(metadata_condition)
            if metadata_query and metadata_query != {"match_all": {}}:
                query_parts.append(metadata_query)

        # 添加向量查询（如果支持）
        if vector_query:
            query_parts.append(vector_query)

        # 构建最终查询
        if not query_parts:
            es_query: Dict[str, Any] = {"match_all": {}}
        elif len(query_parts) == 1:
            es_query = query_parts[0]
        else:
            es_query = {"bool": {"must": query_parts}}

        return {
            "query": es_query,
            "size": size,
            "from": from_,
            "_source": {
                "excludes": ["vector"]  # 排除向量字段以减少响应大小
            },
        }
