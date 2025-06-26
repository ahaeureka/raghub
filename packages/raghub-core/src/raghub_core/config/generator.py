import inspect
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union, get_args, get_origin

from pydantic import BaseModel


class TomlConfigGenerator:
    """
    Automatically generates TOML configuration files from Pydantic BaseModel classes.
    """

    @classmethod
    def generate_toml_config(cls, model_cls: Type[BaseModel], include_comments: bool = True) -> Tuple[str, str]:
        """
        Generates TOML and Markdown configuration files from a Pydantic BaseModel class.
        Args:
            model_cls: Pydantic BaseModel class to generate configuration for.
            include_comments: Whether to include comments in the generated configuration files.
        Returns:
            Tuple[str, str]: TOML content and Markdown content as strings.
        Raises:
            ValueError: If the provided class is not a subclass of BaseModel.
        """
        toml_lines = []
        markdown_lines = []
        if include_comments:
            toml_lines.extend(
                [
                    f"# {model_cls.__name__} Configuration File",
                    f"# Auto-generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    f"# Description: {model_cls.__doc__ or 'Configuration settings'}",
                    "",
                ]
            )
            markdown_lines.extend(
                [
                    f"**{model_cls.__name__} Configuration File Documentation**  ",
                    f"**Auto-generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**  ",
                    f"**Description: {model_cls.__doc__ or 'Configuration settings'}** ",
                    "",
                ]
            )

        # Recursively build the configuration structure
        config_structure = cls._build_config_structure(model_cls)
        # Generate TOML and Markdown content
        cls._generate_toml_content(toml_lines, config_structure, include_comments)
        cls._generate_markdown_content(markdown_lines, config_structure, include_comments)

        return "\n".join(toml_lines), "\n".join(markdown_lines)

    @classmethod
    def _build_config_structure(
        cls, model_cls: Type[BaseModel], path: str = "", visited: Optional[Set] = None
    ) -> Dict[str, Any]:
        """
        Builds the configuration structure from a Pydantic BaseModel class.
        Args:
            model_cls: Pydantic BaseModel class to build the configuration structure for.
            path: Current path in the configuration structure.
            visited: Set of already visited model classes to avoid infinite recursion.
        Returns:
            Dict[str, Any]: Configuration structure containing root fields and nested sections.
        Raises:
            ValueError: If the provided class is not a subclass of BaseModel.
        Raises:
            RecursionError: If there is a circular reference in the model structure.
        """
        if visited is None:
            visited = set()

        if model_cls in visited:
            return {}

        visited.add(model_cls)

        structure: Dict[str, Dict[str, Any]] = {
            "root_fields": {},  # Root level fields
            "sections": {},  # Nested sections
            "model_info": {"name": model_cls.__name__, "doc": model_cls.__doc__ or "", "path": path},
        }

        for field_name, field_info in model_cls.model_fields.items():
            field_path = f"{path}.{field_name}" if path else field_name
            field_type = field_info.annotation
            field_default = field_info.default if field_info.default is not ... else None
            # Check if the field is a nested model
            nested_model = cls._extract_nested_model(field_type)

            if nested_model:
                # Recursively build the structure for nested models
                nested_structure = cls._build_config_structure(nested_model, field_path, visited.copy())

                structure["sections"][field_name] = {
                    "field_info": {
                        "name": field_name,
                        "type": cls._get_type_string(field_type),
                        "description": field_info.json_schema_extra["description"] or "",
                        "required": cls._is_required(field_info),
                        "path": field_path,
                        "pydantic_field": field_info,
                    },
                    "nested_structure": nested_structure,
                }
            else:
                # Base types like str, int, float, bool, list, dict
                default_value = cls._get_default_value(field_default, field_type)
                structure["root_fields"][field_name] = {
                    "name": field_name,
                    "value": default_value,
                    "type": cls._get_type_string(field_type),
                    "description": field_info.json_schema_extra["description"] or "",
                    "required": cls._is_required(field_info),
                    "path": field_path,
                    "pydantic_field": field_info,
                }

        return structure

    @classmethod
    def _generate_toml_content(
        cls, toml_lines: List[str], structure: Dict[str, Any], include_comments: bool, section_path: str = ""
    ):
        """
        Generates TOML content from the configuration structure.

        Args:
            toml_lines: List to store TOML lines.
            structure: Configuration structure containing root fields and nested sections.
            include_comments: Whether to include comments in the generated TOML content.
            section_path: Current section path for nested sections.

        """
        # Root level fields
        if structure["root_fields"]:
            if include_comments and section_path:
                pass
                # toml_lines.extend([f"# {section_path.upper()} - Root Level Fields", ""])
            elif include_comments and not section_path:
                toml_lines.extend(["# Root Level Configuration", ""])

            for field_name, field_data in structure["root_fields"].items():
                if include_comments:
                    toml_lines.append(f"# {field_data['description']}")
                    # cls._add_field_comments(toml_lines, field_name, field_data)

                value = field_data["value"]
                if value is not None:
                    toml_lines.append(f"{field_name} = {cls._format_toml_value(value)}")
                else:
                    toml_lines.append(f"# {field_name} = # Required field, no default value")
                toml_lines.append("")

        # Handle nested sections
        for section_name, section_info in structure["sections"].items():
            current_section_path = f"{section_path}.{section_name}" if section_path else section_name

            if include_comments:
                # toml_lines.extend(
                #     [
                #         f"# {current_section_path.upper()} SECTION",
                #         f"# Type: {section_info['field_info']['type']}",
                #         f"# Required: {section_info['field_info']['required']}",
                #     ]
                # )
                if section_info["field_info"]["description"]:
                    toml_lines.append(f"# {section_info['field_info']['description']}")
                toml_lines.append("")

            # Add section header
            toml_section_header = f"[{current_section_path}]"
            toml_lines.append(toml_section_header)
            toml_lines.append("")

            # Recursively generate content for nested sections
            nested_structure = section_info["nested_structure"]
            cls._generate_toml_content(toml_lines, nested_structure, include_comments, current_section_path)

    @classmethod
    def _add_field_comments(cls, toml_lines: List[str], field_name: str, field_data: Dict[str, Any]):
        """
        Add comments for a field in the TOML configuration.
        """
        comments = [
            f"# {field_name}:",
            f"#   Type: {field_data['type']}",
            f"#   Required: {field_data['required']}",
            f"#   Path: {field_data['path']}",
        ]

        if field_data["description"]:
            comments.append(f"#   Description: {field_data['description']}")

        # Add additional metadata from Pydantic field info
        if "pydantic_field" in field_data:
            field_info = field_data["pydantic_field"]
            if hasattr(field_info, "json_schema_extra") and field_info.json_schema_extra:
                extra = field_info.json_schema_extra
                if "tags" in extra:
                    comments.append(f"#   Tags: {', '.join(extra['tags'])}")
                if "examples" in extra:
                    examples_str = ", ".join([str(ex) for ex in extra["examples"]])
                    comments.append(f"#   Examples: {examples_str}")
                if "constraints" in extra:
                    comments.append(f"#   Constraints: {extra['constraints']}")

        # Add default value comment
        if field_data["value"] is not None:
            comments.append(f"#   Default: {field_data['value']}")
        else:
            comments.append("#   Default: None (Required)")

        toml_lines.extend(comments)

    @classmethod
    def _extract_nested_model(cls, annotation) -> Type[BaseModel] | None:
        """
        Extracts a BaseModel type from an annotation, supporting complex nesting.

        Args:
            annotation: The type annotation to inspect.
        """
        # Return the annotation if it is a subclass of BaseModel
        if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
            return annotation

        # Handle Union, Optional, List, Dict, etc.
        origin = get_origin(annotation)
        args = get_args(annotation)

        if origin is not None and args:
            # 处理Optional[Model] (实际上是Union[Model, None])
            if origin is Union:
                for arg in args:
                    if arg is not type(None):  # 跳过None类型
                        nested_model = cls._extract_nested_model(arg)
                        if nested_model:
                            return nested_model

        return None

    @classmethod
    def _get_default_value(cls, default_value: Any, field_type: Type) -> Any:
        """获取字段的默认值"""
        if default_value is not None and default_value is not ...:
            return default_value

        # 根据类型提供示例默认值
        origin = get_origin(field_type)
        if origin is list or field_type is list:
            return []
        elif origin is dict or field_type is dict:
            return {}
        elif field_type is str:
            return ""
        elif field_type is int:
            return 0
        elif field_type is float:
            return 0.0
        elif field_type is bool:
            return False

        return None

    @classmethod
    def _is_required(cls, field_info: Any) -> bool:
        """检查字段是否为必需的"""
        origin = get_origin(field_info.annotation)
        args = get_args(field_info.annotation)
        if origin is Union:
            # 如果是Optional类型，检查是否包含None
            return not (len(args) == 2 and type(None) in args)
        return True

    @classmethod
    def _get_type_string(cls, field_type: Type) -> str:
        """获取类型的字符串表示"""
        origin = get_origin(field_type)
        args = get_args(field_type)

        if origin is Union:
            # 处理Optional类型
            if len(args) == 2 and type(None) in args:
                non_none_type = args[0] if args[1] is type(None) else args[1]
                return f"{cls._get_simple_type_name(non_none_type)}"
            else:
                types = [cls._get_simple_type_name(arg) for arg in args]
                return f"{'|'.join(types)}"
        elif origin is list:
            if args:
                return f"List[{cls._get_simple_type_name(args[0])}]"
            return "List"
        elif origin is dict:
            if len(args) >= 2:
                return f"Dict[{cls._get_simple_type_name(args[0])}, {cls._get_simple_type_name(args[1])}]"
            return "Dict"
        else:
            return cls._get_simple_type_name(field_type)

    @classmethod
    def _get_simple_type_name(cls, field_type: Type) -> str:
        """获取简单类型名称"""
        if hasattr(field_type, "__name__"):
            return field_type.__name__
        else:
            return str(field_type)

    @classmethod
    def _format_toml_value(cls, value: Any) -> str:
        """格式化TOML值"""
        if isinstance(value, str):
            return f'"{value}"'
        elif isinstance(value, bool):
            return str(value).lower()
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, list):
            if not value:
                return "[]"
            formatted_items = [cls._format_toml_value(item) for item in value]
            return f"[{', '.join(formatted_items)}]"
        elif isinstance(value, dict):
            # 对于字典，这里简化处理
            return "{}"
        else:
            return f'"{str(value)}"'

    @classmethod
    def _generate_markdown_content(
        cls, markdown_lines: List[str], structure: Dict[str, Any], include_comments: bool, section_path: str = ""
    ):
        """
        生成Markdown内容

        Args:
            markdown_lines: Markdown行列表
            structure: 配置结构
            include_comments: 是否包含注释
            section_path: 当前section路径
        """
        # 处理根级别字段
        if structure["root_fields"]:
            if not section_path:
                markdown_lines.extend(["# Root Level Configuration", ""])
            else:
                # markdown_lines.extend([f"**Configuration for {section_path}**", ""])
                pass

            # 添加表格头
            markdown_lines.extend(
                [
                    "| Configuration | Type | Required | Description | Default |",
                    "|---------------|------|----------|-------------|---------|",
                ]
            )

            # 添加表格内容
            for field_name, field_data in structure["root_fields"].items():
                row = cls._format_table_row(field_name, field_data)
                markdown_lines.append(row)

            markdown_lines.append("")

        # 处理嵌套sections
        for section_name, section_info in structure["sections"].items():
            current_section_path = f"{section_path}.{section_name}" if section_path else section_name

            # 添加section标题
            if section_path:
                markdown_lines.extend(
                    [f"## [{current_section_path}]", f"**Configuration for {current_section_path}**", ""]
                )
            else:
                markdown_lines.extend(
                    [f"## [{current_section_path}]", f"**Configuration for {current_section_path}**", ""]
                )

            # 递归处理嵌套结构
            nested_structure = section_info["nested_structure"]
            cls._generate_markdown_content(markdown_lines, nested_structure, include_comments, current_section_path)

    @classmethod
    def _format_markdown_value(cls, value: Any) -> str:
        """格式化Markdown值"""
        if isinstance(value, str):
            return f'"{value}"'
        elif isinstance(value, bool):
            return str(value).lower()
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, list):
            if not value:
                return "[]"
            formatted_items = [cls._format_markdown_value(item) for item in value]
            return f"[{', '.join(formatted_items)}]"
        elif isinstance(value, dict):
            return "{}"
        else:
            return f'"{str(value)}"'

    @classmethod
    def _format_table_row(cls, field_name: str, field_data: Dict[str, Any]) -> str:
        """格式化表格行"""
        required = "true" if field_data["required"] else "false"
        description = field_data["description"] or ""

        # 处理默认值
        if field_data["value"] is not None:
            default_value = cls._format_markdown_value(field_data["value"])
        else:
            default_value = "" if field_data["required"] else "null"

        # 转义管道符号
        description = description.replace("|", "\\|")
        default_value = str(default_value).replace("|", "\\|")

        return f"| {field_name} | {field_data['type']} | {required} | {description} | {default_value} |"
