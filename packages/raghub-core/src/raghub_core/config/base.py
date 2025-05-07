import inspect
import os
import re
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from pydantic import BaseModel, Field, create_model

TBaseParameters = TypeVar("TBaseParameters", bound="BaseParameters")

ENV_VAR_PATTERN = re.compile(r"\$\{env:([^:}]+)(?::-([^}]+))?\}")


class BaseParameters(BaseModel):
    """
    Base class for configuration parameters with extended functionality.
    """

    def __init__(self, **data: Any):
        # 解析环境变量
        for key, value in data.items():
            if isinstance(value, str):
                data[key] = self.parse_env_vars(value)
        # 初始化父类
        super().__init__(**data)

    class Config:
        validate_assignment = True  # 启用赋值验证
        use_enum_values = True  # 使用枚举值而不是枚举对象

    @classmethod
    def from_dict(
        cls: Type[TBaseParameters],
        data: Dict[str, Any],
        ignore_extra_fields: bool = False,
    ) -> TBaseParameters:
        """
        Create an instance from a dictionary.

        Args:
            data: Dictionary containing values for the fields
            ignore_extra_fields: If True, ignore extra fields in the data

        Returns:
            Instance of the class populated from the dictionary
        """
        if ignore_extra_fields:
            # 创建临时模型允许额外字段
            TempModel = create_model("TempModel", __base__=cls, __config__=dict(extra="ignore"))
            return TempModel.model_validate_with_env(data)
        return cls.model_validate_with_env(data)

    def update_from(self, source: Union["BaseParameters", Dict[str, Any]]) -> bool:
        """
        Update fields from another instance or dictionary.

        Args:
            source: Source to update from (instance or dict)

        Returns:
            bool: True if any field was updated
        """
        updated = False

        if isinstance(source, dict):
            source_dict = source
        elif isinstance(source, BaseParameters):
            source_dict = source.model_dump()
        else:
            raise ValueError("Source must be a dict or BaseParameters instance")

        for field_name, field_info in self.model_fields.items():
            # 检查字段是否有 fixed 标签
            if getattr(field_info, "fixed", False):
                continue

            if field_name in source_dict:
                new_value = source_dict[field_name]
                current_value = getattr(self, field_name)

                if new_value is not None and new_value != current_value:
                    setattr(self, field_name, new_value)
                    updated = True

        return updated

    @classmethod
    def parse_env_vars(cls, value: Any) -> Any:
        """
        递归解析环境变量插值.
        """
        if isinstance(value, str):
            match = ENV_VAR_PATTERN.search(value)
            if match:
                var_name = match.group(1)
                default_value = match.group(2) if len(match.groups()) > 1 else None
                env_value = os.getenv(var_name, default_value)
                if env_value is None:
                    raise ValueError(f"Environment variable {var_name} is not set and no default provided")
                return env_value
            return value
        elif isinstance(value, dict):
            return {k: cls.parse_env_vars(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [cls.parse_env_vars(item) for item in value]
        return value

    @classmethod
    def model_validate_with_env(cls: Type[TBaseParameters], data: Dict[str, Any]) -> TBaseParameters:
        """
        解析环境变量后验证数据.
        """
        parsed_data = cls.parse_env_vars(data)
        return cls.model_validate(parsed_data)

    def __str__(self) -> str:
        return self._get_print_str()

    def _get_print_str(self) -> str:
        """Generate a pretty-printed string representation."""
        items = []
        for name, field in self.model_fields.items():
            value = getattr(self, name)
            items.append(f"{name}={value}")
        return f"{self.__class__.__name__}({', '.join(items)})"

    def to_command_args(self, args_prefix: str = "--") -> List[str]:
        """
        Convert to command line arguments.

        Args:
            args_prefix: Prefix for argument names

        Returns:
            List of command line arguments
        """
        args = []
        for name, value in self.model_dump().items():
            if value is None:
                continue
            arg_name = f"{args_prefix}{name.replace('_', '-')}"
            args.extend([arg_name, str(value)])
        return args

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()

    @classmethod
    def get_parameter_descriptions(cls) -> List[Dict[str, Any]]:
        """
        Get descriptions of all parameters.

        Returns:
            List of parameter descriptions with name, type, default and description
        """
        descriptions = []
        for name, field in cls.model_fields.items():
            desc = {
                "name": name,
                "type": field.type_.__name__ if hasattr(field, "type_") else str(field.annotation),
                "default": field.default if field.default != inspect.Parameter.empty else None,
                "description": getattr(field.json_schema_extra, "description", "") or "",
                "tags": getattr(field.json_schema_extra, "tags", []),
            }
            descriptions.append(desc)
        return descriptions

    @classmethod
    def field(
        cls,
        default: Any = None,
        *,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        fixed: bool = False,
        **kwargs,
    ) -> Any:
        """
        Custom field definition with additional metadata.

        Args:
            default: Default value
            description: Field description
            tags: List of tags for the field
            fixed: If True, field cannot be updated
            **kwargs: Additional field arguments

        Returns:
            FieldInfo instance
        """
        metadata = kwargs.pop("metadata", {})
        if description:
            metadata["description"] = description
        if tags:
            metadata["tags"] = tags
        if fixed:
            metadata["fixed"] = True

        return Field(default, json_schema_extra=metadata, **kwargs)
