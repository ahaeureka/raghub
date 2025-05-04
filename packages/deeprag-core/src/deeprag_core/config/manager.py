import os
import tomllib
from pathlib import Path
from typing import Type, TypeVar

from deeprag_core.utils.file.project import ProjectHelper
from loguru import logger

from deeprag_core.config.base import BaseParameters, TBaseParameters

T = TypeVar("T", bound="BaseParameters")  # 定义一个类型变量，限定为 BaseParameters 或其子类


# 配置加载器
class ConfigLoader:
    @staticmethod
    def load(cls: Type[TBaseParameters], file_path: str) -> TBaseParameters:
        try:
            logger.info(f"Environment variables: {os.environ}")  # 添加日志记录以调试环境变量
            content = Path(file_path).read_text(encoding="utf-8")
            data = tomllib.loads(content)
            return cls(**data)
        except Exception as e:
            raise RuntimeError(f"Failed to load {file_path} config: {str(e)}")

    @staticmethod
    def default_config_paths() -> str:
        user_dir = Path.home()
        project_dir = ProjectHelper.get_project_root()
        paths = [
            os.path.join(user_dir, ".config", "deeprag", "config.toml"),
            os.path.join(os.getcwd(), "config.toml"),
            os.path.join(project_dir, ".config.toml"),
            os.path.join(project_dir, ".devcontainer", ".dev.toml"),
            os.path.join(project_dir, ".devcontainer", "dev.toml"),
        ]
        for path in paths:
            print(f"Checking config path: {path}")
            if os.path.exists(path):
                return path
        return os.path.join(project_dir, "config.toml")  # 默认路径
