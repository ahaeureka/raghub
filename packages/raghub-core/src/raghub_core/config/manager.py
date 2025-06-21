import json
import os
import tomllib
from pathlib import Path
from typing import Type, TypeVar

from loguru import logger
from raghub_core.config.base import BaseParameters, TBaseParameters
from raghub_core.utils.file.project import ProjectHelper

T = TypeVar("T", bound="BaseParameters")  # 定义一个类型变量，限定为 BaseParameters 或其子类


# 配置加载器
class ConfigLoader:
    @staticmethod
    def load(cls: Type[TBaseParameters], file_path: str) -> TBaseParameters:
        try:
            content = Path(file_path).read_text(encoding="utf-8")
            data = tomllib.loads(content)
            logger.info(f"Loading config from {file_path}")
            if not isinstance(data, dict):
                raise ValueError(f"Config file {file_path} does not contain a valid dictionary.")
            logger.debug(f"Config data: {json.dumps(data, ensure_ascii=False, indent=4)}")
            return cls(**data)
        except Exception as e:
            raise RuntimeError(f"Failed to load {file_path} config: {str(e)}")

    @staticmethod
    def default_config_paths() -> str:
        user_dir = Path.home()
        project_dir = ProjectHelper.get_project_root()
        paths = [
            os.path.join(user_dir, ".config", "raghub", "config.toml"),
            os.path.join(os.getcwd(), "config.toml"),
            os.path.join(project_dir, ".config.toml"),
            os.path.join(project_dir, ".devcontainer", ".dev.toml"),
            os.path.join(project_dir, ".devcontainer", "dev.toml"),
        ]
        for path in paths:
            logger.info(f"Checking config path: {path}")
            if os.path.exists(path):
                return path
        logger.warning(
            f"No config file found at {json.dumps(paths, ensure_ascii=False, indent=4)}, using default path."
        )
        return os.path.join(project_dir, "config.toml")  # 默认路径
