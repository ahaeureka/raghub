# -*- encoding: utf-8 -*-
"""
@File    :   cli.py
@Time    :   2025/04/18 17:40:16
@Desc    :   CLI入口文件
"""

import asyncio
import inspect
from typing import Type

import typer
from pydantic_core import PydanticUndefinedType
from raghub_core.config.manager import ConfigLoader
from raghub_core.utils.logger.logger import init_logging
from typing_extensions import Annotated

from raghub_interfaces.config.interface_config import InerfaceConfig
from raghub_interfaces.interfaces.interface import BaseInterface
from raghub_interfaces.schemes.params import RunnerParams

cmder = typer.Typer()
start_cmder = typer.Typer()
cmder.add_typer(start_cmder, name="start")
app_config = None
use_async = False


def create_runner_cmd_option(component_cls: Type[BaseInterface], prams_class: Type[RunnerParams]):
    """
    Create a Typer command option for the given APPRunnerParams class
    """
    fields = []
    # 动态生成字段列表，将每个字段转换为 Typer Option 参数
    for field_name, field in prams_class.model_fields.items():
        default_value = None
        if not isinstance(field.default, PydanticUndefinedType) and field.default is not None:
            default_value = field.default
        fields.append((field_name, (Annotated[field.annotation, typer.Option(help=field.description)], default_value)))
    fields.append(
        (
            "config",
            (
                Annotated[
                    str,
                    typer.Option(
                        ConfigLoader.default_config_paths(),
                        "--config",
                        "-c",
                        help="Path to the application configuration file",
                    ),
                ],
                ConfigLoader.default_config_paths(),
            ),
        )
    )
    # fields.append(("log_level", (Annotated[str, typer.Option("DEBUG", "--log-level")], "DEBUG")))  # 添加 input 参数
    # fields.append(("log_dir", (Annotated[str, typer.Option("logs", "--log-dir")], "logs")))  # 添加 input 参数

    def run(**kwargs):
        """Callable behavior: print fields or execute custom logic"""
        kwargs.pop("self", None)
        p = prams_class(**kwargs)  # 解包参数并创建模型实例
        # p = input
        config_path = kwargs.get("config", "/app/.devcontainer/online.toml")  # 获取配置文件路径
        # with_async = kwargs.get("with_async", False)  # 获取是否使用异步模式
        app_config = ConfigLoader.load(InerfaceConfig, config_path)
        # log_level = kwargs.pop("log_level", "DEBUG")  # 获取日志级别
        # log_dir = kwargs.pop("log_dir", "logs")
        init_logging(level=app_config.logger.log_level, log_dir=app_config.logger.log_dir)
        # use_async = with_async
        component = component_cls(app_config)
        asyncio.run(component(p))

    new_params = [
        inspect.Parameter(
            name=name, kind=inspect.Parameter.POSITIONAL_OR_KEYWORD, default=value[1], annotation=value[0]
        )
        for name, value in fields
    ]
    run.__signature__ = inspect.Signature(new_params)  # type: ignore[attr-defined]
    run.__annotations__ = {name: value[0] for name, value in fields}
    return run


def build_app_cmd():
    """
    Build app command dynamically based on registered components
    """
    # Import components here to register them,like:
    from raghub_app import apps  # noqa: F401
    from raghub_ext import storage_ext  # noqa: F401

    from raghub_interfaces import (
        interfaces,  # noqa: F401
        services,  # noqa: F401
    )
    from raghub_interfaces.registry.register import Registry

    components = Registry.list_components_name()
    typer.echo(f"Building app from {components} command...")
    for component_name in components:
        component = Registry.get_component(component_name)
        run_method = component.__call__
        sig = inspect.signature(run_method)

        params_class = next(
            (
                p.annotation
                for p in sig.parameters.values()
                if inspect.isclass(p.annotation) and issubclass(p.annotation, RunnerParams)
            ),
            None,
        )
        if not params_class:
            typer.echo(f"No RunnerParams subclass found for component {component_name}")
            continue
            # 添加参数

        # 注册命令
        typer.echo(f"Registering command: {component_name}")
        start_cmder.command(name=component_name, help=component.description)(
            create_runner_cmd_option(component, params_class)
        )


def main():
    build_app_cmd()
    cmder()


if __name__ == "__main__":
    main()
