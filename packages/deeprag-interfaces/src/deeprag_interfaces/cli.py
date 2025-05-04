# -*- encoding: utf-8 -*-
"""
@File    :   cli.py
@Time    :   2025/04/18 17:40:16
@Desc    :   CLI入口文件
"""

import inspect
from typing import Type

import trio_asyncio
import typer
from deeprag_app.app_schemas.app import APPRunnerParams
from deeprag_app.apps.common.app import BaseAPP
from deeprag_app.apps.registry.register import Registry
from deeprag_app.config.config_models import APPConfig
from deeprag_core.utils.configure.manager import ConfigLoader
from deeprag_core.utils.logger.logger import init_logging
from pydantic_core import PydanticUndefinedType
from typing_extensions import Annotated

cmder = typer.Typer()
start_cmder = typer.Typer()
cmder.add_typer(start_cmder, name="start")
app_config = None
use_async = False


@start_cmder.command("webserver")
def start_httpserver():
    """Start http server"""
    typer.echo("Starting HTTP server...")


def create_runner_cmd_option(component_cls: Type[BaseAPP], prams_class: Type[APPRunnerParams]):
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
    fields.append(("with_async", (Annotated[bool, typer.Option(help="Run the application with async mode")], False)))
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
        config_path = kwargs.get("config", "/app/.devcontainer/dev.toml")  # 获取配置文件路径
        with_async = kwargs.get("with_async", False)  # 获取是否使用异步模式
        app_config = ConfigLoader.load(APPConfig, config_path)
        # log_level = kwargs.pop("log_level", "DEBUG")  # 获取日志级别
        # log_dir = kwargs.pop("log_dir", "logs")
        init_logging(level=app_config.logger.log_level, log_dir=app_config.logger.log_dir)
        use_async = with_async
        component = component_cls(app_config)
        component.initialization()
        if not use_async:
            return component.run(p)
        return trio_asyncio.run(component.arun, p)

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
    # from template_app.apps.template_app import TemplateApp  # noqa: F401

    components = Registry.list_components_name()
    typer.echo(f"Building app from {components} command...")
    for component_name in components:
        component = Registry.get_component(component_name)
        run_method = component.run
        sig = inspect.signature(run_method)

        params_class = next(
            (
                p.annotation
                for p in sig.parameters.values()
                if inspect.isclass(p.annotation) and issubclass(p.annotation, APPRunnerParams)
            ),
            None,
        )
        if not params_class:
            typer.echo(f"No APPRunnerParams subclass found for component {component_name}")
            continue
            # 添加参数

        # 注册命令
        typer.echo(f"Registering command: {component_name}")
        start_cmder.command(name=component_name, help=component.description)(
            create_runner_cmd_option(component, params_class)
        )


if __name__ == "__main__":
    # 构建动态命令
    print(f"start {create_runner_cmd_option.__annotations__}, {create_runner_cmd_option.__dict__}")

    build_app_cmd()
    # print(f"start {start_server.__annotations__}")
    cmder()
