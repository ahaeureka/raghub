import sys
from pathlib import Path

from loguru import logger


def init_logging(level="DEBUG", log_dir="logs"):
    """项目全局日志初始化，只需调用一次."""

    # 移除默认的stderr处理器
    logger.remove()

    # 自定义格式
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{file}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level> | <level>{extra}</level>"
    )

    # 控制台输出配置（带颜色）
    logger.add(
        sys.stderr,
        colorize=True,
        format=log_format,
        level=level,
        backtrace=True,
        diagnose=True,
    )
    if log_dir:
        # 文件输出配置（无颜色）
        logger.add(
            Path(log_dir) / "app_{time:YYYY-MM-DD}.log",
            rotation="00:00",  # 每天午夜轮转
            retention="30 days",
            compression="zip",
            format=log_format.replace("<green>", "")
            .replace("</green>", "")
            .replace("<level>", "")
            .replace("</level>", "")
            .replace("<cyan>", "")
            .replace("</cyan>", ""),
            level=level,
            enqueue=True,  # 多进程安全
        )

        # 单独的错误日志
        logger.add(
            Path(log_dir) / "error_{time:YYYY-MM-DD}.log",
            rotation="00:00",
            retention="7 days",
            level="ERROR",
            format=log_format,
            compression="zip",
        )
