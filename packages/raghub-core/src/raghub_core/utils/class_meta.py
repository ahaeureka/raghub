import inspect
import multiprocessing
import multiprocessing.synchronize
from abc import ABC, ABCMeta
from typing import Any, Dict, Type, TypeVar

from loguru import logger

T = TypeVar("T")


class SingletonMeta(type):
    _instances: Dict[Type, Any] = {}
    _locks: Dict[str, multiprocessing.synchronize.Lock] = {}  # 单一锁，适用于线程和进程

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            lock = cls._locks.get(cls.__name__)
            if lock is None:
                lock = multiprocessing.Lock()
                cls._locks[cls.__name__] = lock
            with lock:
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        return cls._instances[cls]


class RegsiterMeta(type):
    _registry: Dict[str, Type] = {}

    def __init__(cls, name, bases, dct):
        super().__init__(name, bases, dct)
        if not hasattr(cls, "_registry"):
            cls._registry = {}  # 基类初始化注册表
        elif "name" in dct and cls.name:
            cls._registry[cls.name] = cls  # 子类注册

    @classmethod
    def registry(cls) -> Dict[str, Type]:
        """
        获取注册表
        :return: 注册表字典
        """
        return cls._registry


class SingletonRegisterMeta(SingletonMeta, RegsiterMeta, ABCMeta):
    pass


class RegisterABCMeta(RegsiterMeta, ABCMeta):
    pass


class ClassFactory:
    @staticmethod
    def get_instance(provider_name: str, type_cls: Type[T] | Type[ABC], *args, **kwargs) -> T:
        cls = SingletonRegisterMeta._registry.get(provider_name)
        if not cls:
            raise ValueError(f"Unsupported provider: {provider_name} for {type_cls.__name__}")
        if not issubclass(cls, type_cls):
            raise ValueError(f"Provider {provider_name} is not a subclass of {type_cls.__name__}")
        # 获取构造函数签名
        try:
            signature = inspect.signature(cls.__init__)
        except ValueError:
            # 如果 __init__ 不存在（例如抽象类），使用 object.__init__ 作为默认
            signature = inspect.signature(object.__init__)
            logger.warning(f"Using object.__init__ for {cls.__name__} as it has no __init__ method.")

        # 过滤 kwargs：只保留构造函数中声明的参数（排除 self）
        filtered_kwargs = {
            name: value for name, value in kwargs.items() if name in signature.parameters and name != "self"
        }
        logger.debug(f"Filtered kwargs for {cls.__name__}: {filtered_kwargs}: {kwargs}")
        return cls(*args, **filtered_kwargs)
