import threading
from multiprocessing import Lock
from typing import Dict, Optional, Type

from raghub_interfaces.interfaces.interface import BaseInterface


class Registry:
    _components: Dict[str, Type[BaseInterface]] = {}
    _thread_lock = threading.Lock()
    _process_lock = Lock()

    @classmethod
    def register(cls, name: Optional[str] = None):
        def decorator(components_cls: Type[BaseInterface]):
            with cls._thread_lock, cls._process_lock:  # 合并为单个with语句
                if not issubclass(components_cls, BaseInterface):
                    raise TypeError(f"{components_cls.__name__} must be a subclass of BaseInterface")
                cls_name = name or components_cls.name
                if not cls_name:
                    raise ValueError(f"{components_cls.__name__} must have a name attribute")
                cls._components[cls_name] = components_cls
            return components_cls

        return decorator

    @classmethod
    def get_component(cls, name) -> Optional[Type[BaseInterface]]:
        return cls._components.get(name)

    @classmethod
    def list_components_name(cls):
        return list(cls._components.keys())
