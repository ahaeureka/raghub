import threading
from multiprocessing import Lock
from typing import Dict, Optional, Type

from deeprag_app.apps.common.app import BaseAPP


class Registry:
    _components: Dict[str, Type[BaseAPP]] = {}
    _thread_lock = threading.Lock()
    _process_lock = Lock()

    @classmethod
    def register(cls, **kwargs):
        def decorator(components_cls: Type[BaseAPP]):
            with cls._thread_lock, cls._process_lock:  # 合并为单个with语句
                cls._components[components_cls.name] = components_cls
            return components_cls

        return decorator

    @classmethod
    def get_component(cls, name) -> Optional[Type[BaseAPP]]:
        return cls._components.get(name)

    @classmethod
    def list_components_name(cls):
        return list(cls._components.keys())
