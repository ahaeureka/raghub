from abc import abstractmethod
from typing import Any

from deeprag_core.utils.class_meta import SingletonRegisterMeta


class CacheStorage(metaclass=SingletonRegisterMeta):
    name = ""

    @abstractmethod
    def init(self):
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def set(self, key: str, value: Any, ttl=None):
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def get(self, key: str) -> str:
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def delete(self, key: str):
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def clear(self):
        raise NotImplementedError("Subclasses should implement this method.")
