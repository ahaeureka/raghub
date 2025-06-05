from abc import abstractmethod
from typing import Any

from raghub_core.utils.class_meta import SingletonRegisterMeta


class CacheStorage(metaclass=SingletonRegisterMeta):
    name = ""

    @abstractmethod
    async def init(self):
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    async def aset(self, key: str, value: Any, ttl=None):
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    async def aget(self, key: str) -> str:
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    async def adelete(self, key: str):
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    async def aclear(self):
        raise NotImplementedError("Subclasses should implement this method.")
