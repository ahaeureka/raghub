import asyncio
import os
from typing import Any

from langchain_core.runnables.config import run_in_executor
from raghub_core.storage.cache import CacheStorage


class DiskCacheStorage(CacheStorage):
    name = "disk_cache"

    def __init__(self, cache_dir: str):
        super().__init__()
        import diskcache as dc

        self._cache_dir = cache_dir
        self._cache: dc.Cache = None

    async def init(self):
        os.makedirs(self._cache_dir, exist_ok=True)  # Ensure the cache directory exists
        import diskcache as dc

        await asyncio.sleep(0.01)  # Yield control to the event loop
        self._cache = dc.Cache(self._cache_dir)

    def set(self, key: str, value: Any, ttl=None):
        self._cache.set(key, value, expire=ttl)

    def get(self, key: str) -> str:
        return self._cache.get(key)

    def delete(self, key: str):
        self._cache.delete(key)

    def clear(self):
        self._cache.clear()

    async def aset(self, key: str, value: Any, ttl=None):
        await run_in_executor(None, self._cache.set, key, value, expire=ttl)

    async def aget(self, key: str) -> str:
        return await run_in_executor(None, self._cache.get, key)

    async def adelete(self, key: str):
        await run_in_executor(None, self._cache.delete, key)

    async def aclear(self):
        await run_in_executor(None, self._cache.clear)
