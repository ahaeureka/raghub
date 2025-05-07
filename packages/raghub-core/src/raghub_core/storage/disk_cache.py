import os
from typing import Any

from raghub_core.storage.cache import CacheStorage


class DiskCacheStorage(CacheStorage):
    name = "disk_cache"

    def __init__(self, cache_dir: str):
        super().__init__()
        import diskcache as dc

        self._cache_dir = cache_dir
        self._cache: dc.Cache = None

    def init(self):
        os.makedirs(self._cache_dir, exist_ok=True)  # Ensure the cache directory exists
        import diskcache as dc

        self._cache = dc.Cache(self._cache_dir)

    def set(self, key: str, value: Any, ttl=None):
        self._cache.set(key, value, expire=ttl)

    def get(self, key: str) -> str:
        return self._cache.get(key)

    def delete(self, key: str):
        self._cache.delete(key)

    def clear(self):
        self._cache.clear()
