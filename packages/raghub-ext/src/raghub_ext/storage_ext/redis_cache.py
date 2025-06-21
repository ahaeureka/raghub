import pickle
from typing import Any, Optional

from raghub_core.storage.cache import CacheStorage


class RedisCacheStorage(CacheStorage):
    name = "redis_cache"

    def __init__(self, host="localhost", port=6379, db=0, auth: Optional[str] = None):
        super().__init__()
        self._host = host
        self._port = port
        self._db = db
        self._auth = auth
        try:
            import redis as sync_redis
            import redis.asyncio as redis
        except ImportError:
            raise ImportError("Please install redis-py with pip install redis")
        self._redis: Optional[redis.Redis] = None  # Redis 客户端实例
        self._sync_client: Optional[sync_redis.Redis] = None  # 同步 Redis 客户端实例

    async def init(self):
        """初始化 Redis 连接"""
        import redis
        import redis.asyncio as async_redis

        pool = async_redis.ConnectionPool.from_url(
            f"redis://{self._host}:{self._port}/{self._db}",
            password=self._auth,
        )
        self._redis = async_redis.Redis.from_pool(pool)
        self._sync_client = redis.Redis(
            host=self._host,
            port=self._port,
            db=self._db,
            password=self._auth,
        )
        # 验证连接
        await self._redis.ping()
        await self._redis.close()

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        存储键值对到 Redis，支持设置 TTL。
        如果 value 是字符串，直接编码为 bytes；
        否则使用 pickle 序列化。
        """
        if self._sync_client is None:
            raise RuntimeError("Redis client is not initialized. Call init() first.")
        if isinstance(value, str):
            value_bytes = value.encode("utf-8")
        else:
            value_bytes = pickle.dumps(value)

        if ttl:
            self._sync_client.set(key, value_bytes, ex=ttl)
        else:
            self._sync_client.set(key, value_bytes)

    def get(self, key: str) -> str:
        """
        获取键对应的值。尝试以 UTF-8 解码；
        如果失败，尝试用 pickle 反序列化。
        返回值为字符串。
        """
        if self._sync_client is None:
            raise RuntimeError("Redis client is not initialized. Call init() first.")
        value = self._sync_client.get(key)
        if value is None:
            return ""
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return pickle.loads(value).decode("utf-8")  # 假设存储的是可解码字符串

    def delete(self, key: str):
        """删除指定键"""
        if self._sync_client is None:
            raise RuntimeError("Redis client is not initialized. Call init() first.")
        self._sync_client.delete(key)

    def clear(self):
        """清空当前 Redis 数据库"""
        if self._sync_client is None:
            raise RuntimeError("Redis client is not initialized. Call init() first.")
        self._sync_client.flushdb()

    async def aset(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        存储键值对到 Redis，支持设置 TTL。
        如果 value 是字符串，直接编码为 bytes；
        否则使用 pickle 序列化。
        """
        if self._redis is None:
            raise RuntimeError("Redis client is not initialized. Call init() first.")
        if isinstance(value, str):
            value_bytes = value.encode("utf-8")
        else:
            value_bytes = pickle.dumps(value)

        if ttl:
            await self._redis.set(key, value_bytes, ex=ttl)
        else:
            await self._redis.set(key, value_bytes)

    async def aget(self, key: str) -> str:
        """
        获取键对应的值。尝试以 UTF-8 解码；
        如果失败，尝试用 pickle 反序列化。
        返回值为字符串。
        """
        if self._redis is None:
            raise RuntimeError("Redis client is not initialized. Call init() first.")
        value = await self._redis.get(key)
        if value is None:
            return ""
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return pickle.loads(value).decode("utf-8")  # 假设存储的是可解码字符串

    async def adelete(self, key: str):
        """删除指定键"""
        if self._redis is None:
            raise RuntimeError("Redis client is not initialized. Call init() first.")
        self._redis.delete(key)

    async def aclear(self):
        """清空当前 Redis 数据库"""
        await self._redis.flushdb()
