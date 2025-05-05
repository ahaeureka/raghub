import pickle
from typing import Any

from deeprag_core.storage.cache import CacheStorage
from git import Optional


class RedisCacheStorage(CacheStorage):
    name = "redis_cache"

    def __init__(self, host="localhost", port=6379, db=0):
        super().__init__()
        self._host = host
        self._port = port
        self._db = db
        try:
            import redis
        except ImportError:
            raise ImportError("Please install redis-py with pip install redis")
        self._redis: Optional[redis.Redis] = None  # Redis 客户端实例

    def init(self):
        """初始化 Redis 连接"""
        import redis

        self._redis = redis.Redis(host=self._host, port=self._port, db=self._db)
        # 验证连接
        self._redis.ping()

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        存储键值对到 Redis，支持设置 TTL。
        如果 value 是字符串，直接编码为 bytes；
        否则使用 pickle 序列化。
        """
        if isinstance(value, str):
            value_bytes = value.encode("utf-8")
        else:
            value_bytes = pickle.dumps(value)

        if ttl:
            self._redis.set(key, value_bytes, ex=ttl)
        else:
            self._redis.set(key, value_bytes)

    def get(self, key: str) -> str:
        """
        获取键对应的值。尝试以 UTF-8 解码；
        如果失败，尝试用 pickle 反序列化。
        返回值为字符串。
        """
        value = self._redis.get(key)
        if value is None:
            return ""
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return pickle.loads(value).decode("utf-8")  # 假设存储的是可解码字符串

    def delete(self, key: str):
        """删除指定键"""
        self._redis.delete(key)

    def clear(self):
        """清空当前 Redis 数据库"""
        self._redis.flushdb()
