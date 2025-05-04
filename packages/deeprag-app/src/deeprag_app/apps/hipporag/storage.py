import json
from typing import List, Optional

from deeprag_app.app_schemas.hipporag_models import OpenIEInfo
from deeprag_app.config.config_models import CacheConfig, DatabaseConfig
from deeprag_core.storage.cache import CacheStorage
from deeprag_core.storage.sql import SQLStorage
from deeprag_core.utils.class_meta import ClassFactory
from sqlalchemy import Engine


class HippoRAGStorage:
    def __init__(self, database_config: DatabaseConfig, cache_config: CacheConfig):
        self._database_config = database_config
        self._engine: Optional[Engine] = None
        self._cache_config = cache_config
        self._cache: Optional[CacheStorage] = None
        self._cache_prefix = "deeprag:hipporag"

    def init(self):
        # Initialize storage based on the database configuration
        self._db = ClassFactory.get_instance(
            self._database_config.provider, SQLStorage, db_url=self._database_config.url
        )
        self._db.init()
        self._cache = ClassFactory.get_instance(
            self._cache_config.provider, CacheStorage, cache_dir=self._cache_config.cache_dir
        )
        self._cache.init()

    def save_openie_info(self, openie_info: List[OpenIEInfo]):
        self._db.batch_add(openie_info)

    def get_openie_info(self, key: str) -> Optional[OpenIEInfo]:
        return self._db.get(key)

    def set_ent_node_to_chunk_ids(self, ent_node_id: str, ent_node_to_chunk_ids: List[str]):
        if not self._cache:
            raise ValueError("Cache is not initialized")
        # Store the mapping of entity node ID to chunk IDs in the cache
        self._cache.set(f"{self._cache_prefix}:entity:{ent_node_id}", json.dumps(ent_node_to_chunk_ids))

    def get_ent_node_to_chunk_ids(self, ent_node_id: str) -> Optional[List[str]] | None:
        if not self._cache:
            raise ValueError("Cache is not initialized")
        # Retrieve the mapping of entity node ID to chunk IDs from the cache
        cached_value = self._cache.get(f"{self._cache_prefix}:entity:{ent_node_id}")
        if cached_value:
            return json.loads(cached_value)
        return None

    def set_node_to_node_stats(self, from_node_key: str, to_node_key: str, stats: float):
        if not self._cache:
            raise ValueError("Cache is not initialized")
        # Store the node-to-node statistics in the cache
        self._cache.set(f"{self._cache_prefix}:node_stats:{from_node_key}:{to_node_key}", stats)

    def get_node_to_node_stats(self, from_node_key: str, to_node_key: str) -> Optional[float]:
        if not self._cache:
            raise ValueError("Cache is not initialized")
        # Retrieve the node-to-node statistics from the cache
        cached_value = self._cache.get(f"{self._cache_prefix}:node_stats:{from_node_key}:{to_node_key}")
        if cached_value:
            return float(cached_value)
        return 0.0
