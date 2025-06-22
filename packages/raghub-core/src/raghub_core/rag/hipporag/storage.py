import json
from typing import Dict, List, Optional, Set, Tuple

from raghub_core.config.raghub_config import CacheConfig, DatabaseConfig, SearchEngineConfig
from raghub_core.schemas.hipporag_models import OpenIEInfo
from raghub_core.storage.cache import CacheStorage
from raghub_core.storage.structed_data import StructedDataStorage
from raghub_core.utils.class_meta import ClassFactory
from raghub_core.utils.misc import compute_mdhash_id
from sqlmodel import JSON, and_, cast, select

from .hipporag_storage import HipporagStorage


class HippoRAGLocalStorage(HipporagStorage):
    name = "hipporag_storage_local"

    def __init__(self, db_config: DatabaseConfig, cache_config: CacheConfig, search_engine_config: SearchEngineConfig):
        self._db_config = db_config
        self._cache_config = cache_config
        self._search_engine_config = search_engine_config
        self._cache: Optional[CacheStorage] = None
        self._cache_prefix = "raghub:hipporag"
        self._db: Optional[StructedDataStorage] = None

    async def create_new_index(self, label: str):
        pass

    async def init(self):
        # Initialize storage based on the database configuration
        self._db = ClassFactory.get_instance(
            self._db_config.provider, StructedDataStorage, **self._db_config.model_dump()
        )
        await self._db.init()
        self._cache = ClassFactory.get_instance(
            self._cache_config.provider, CacheStorage, **self._cache_config.model_dump()
        )
        await self._cache.init()

    async def save_openie_info(self, label: str, openie_info: List[OpenIEInfo]):
        if not self._db:
            raise ValueError("Database is not initialized")
        # Save OpenIEInfo objects to the database
        await self._db.batch_add(openie_info)

    async def get_openie_info(self, label: str, keys: List[str]) -> List[OpenIEInfo]:
        if not self._db:
            raise ValueError("Database is not initialized")
        return await self._db.get(keys, OpenIEInfo)

    async def delete_openie_info(self, label: str, keys: List[str]):
        if not self._db:
            raise ValueError("Database is not initialized")
        await self._db.delete(keys, OpenIEInfo)

    async def set_ent_node_to_chunk_ids(self, label: str, ent_node_id: str, ent_node_to_chunk_ids: List[str]):
        if not self._cache:
            raise ValueError("Cache is not initialized")
        # Store the mapping of entity node ID to chunk IDs in the cache
        key = self.get_ent_node_to_chunk_cache_key(label, ent_node_id)
        await self._cache.aset(
            key, json.dumps(ent_node_to_chunk_ids, ensure_ascii=False), 60 * 60
        )  # Set cache expiration time to 5 minutes

    async def get_ent_node_to_chunk_ids(self, label: str, ent_node_id: str) -> Optional[List[str]] | None:
        if not self._cache:
            raise ValueError("Cache is not initialized")
        # Retrieve the mapping of entity node ID to chunk IDs from the cache
        key = self.get_ent_node_to_chunk_cache_key(label, ent_node_id)
        cached_value = await self._cache.aget(key)
        if cached_value:
            return json.loads(cached_value)
        return None

    async def set_node_to_node_stats(self, label: str, from_node_key: str, to_node_key: str, stats: float):
        if not self._cache:
            raise ValueError("Cache is not initialized")
        # Store the node-to-node statistics in the cache
        key = self.get_node_to_node_stats_cache_key(label, from_node_key, to_node_key)
        await self._cache.aset(key, stats, 60 * 60)  # Set cache expiration time to 1 hour

    async def delete_nodes_cache(self, label: str, node_keys: List[str]):
        if not self._cache:
            raise ValueError("Cache is not initialized")
        # Delete the node cache from the cache
        for node_key in node_keys:
            key = f"{self._cache_prefix}:{label}:node_stats:{compute_mdhash_id(label, node_key)}:*"
            await self._cache.adelete(key)

    async def get_node_to_node_stats(self, label: str, from_node_key: str, to_node_key: str) -> Optional[float]:
        if not self._cache:
            raise ValueError("Cache is not initialized")
        # Retrieve the node-to-node statistics from the cache
        key = self.get_node_to_node_stats_cache_key(label, from_node_key, to_node_key)
        cached_value = await self._cache.aget(key)
        if cached_value:
            return float(cached_value)
        return 0.0

    async def set_triples_to_docs(self, label: str, triples: Dict[str, Set[str]]):
        if not self._cache:
            raise ValueError("Cache is not initialized")
        for key, triple in triples.items():
            key_tulple: Tuple[str, str, str] = eval(key)
            key = self.get_triples_to_docs_cache_key(label, key_tulple)

            await self._cache.aset(key, json.dumps(list(triple), ensure_ascii=False), 5 * 60)  # 设置缓存过期时间为5分钟

    async def get_docs_from_triples(self, label, triples: Tuple[str, str, str]) -> List[str]:
        key = self.get_triples_to_docs_cache_key(label, triples)
        if not self._cache:
            raise ValueError("Cache is not initialized")
        ret = await self._cache.aget(key)
        if ret:
            return json.loads(ret)
        # 如果缓存中没有，则从数据库中查询
        docs = await self.get_docs_from_triple_sql(triples)  # noqa: F811
        if docs:
            # 将查询结果存入缓存
            await self._cache.aset(key, json.dumps(docs, ensure_ascii=False), 60 * 60 * 24)
            return docs
        return []

    async def get_docs_from_triple_sql(self, triples: Tuple[str, str, str]) -> List[str]:
        triple_str = str(triples)

        # 构建查询条件
        openie_condition = cast(OpenIEInfo.extracted_triples, JSON).contains(triple_str)

        # 查询OpenIEInfo表
        openie_query = select(OpenIEInfo.idx).where(and_(openie_condition, OpenIEInfo.is_deleted == False))  # noqa: E712
        if not self._db:
            raise ValueError("Database is not initialized")
        result = await self._db.exec(openie_query)
        return [row.idx for row in result]

    async def delete_ent_node_to_chunk_ids(self, label, ent_node_ids: List[str]):
        if not self._cache:
            raise ValueError("Cache is not initialized")
        # Delete the mapping of entity node ID to chunk IDs from the cache
        for ent_node_id in ent_node_ids:
            key = self.get_ent_node_to_chunk_cache_key(label, ent_node_id)
            await self._cache.adelete(key)

    async def delete_node_to_node_stats(self, label, from_node_key: str, to_node_key: str):
        if not self._cache:
            raise ValueError("Cache is not initialized")
        # Delete the node-to-node statistics from the cache
        key = self.get_node_to_node_stats_cache_key(label, from_node_key, to_node_key)
        await self._cache.adelete(key)

    async def delete_triples_to_docs(self, label, triples: List[Tuple[str, str, str]]):
        if not self._cache:
            raise ValueError("Cache is not initialized")
        # Delete the mapping of triples to documents from the cache
        for triple in triples:
            key = self.get_triples_to_docs_cache_key(label, triple)
            await self._cache.adelete(key)

    def get_ent_node_to_chunk_cache_key(self, label: str, ent_node_id: str) -> str:
        return f"{self._cache_prefix}:{label}:entity:{ent_node_id}"

    def get_node_to_node_stats_cache_key(self, label: str, from_node_key: str, to_node_key: str) -> str:
        return f"""{self._cache_prefix}:{label}:
        node_stats:{compute_mdhash_id(label, from_node_key)}:{compute_mdhash_id(label, to_node_key)}"""

    def get_triples_to_docs_cache_key(self, label, triples: Tuple[str, str, str]) -> str:
        key = "_".join(triples)
        return f"{self._cache_prefix}:{label}:triple:{compute_mdhash_id(label, key)}"
