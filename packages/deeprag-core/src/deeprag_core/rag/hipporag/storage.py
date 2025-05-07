import json
from typing import Dict, List, Optional, Set, Tuple

from deeprag_core.config.deeprag_config import CacheConfig, DatabaseConfig, SearchEngineConfig
from deeprag_core.schemas.hipporag_models import OpenIEInfo
from deeprag_core.storage.cache import CacheStorage
from deeprag_core.storage.structed_data import StructedDataStorage
from deeprag_core.utils.class_meta import ClassFactory
from deeprag_core.utils.misc import compute_mdhash_id
from sqlmodel import JSON, and_, cast, select

from .hipporag_storage import HipporagStorage


class HippoRAGLocalStorage(HipporagStorage):
    name = "hipporag_storage_local"

    def __init__(self, db_config: DatabaseConfig, cache_config: CacheConfig, search_engine_config: SearchEngineConfig):
        self._db_config = db_config
        self._cache_config = cache_config
        self._search_engine_config = search_engine_config
        self._cache: Optional[CacheStorage] = None
        self._cache_prefix = "deeprag:hipporag"

    def create_new_index(self, label: str):
        pass

    def init(self):
        # Initialize storage based on the database configuration
        self._db = ClassFactory.get_instance(
            self._db_config.provider, StructedDataStorage, **self._db_config.model_dump()
        )
        self._db.init()
        self._cache = ClassFactory.get_instance(
            self._cache_config.provider, CacheStorage, **self._cache_config.model_dump()
        )
        self._cache.init()

    def save_openie_info(self, label: str, openie_info: List[OpenIEInfo]):
        self._db.batch_add(openie_info)

    def get_openie_info(self, label: str, keys: List[str]) -> List[OpenIEInfo]:
        return self._db.get(keys, OpenIEInfo)

    def delete_openie_info(self, label: str, keys: List[str]):
        self._db.delete(keys, OpenIEInfo)

    def set_ent_node_to_chunk_ids(self, label: str, ent_node_id: str, ent_node_to_chunk_ids: List[str]):
        if not self._cache:
            raise ValueError("Cache is not initialized")
        # Store the mapping of entity node ID to chunk IDs in the cache
        key = self.get_ent_node_to_chunk_cache_key(label, ent_node_id)
        self._cache.set(key, json.dumps(ent_node_to_chunk_ids, ensure_ascii=False))

    def get_ent_node_to_chunk_ids(self, label: str, ent_node_id: str) -> Optional[List[str]] | None:
        if not self._cache:
            raise ValueError("Cache is not initialized")
        # Retrieve the mapping of entity node ID to chunk IDs from the cache
        key = self.get_ent_node_to_chunk_cache_key(label, ent_node_id)
        cached_value = self._cache.get(key)
        if cached_value:
            return json.loads(cached_value)
        return None

    def set_node_to_node_stats(self, label: str, from_node_key: str, to_node_key: str, stats: float):
        if not self._cache:
            raise ValueError("Cache is not initialized")
        # Store the node-to-node statistics in the cache
        key = self.get_node_to_node_stats_cache_key(label, from_node_key, to_node_key)
        self._cache.set(key, stats)

    def get_node_to_node_stats(self, label: str, from_node_key: str, to_node_key: str) -> Optional[float]:
        if not self._cache:
            raise ValueError("Cache is not initialized")
        # Retrieve the node-to-node statistics from the cache
        key = self.get_node_to_node_stats_cache_key(label, from_node_key, to_node_key)
        cached_value = self._cache.get(key)
        if cached_value:
            return float(cached_value)
        return 0.0

    def set_triples_to_docs(self, label: str, triples: Dict[str, Set[str]]):
        if not self._cache:
            raise ValueError("Cache is not initialized")
        for key, triple in triples.items():
            key_tulple: Tuple[str, str, str] = eval(key)
            key = self.get_triples_to_docs_cache_key(label, key_tulple)

            self._cache.set(key, json.dumps(list(triple), ensure_ascii=False))

    def get_docs_from_triples(self, label, triples: Tuple[str, str, str]) -> List[str]:
        key = self.get_triples_to_docs_cache_key(label, triples)
        if not self._cache:
            raise ValueError("Cache is not initialized")
        ret = self._cache.get(key)
        if ret:
            return json.loads(ret)
        # 如果缓存中没有，则从数据库中查询
        docs = self.get_docs_from_triple_sql(triples)  # noqa: F811
        if docs:
            # 将查询结果存入缓存

            self._cache.set(key, json.dumps(docs, ensure_ascii=False))
            return docs
        return []

    def get_docs_from_triple_sql(self, triples: Tuple[str, str, str]) -> List[str]:
        triple_str = str(triples)

        # 构建查询条件
        openie_condition = cast(OpenIEInfo.extracted_triples, JSON).contains(triple_str)

        # 查询OpenIEInfo表
        openie_query = select(OpenIEInfo.idx).where(and_(openie_condition, OpenIEInfo.is_deleted == False))  # noqa: E712
        result = self._db.exec(openie_query)
        return [row.idx for row in result]

    def delete_ent_node_to_chunk_ids(self, label, ent_node_ids: List[str]):
        if not self._cache:
            raise ValueError("Cache is not initialized")
        # Delete the mapping of entity node ID to chunk IDs from the cache
        for ent_node_id in ent_node_ids:
            key = self.get_ent_node_to_chunk_cache_key(label, ent_node_id)
            self._cache.delete(key)

    def delete_node_to_node_stats(self, label, from_node_key: str, to_node_key: str):
        if not self._cache:
            raise ValueError("Cache is not initialized")
        # Delete the node-to-node statistics from the cache
        key = self.get_node_to_node_stats_cache_key(label, from_node_key, to_node_key)
        self._cache.delete(key)

    def delete_triples_to_docs(self, label, triples: List[Tuple[str, str, str]]):
        if not self._cache:
            raise ValueError("Cache is not initialized")
        # Delete the mapping of triples to documents from the cache
        for triple in triples:
            key = self.get_triples_to_docs_cache_key(label, triple)
            self._cache.delete(key)

    def get_ent_node_to_chunk_cache_key(self, label: str, ent_node_id: str) -> str:
        return f"{self._cache_prefix}:{label}:entity:{ent_node_id}"

    def get_node_to_node_stats_cache_key(self, label: str, from_node_key: str, to_node_key: str) -> str:
        return f"""{self._cache_prefix}:{label}:
        node_stats:{compute_mdhash_id(from_node_key)}:{compute_mdhash_id(to_node_key)}"""

    def get_triples_to_docs_cache_key(self, label, triples: Tuple[str, str, str]) -> str:
        key = "_".join(triples)
        return f"{self._cache_prefix}:{label}:triple:{compute_mdhash_id(key)}"
