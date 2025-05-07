import json
from typing import List, Tuple

from raghub_core.config.raghub_config import DatabaseConfig, SearchEngineConfig
from raghub_core.rag.hipporag.storage import HippoRAGLocalStorage
from raghub_core.schemas.hipporag_models import OpenIEInfo
from raghub_core.storage.search_engine import SearchEngineStorage
from raghub_core.utils.class_meta import ClassFactory

# class OpenIEElasticModel(OpenIEInfo):
#     extracted_triples: List[Dict[str, str]] = Field(
#         ...,
#         description="List of triples extracted from the text.",
#         sa_column=Column(JSON, doc="List of triples extracted from the text."),
#     )


class HipporagOnlineStorage(HippoRAGLocalStorage):
    """
    This class is a placeholder for the HipporagOnlineStorage.
    It is not implemented yet and serves as a stub for future development.
    """

    name = "hipporag_storage_online"

    def __init__(
        self, db_config: DatabaseConfig, cache_config: DatabaseConfig, search_engine_config: SearchEngineConfig
    ):
        """
        Initialize the HipporagOnlineStorage class.
        """
        super().__init__(db_config, cache_config, search_engine_config)
        self._search_engine = ClassFactory.get_instance(
            search_engine_config.provider, SearchEngineStorage, **search_engine_config.model_dump()
        )

    def create_new_index(self, label: str):
        index_body = {
            "mappings": {
                "properties": {
                    "idx": {"type": "keyword"},
                    "passage": {
                        "type": "text",
                        "analyzer": "standard",
                    },
                    "extracted_entities": {
                        "type": "keyword",
                    },
                    "extracted_triples": {
                        "type": "nested",
                        "properties": {
                            "subject": {"type": "keyword"},
                            "predicate": {"type": "keyword"},
                            "object": {"type": "keyword"},
                        },
                    },
                    "created_at": {
                        "type": "date",
                        "format": "strict_date_optional_time||epoch_millis",
                    },
                    "updated_at": {
                        "type": "date",
                        "format": "strict_date_optional_time||epoch_millis",
                    },
                    "deleted_at": {
                        "type": "date",
                        "format": "strict_date_optional_time||epoch_millis",
                    },
                    "is_deleted": {"type": "boolean"},
                }
            },
        }
        self._search_engine.create_index(label, index_mapping=index_body)

    def init(self):
        super().init()
        self._search_engine.init()

    def _openie_to_elastic_model(self, openie_info: OpenIEInfo) -> OpenIEInfo:
        """
        Convert OpenIEInfo to OpenIEElasticModel.
        Args:
            openie_info (OpenIEInfo): The OpenIEInfo object to be converted.
        Returns:
            OpenIEElasticModel: The converted OpenIEElasticModel object.
        """
        return OpenIEInfo(
            idx=openie_info.idx,
            passage=openie_info.passage,
            extracted_entities=openie_info.extracted_entities,
            extracted_triples=[
                {"subject": triple[0], "predicate": triple[1], "object": triple[2]}
                for triple in openie_info.extracted_triples
            ],
            created_at=openie_info.created_at,
            updated_at=openie_info.updated_at,
            deleted_at=openie_info.deleted_at,
            is_deleted=openie_info.is_deleted,
        )

    def _elastic_model_to_openie(self, openie_info: OpenIEInfo) -> OpenIEInfo:
        """
        Convert OpenIEElasticModel to OpenIEInfo.
        Args:
            openie_info (OpenIEElasticModel): The OpenIEElasticModel object to be converted.
        Returns:
            OpenIEInfo: The converted OpenIEInfo object.
        """
        return OpenIEInfo(
            idx=openie_info.idx,
            passage=openie_info.passage,
            extracted_entities=openie_info.extracted_entities,
            extracted_triples=[
                [triple["subject"], triple["predicate"], triple["object"]] for triple in openie_info.extracted_triples
            ],
            created_at=openie_info.created_at,
            updated_at=openie_info.updated_at,
            deleted_at=openie_info.deleted_at,
            is_deleted=openie_info.is_deleted,
        )

    def save_openie_info(self, label: str, openie_info: List[OpenIEInfo]):
        """
        Save OpenIE information to the storage.
        Args:
            openie_info (List[OpenIEInfo]): A list of OpenIEInfo objects to be saved.
        """

        self._search_engine.insert_document(
            label, documents=[self._openie_to_elastic_model(info) for info in openie_info]
        )

    def get_openie_info(self, label: str, keys: List[str]) -> List[OpenIEInfo]:
        """
        Retrieve OpenIE information from the storage.
        Args:
            keys (List[str]): A list of keys to retrieve OpenIE information.
        Returns:
            List[OpenIEInfo]: A list of OpenIEInfo objects retrieved from the storage.
        """
        docs = self._search_engine.get_documents(
            label,
            query={"query": {"terms": {"idx": keys}}},
            model_cls=OpenIEInfo,
        )
        if not docs:
            return []
        openie_info_list = [self._elastic_model_to_openie(doc) for doc in docs]
        return openie_info_list

    def delete_openie_info(self, label: str, keys: List[str]):
        """
        Delete OpenIE information from the storage.
        Args:
            keys (List[str]): A list of keys to delete OpenIE information.
        """
        self._search_engine.delete_documents(label, keys=keys)

    def get_docs_from_triples(self, label: str, triples: Tuple[str, str, str]) -> List[str]:
        """
        Retrieve the documents associated with a given triple from the cache.
        Args:
            triples (Tuple[str, str, str]): A tuple representing the triple.
        Returns:
            List[str]: A list of document IDs associated with the triple.
        """
        key = self.get_triples_to_docs_cache_key(triples)
        if not self._cache:
            raise ValueError("Cache is not initialized")
        ret = self._cache.get(key)
        if ret:
            return json.loads(ret)
        # 如果缓存中没有，则从数据库中查询
        docs = self.get_docs_from_es_use_triples(label, triples)  # noqa: F811
        if docs:
            # 将查询结果存入缓存

            self._cache.set(key, json.dumps(docs, ensure_ascii=False))
            return docs
        return []

    def get_docs_from_es_use_triples(self, label: str, triples: Tuple[str, str, str]) -> List[str]:
        query = {
            "query": {
                "should": [
                    {
                        "nested": {
                            "path": "extracted_triples",
                            "query": {
                                "bool": {
                                    "must": [
                                        {"term": {"extracted_triples.subject": triples[0]}},
                                        {"term": {"extracted_triples.predicate": triples[1]}},
                                        {"term": {"extracted_triples.object": triples[2]}},
                                    ]
                                }
                            },
                        }
                    }
                ],
                "minimum_should_match": 1,
            }
        }
        docs = self._search_engine.get_documents(
            label,
            query=query,
            model_cls=OpenIEInfo,
        )
        return [doc.idx for doc in docs]
