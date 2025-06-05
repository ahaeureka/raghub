import datetime
from copy import deepcopy
from typing import Dict, List, Optional, Type

from loguru import logger
from raghub_core.storage.search_engine import SearchEngineStorage
from raghub_core.utils.misc import get_primary_key_names
from sqlmodel import SQLModel


class ElasticsearchEngine(SearchEngineStorage):
    name = "elasticsearch"

    def __init__(
        self,
        host: str,
        port: int,
        username: Optional[str] = None,
        password: Optional[str] = None,
        use_ssl=False,
        verify_certs=False,
        index_name_prefix="raghub",
    ):
        try:
            from elasticsearch import AsyncElasticsearch
        except ImportError:
            raise ImportError("Please install elasticsearch with pip install elasticsearch")
        self._client: Optional[AsyncElasticsearch] = None
        self._host = host
        self._port = port
        self._username = username
        self._password = password
        self._use_ssl = use_ssl
        self._verify_certs = verify_certs
        self._index_name_prefix = index_name_prefix

    async def init(self):
        """
        Initialize Elasticsearch client
        """
        from elasticsearch import AsyncElasticsearch

        uri = f"http://{self._host}:{self._port}"
        if self._use_ssl:
            uri = f"https://{self._host}:{self._port}"
        self._client = AsyncElasticsearch(
            hosts=[uri],
            http_auth=(self._username, self._password) if self._username and self._password else None,
            verify_certs=self._verify_certs,
        )
        if not await self._client.ping():
            raise ConnectionError("Elasticsearch connection failed.")
        else:
            logger.info("Elasticsearch connection successful.")

    async def create_index(self, index_name: str, index_mapping: Optional[dict] = None):
        """
        Create an index in Elasticsearch
        """
        if not self._client:
            raise ValueError("Elasticsearch client is not initialized. Call init() first.")
        index_name = f"{self._index_name_prefix}_{index_name}"
        if not await self._client.indices.exists(index=index_name):
            await self._client.indices.create(index=index_name, body=index_mapping)
            logger.info(f"Index {index_name} created.")
        else:
            logger.info(f"Index {index_name} already exists.")

    async def insert_document(self, index_name: str, documents: list[SQLModel]):
        """
        Insert a document into Elasticsearch
        """
        if not self._client:
            raise ValueError("Elasticsearch client is not initialized. Call init() first.")
        index_name = f"{self._index_name_prefix}_{index_name}"
        operations = []
        for doc in documents:
            operations.append(
                {
                    "index": {
                        "_index": index_name,
                        "_id": getattr(doc, get_primary_key_names(doc)[0]),
                    }
                }
            )
            d = doc.model_dump()
            d["created_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            d["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            d["is_deleted"] = False
            d["deleted_at"] = None
            operations.append(d)
        res = await self._client.bulk(index=index_name, operations=operations)
        logger.info(f"Document inserted into index {index_name} {res}.")

    def _transform_triples(self, triples: List[Dict[str, str]]) -> List[List[str]]:
        """
        Transform triples to the format of list of list
        """
        transformed_triples = []
        for triple in triples:
            transformed_triples.append([triple["subject"], triple["predicate"], triple["object"]])
        return transformed_triples

    async def get_documents(
        self, index_name: str, query: dict, model_cls: Type[SQLModel], filter_deleted: bool = True
    ) -> list[SQLModel]:
        """
        Get documents from Elasticsearch
        """
        if not self._client:
            raise ValueError("Elasticsearch client is not initialized. Call init() first.")
        index_name = f"{self._index_name_prefix}_{index_name}"
        new_body = deepcopy(query)

        if filter_deleted:
            original_query = new_body.get("query", {})
            new_query = {"bool": {"must": [original_query], "must_not": {"term": {"is_deleted": True}}}}
            new_body["query"] = new_query
        response = await self._client.search(index=index_name, body=query)
        hits = response["hits"]["hits"]
        documents: List[SQLModel] = []
        for hit in hits:
            source = hit["_source"]
            # source["extracted_triples"] = self._transform_triples(source["extracted_triples"])
            doc = model_cls.model_validate(source)
            documents.append(doc)
        return documents

    async def delete_documents(self, index_name: str, keys: List[str], soft_deleted: bool = True):
        """
        Delete documents from Elasticsearch
        """
        if not self._client:
            raise ValueError("Elasticsearch client is not initialized. Call init() first.")
        index_name = f"{self._index_name_prefix}_{index_name}"
        actions = [
            {
                "update": {
                    "_index": index_name,
                    "_id": key,
                    "_source": {
                        "is_deleted": True,
                        "deleted_at": datetime.datetime.now(datetime.timezone.utc),
                    },
                }
            }
            for key in keys
        ]
        if not soft_deleted:
            actions = [
                {
                    "delete": {
                        "_index": index_name,
                        "_id": key,
                    }
                }
                for key in keys
            ]
        await self._client.bulk(operations=actions)
        logger.info(f"Documents with keys {keys} deleted from index {index_name}.")
