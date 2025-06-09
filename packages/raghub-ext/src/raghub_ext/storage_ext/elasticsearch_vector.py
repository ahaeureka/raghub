from typing import Any, Dict, List, Optional, Tuple

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from loguru import logger
from pydantic import BaseModel
from raghub_core.embedding.base_embedding import BaseEmbedding
from raghub_core.schemas.document import Document
from raghub_core.storage.embedding_adapter import LangchainEmbeddings
from raghub_core.storage.vector import VectorStorage


class ElasticsearchVectorStorage(VectorStorage):
    name = "elasticsearch_vector"

    def __init__(
        self,
        embedder: BaseEmbedding,
        host: str = "localhost",
        port: int = 9200,
        use_ssl: bool = False,
        verify_certs: bool = True,
        username: Optional[str] = "elastic",
        password: Optional[str] = None,
        index_name_prefix: str = "raghub_index",
    ):
        try:
            from elasticsearch import AsyncElasticsearch, Elasticsearch

        except ImportError:
            raise ImportError("Please install langchain_elasticsearch with pip install langchain_elasticsearch")

        self._embedder = embedder
        self._host = host
        self._port = port
        self._use_ssl = use_ssl
        self._verify_certs = verify_certs
        self._http_auth = (username, password) if username and password else None
        self._client: Optional[Elasticsearch] = None
        self._async_client: Optional[AsyncElasticsearch] = None
        self._index_name_prefix = index_name_prefix

    async def init(self):
        """
        Initialize Elasticsearch client
        """
        try:
            from elasticsearch import AsyncElasticsearch, Elasticsearch

            if not self._use_ssl:
                self._client = Elasticsearch(
                    hosts=[f"http://{self._host}:{self._port}"],
                    verify_certs=self._verify_certs,
                    http_auth=self._http_auth,
                )
                self._async_client = AsyncElasticsearch(
                    hosts=[f"http://{self._host}:{self._port}"],
                    verify_certs=self._verify_certs,
                    http_auth=self._http_auth,
                )
            else:
                self._client = Elasticsearch(
                    hosts=[f"https://{self._host}:{self._port}"],
                    verify_certs=self._verify_certs,
                    http_auth=self._http_auth,
                )
                self._async_client = AsyncElasticsearch(
                    hosts=[f"https://{self._host}:{self._port}"],
                    verify_certs=self._verify_certs,
                    http_auth=self._http_auth,
                )
            if not await self._async_client.ping():
                raise ConnectionError("Elasticsearch connection failed.")
            logger.debug("ElasticsearchVectorStorage successfully initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Elasticsearch vector store: {e}")
            raise

    def _es_store_for_index(self, index_name: str) -> VectorStore:
        """
        创建 ElasticsearchStore 实例
        """
        from langchain_elasticsearch import ElasticsearchStore

        if not self._client:
            raise ValueError("Elasticsearch client is not initialized.")
        from langchain_elasticsearch._utilities import DistanceStrategy

        return ElasticsearchStore(
            es_connection=self._client,
            index_name=f"{self._index_name_prefix}_{index_name}",
            embedding=self._build_embedding_function(),
            distance_strategy=DistanceStrategy.COSINE,
        )

    def _build_embedding_function(self) -> Embeddings:
        return LangchainEmbeddings(self._embedder)

    async def add_documents(self, index_name: str, texts: List[Document]) -> List[Document]:
        """
        将文档添加到 Elasticsearch 索引
        """
        if not self._client:
            raise ValueError("Elasticsearch client is not initialized.")

        ids = [doc.uid for doc in texts]
        contents = [doc.content for doc in texts]
        metadatas: List[Dict[str, Any]] = []
        for doc in texts:
            metadata = doc.metadata
            if isinstance(metadata, dict):
                metadatas.append(metadata)
            elif isinstance(metadata, BaseModel):
                metadatas.append(metadata.model_dump())
            else:
                metadatas.append({})

        if not self._client:
            raise ValueError("Elasticsearch client is not initialized.")
        ids = await self._es_store_for_index(index_name).aadd_texts(texts=contents, metadatas=metadatas, ids=ids)
        return await self.get_by_ids(index_name=index_name, ids=ids)

    async def get(self, index_name: str, uid: str) -> Optional[Document] | None:
        """
        根据 ID 获取单个文档
        """
        if not self._client:
            raise ValueError("Elasticsearch client is not initialized.")
        if self._client is None:
            raise ValueError("Elasticsearch client is not initialized.")
        result = await self.get_by_ids(index_name, ids=[uid])
        if not result:
            return None

        return result[0]

    async def get_by_ids(self, index_name: str, ids: List[str]) -> List[Document]:
        """
        根据多个 ID 获取文档
        """
        if not self._async_client:
            raise ValueError("Elasticsearch client is not initialized.")
        body = {"docs": [{"_index": f"{self._index_name_prefix}_{index_name}", "_id": doc_id} for doc_id in ids]}
        response = await self._async_client.mget(body=body)
        docs: List[Document] = []  # noqa: F811
        for res in response["docs"]:
            if not res.get("found"):
                logger.warning(f"Document {index_name} with ID {res['_id']} not found:{res}.")
                continue
            source = res["_source"]
            source["uid"] = res["_id"]  # 将 _id 转换为 uid
            source["content"] = source.get("text", "")
            source["summary"] = source.get("metadata", {}).pop("summary", "")
            source["embedding"] = source.pop("vector", [])
            doc = Document(**source)
            docs.append(doc)
        return docs

    async def delete(self, index_name: str, ids: List[str]) -> bool:
        """
        删除指定 ID 的文档
        """
        if not self._client:
            raise ValueError("Elasticsearch client is not initialized.")

        try:
            await self._es_store_for_index(index_name).adelete(ids=ids)
            return True
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return False

    def _build_metadata_query(self, metadata_filter: Dict[str, Any]) -> Dict[str, Any]:
        bool_query: Dict[str, Any] = {"bool": {"must": []}}
        for key, value in metadata_filter.items():
            if isinstance(value, list):
                # 如果值是列表，则使用 terms 查询
                bool_query["bool"]["must"].append({"terms": {f"metadata.{key}.keyword": value}})
            else:
                # 否则使用 term 查询
                bool_query["bool"]["must"].append({"term": {f"metadata.{key}.keyword": value}})
        return bool_query

    async def select_on_metadata(self, index_name: str, metadata_filter: Dict[str, Any]) -> List[Document]:
        """
        根据元数据过滤文档
        """
        if not self._async_client:
            raise ValueError("Elasticsearch client is not initialized.")

        # 构造 Elasticsearch 查询语句
        bool_query = self._build_metadata_query(metadata_filter)

        # 确保返回所有字段（包括 _source）
        query = {
            "query": bool_query,
            "_source": True,  # 显式指定返回所有字段
        }
        result = await self._async_client.search(index=f"{self._index_name_prefix}_{index_name}", body=query)
        if not result or len(result["hits"]["hits"]) == 0:
            logger.warning(f"No documents found in index {self._index_name_prefix}_{index_name} with {result}")
            return []

        # 处理 Elasticsearch 返回的结果
        documents = []
        for hit in result["hits"]["hits"]:
            source = hit["_source"]
            _id = hit["_id"]
            doc = Document(
                content=source["text"],
                metadata=source["metadata"],
                uid=_id,
                embedding=source["vector"],
            )
            documents.append(doc)
        return documents
        # return self._build_doc_from_result(result)

    async def similarity_search_by_vector(
        self, index_name: str, embedding: List[float], k: int, filter: Optional[Dict[str, str]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Similarity search by vector
        Args:
            index_name : str
                The name of the index to search in.
            embedding : List[float]
                The embedding vector to search for.
            k : int
                The number of top results to return.
            filter : Optional[Dict[str, str]]
                Optional filter to apply to the search results.
        Returns:
            List[Tuple[Document, float]]:
                A list of tuples containing the Document object and its corresponding score.
        """
        if not self._client:
            raise ValueError("Elasticsearch client is not initialized.")
        bool_query: Dict[str, Any] | None = None

        def _doc_builder(doc: Dict[str, Any]) -> Document:
            return Document(content=doc["_source"]["text"], metadata=doc["_source"]["metadata"], uid=doc["_id"])

        if filter:
            bool_query = self._build_metadata_query(filter)
        results = await self._es_store_for_index(index_name).asimilarity_search_by_vector_with_relevance_scores(  # type: ignore[attr-defined]
            embedding=embedding, k=k, filter=[bool_query], doc_builder=_doc_builder
        )
        return [(doc, score) for doc, score in results]

    async def similar_search_with_scores(
        self, index_name: str, query: str, k: int, filter: Optional[Dict[str, str]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Similarity search with scores
        Args:
            index_name : str
                The name of the index to search in.
            query : str
                The query string to search for.
            k : int
                The number of top results to return.
            filter : Optional[Dict[str, str]]
                Optional filter to apply to the search results.
        Returns:
            List[Tuple[Document, float]]:
                A list of tuples containing the Document object and its corresponding score.
        """
        if not self._client:
            raise ValueError("Elasticsearch client is not initialized.")
        bool_query: Dict[str, Any] | None = None

        def _doc_builder(doc: Dict[str, Any]) -> Document:
            return Document(content=doc["_source"]["text"], metadata=doc["_source"]["metadata"], uid=doc["_id"])

        if filter:
            bool_query = self._build_metadata_query(filter)
        import elasticsearch

        try:
            results = await self._es_store_for_index(index_name).asimilarity_search_with_score(  # type: ignore[attr-defined]
                query=query, k=k, filter=bool_query, doc_builder=_doc_builder
            )
        except elasticsearch.NotFoundError:
            return []
        return [(doc, score) for doc, score in results]

    def _build_doc_from_result(self, result: Dict[str, Any]) -> List[Document]:
        """
        将 Elasticsearch 返回结果转换为 Document 对象
        """
        if not result or not result["ids"]:
            return []

        documents = []
        for i in range(len(result["ids"])):
            doc = Document(
                content=result["documents"][i],
                metadata=result["metadatas"][i],
                uid=result["ids"][i],
                embedding=result["embeddings"][i],
                summary=result["metadatas"][i]["summary"] if "summary" in result["metadatas"][i] else "",
            )
            documents.append(doc)
        return documents

    async def asimilar_search_with_scores(self, index_name, query, k, filter=None):
        """
        Asynchronous similarity search with scores
        Args:
            index_name : str
                The name of the index to search in.
            query : str
                The query string to search for.
            k : int
                The number of top results to return.
            filter : Optional[Dict[str, str]]
                Optional filter to apply to the search results.
        Returns:
            List[Tuple[Document, float]]:
                A list of tuples containing the Document object and its corresponding score.
        """
        return await self.similar_search_with_scores(index_name, query, k, filter)

    async def create_index(self, index_name):
        pass
