from typing import Any, Dict, List, Optional, Tuple

from deeprag_core.embedding.base_embedding import BaseEmbedding
from deeprag_core.schemas.document import Document
from deeprag_core.storage.embedding_adapter import LangchainEmbeddings
from deeprag_core.storage.vector import VectorStorage
from langchain_core.embeddings import Embeddings
from loguru import logger
from pydantic import BaseModel


class ElasticsearchVectorStorage(VectorStorage):
    name = "elasticsearch"

    def __init__(
        self,
        embedder: BaseEmbedding,
        index_name: str,
        host: str = "localhost",
        port: int = 9200,
        use_ssl: bool = False,
        verify_certs: bool = True,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        try:
            from langchain_elasticsearch import ElasticsearchStore
        except ImportError:
            raise ImportError("Please install langchain_elasticsearch with pip install langchain_elasticsearch")

        self._embedder = embedder
        self._index_name = index_name
        self._host = host
        self._port = port
        self._use_ssl = use_ssl
        self._verify_certs = verify_certs
        self._http_auth = (username, password) if username and password else None
        self._client: Optional[ElasticsearchStore] = None

    def init(self):
        """
        初始化 ElasticsearchStore 并创建索引（如果不存在）
        """
        try:
            from elasticsearch import Elasticsearch
            from langchain_elasticsearch import ElasticsearchStore

            # 创建 Elasticsearch 客户端
            es_client = Elasticsearch(
                hosts=[f"{self._host}:{self._port}"],
                use_ssl=self._use_ssl,
                verify_certs=self._verify_certs,
                http_auth=self._http_auth,
            )

            # # 检查索引是否存在
            # if not es_client.indices.exists(index=self._index_name):
            #     # 如果不存在，创建索引并设置 mapping
            #     mapping = {
            #         "mappings": {
            #             "properties": {
            #                 "text": {"type": "text"},
            #                 "metadata": {"type": "object"},
            #                 "id": {"type": "keyword"},
            #                 "embedding": {"type": "dense_vector", "dims": len(self._embedder.encode(["test"]))},
            #             }
            #         }
            #     }
            #     es_client.indices.create(index=self._index_name, body=mapping)

            # 初始化 VectorStore
            self._client = ElasticsearchStore(
                es_client=es_client,
                index_name=self._index_name,
                embedding=self._build_embedding_function(),
                distance_strategy="COSINE",
            )
        except Exception as e:
            logger.error(f"Failed to initialize Elasticsearch vector store: {e}")
            raise

    def _build_embedding_function(self) -> Embeddings:
        return LangchainEmbeddings(self._embedder)

    def add_documents(self, texts: List[Document]) -> List[Document]:
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

        # 批量插入
        logger.debug(f"Adding {len(texts)} documents to Elasticsearch index {self._index_name}")
        if not self._client:
            raise ValueError("Elasticsearch client is not initialized.")
        self._client.add_texts(texts=contents, metadatas=metadatas, ids=ids)
        return texts

    def get(self, uid: str) -> Document:
        """
        根据 ID 获取单个文档
        """
        if not self._client:
            raise ValueError("Elasticsearch client is not initialized.")
        if self._client is None:
            raise ValueError("Elasticsearch client is not initialized.")
        result = self._client.get_by_ids(ids=[uid])
        if not result or len(result["ids"]) == 0:
            return None

        return self._build_doc_from_result(result)[0]

    def get_by_ids(self, ids: List[str]) -> List[Document]:
        """
        根据多个 ID 获取文档
        """
        if not self._client:
            raise ValueError("Elasticsearch client is not initialized.")

        result = self._client.get_by_ids(ids=ids)
        return self._build_doc_from_result(result)

    def delete(self, ids: List[str]) -> bool:
        """
        删除指定 ID 的文档
        """
        if not self._client:
            raise ValueError("Elasticsearch client is not initialized.")

        try:
            self._client.delete(ids=ids)
            return True
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return False

    def _build_metadata_query(self, metadata_filter: Dict[str, Any]) -> Dict[str, Any]:
        """
        构建 Elasticsearch 查询语句
        """
        bool_query: Dict[str, Any] = {"bool": {"must": []}}
        for key, value in metadata_filter.items():
            # 假设 metadata 中的字段是文本类型，使用 match 查询
            bool_query["bool"]["must"].append(
                {
                    "match": {
                        f"metadata.{key}": {
                            "query": value,
                            "case_insensitive": True,
                        }
                    }
                }
            )
        return bool_query

    def select_on_metadata(self, metadata_filter: Dict[str, Any]) -> List[Document]:
        """
        根据元数据过滤文档
        """
        if not self._client:
            raise ValueError("Elasticsearch client is not initialized.")

        # 构造 Elasticsearch 查询语句
        bool_query = self._build_metadata_query(metadata_filter)

        # 确保返回所有字段（包括 _source）
        query = {
            "query": bool_query,
            "_source": True,  # 显式指定返回所有字段
        }

        result = self._client.client.search(self._index_name, query=query)
        if not result or len(result["hits"]["hits"]) == 0:
            return []

        # 处理 Elasticsearch 返回的结果
        documents = []
        for hit in result["hits"]["hits"]:
            source = hit["_source"]
            doc = Document(
                content=source["text"],
                metadata=source["metadata"],
                uid=source["id"],
                embedding=source["embedding"],
            )
            documents.append(doc)
        return documents
        # return self._build_doc_from_result(result)

    def similarity_search_by_vector(
        self, embedding: List[float], k: int, metadata_filter: Optional[Dict[str, str]] = None
    ) -> List[Tuple[Document, float]]:
        """
        向量相似性搜索
        """
        if not self._client:
            raise ValueError("Elasticsearch client is not initialized.")
        bool_query: Dict[str, Any] | None = None
        if metadata_filter:
            # 构建 Elasticsearch 查询语句
            bool_query = self._build_metadata_query(metadata_filter)
        results = self._client.similarity_search_by_vector_with_relevance_scores(
            embedding=embedding, k=k, filter=[bool_query]
        )
        self._client.aadd_documents

        return [
            (Document(content=doc.page_content, metadata=doc.metadata, uid=doc.id), score) for doc, score in results
        ]

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
            )
            documents.append(doc)
        return documents
