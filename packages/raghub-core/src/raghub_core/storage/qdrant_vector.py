from __future__ import annotations

import asyncio
import hashlib
import uuid
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from raghub_core.utils.qdrant_query_converter import QdrantQueryConverter

if TYPE_CHECKING:
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient

from langchain_core.documents import Document as LangchainDocument
from langchain_core.runnables.config import run_in_executor
from langchain_core.vectorstores import VectorStore
from loguru import logger
from raghub_core.embedding.base_embedding import BaseEmbedding
from raghub_core.schemas.document import Document
from raghub_core.storage.embedding_adapter import LangchainEmbeddings
from raghub_core.storage.vector import VectorStorage
from raghub_core.utils.file.project import ProjectHelper


class QdrantVector(VectorStorage):
    """
    A vector store that uses Qdrant as the backend.
    """

    name = "qdrant"

    def __init__(
        self,
        embedder: BaseEmbedding,
        persist_directory: Path = ProjectHelper.get_project_root() / "cache/qdrant",
    ):
        try:
            self._client: Optional[QdrantClient] = None
            self._embedder: BaseEmbedding = embedder
            self._persist_directory: Path = persist_directory
            self._vector_stores: Dict[str, VectorStore] = {}
            self._metadata_filter_convert: QdrantQueryConverter = QdrantQueryConverter()
            logger.debug(f"QdrantVector initialized with embedder: {self._embedder}")
        except ImportError:
            raise ImportError("Qdrant client is not installed. Please install it using `pip install qdrant-client`.")

    @property
    def client(self) -> QdrantClient:
        """Lazy loading of Qdrant client"""
        # Check if client is None or closed
        client_is_closed = False
        if self._client is not None:
            try:
                # Try to perform a simple operation to check if it's still working
                # This will raise RuntimeError if the client is closed
                self._client.get_collections()
            except RuntimeError as e:
                if "closed" in str(e).lower():
                    client_is_closed = True
                else:
                    # Re-raise if it's a different kind of RuntimeError
                    raise
            except (AttributeError, Exception):
                # Consider any other exception as a sign that the client needs to be recreated
                client_is_closed = True

        if self._client is None or client_is_closed:
            try:
                from qdrant_client import QdrantClient

                self._client = QdrantClient(path=self._persist_directory.as_posix())
                logger.debug(f"Qdrant client initialized with path: {self._persist_directory}")
            except ImportError:
                raise ImportError(
                    "Qdrant client is not installed. Please install it using `pip install qdrant-client`."
                )
            except Exception as e:
                logger.error(f"Failed to initialize Qdrant client: {e}")
                raise
        return self._client

    async def init(self):
        """Initialize embedder (client is lazy loaded)"""
        self._embedder.init()
        await asyncio.sleep(0.01)  # Yield control to the event loop

    @lru_cache
    def create_index(self, index_name: str) -> QdrantVectorStore:
        """
        Create a new index in Qdrant.
        """
        from langchain_qdrant import QdrantVectorStore

        if index_name in self._vector_stores:
            return self._vector_stores[index_name]

        # Create collection if it doesn't exist
        try:
            self.client.get_collection(index_name)
        except ValueError:
            # Collection doesn't exist, create it
            from qdrant_client.http.models import Distance, VectorParams

            self.client.create_collection(
                collection_name=index_name,
                vectors_config=VectorParams(size=self._embedder.embedding_dim, distance=Distance.COSINE),
            )

        # Now create the vector store
        vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=index_name,
            embedding=LangchainEmbeddings(self._embedder),
        )
        self._vector_stores[index_name] = vector_store
        return vector_store

    def _qdrant_store_for_index(self, index_name: str) -> VectorStore:
        """Get or create a Qdrant store for the given index name"""
        return self.create_index(index_name)

    def _build_docs(self, documents: List[LangchainDocument], scores: Optional[List[float]] = None) -> List[Document]:
        """Build Document objects from LangChain documents"""
        result_docs = []

        for i, langchain_doc in enumerate(documents):
            # Use metadata directly without string conversion
            metadata = langchain_doc.metadata or {}

            doc = Document(
                content=langchain_doc.page_content,
                metadata=metadata,
                uid=getattr(langchain_doc, "id", None) or str(i),
                embedding=None,  # Qdrant doesn't return embeddings by default
            )
            doc.summary = metadata.get("summary", "")
            result_docs.append(doc)

        return result_docs

    async def add_documents(self, index_name: str, texts: List[Document]) -> List[Document]:
        """Add documents to the index"""
        # Add summary to metadata
        for text in texts:
            if text.metadata is None:
                text.metadata = {}
            text.metadata["summary"] = text.summary

        # Convert IDs for Qdrant storage
        storage_ids = self._convert_ids_for_storage([doc.uid for doc in texts])

        # Convert to LangChain format with converted IDs
        langchain_docs = [
            LangchainDocument(
                page_content=doc.content,
                metadata=doc.metadata,  # Use metadata directly
                id=storage_id,
            )
            for doc, storage_id in zip(texts, storage_ids)
        ]

        ids = await self._qdrant_store_for_index(index_name).aadd_documents(documents=langchain_docs, ids=storage_ids)

        logger.debug(f"Added {len(ids)} documents to index '{index_name}'")
        # Return documents with their original IDs in the same order
        original_ids = [doc.uid for doc in texts]
        return await self.get_by_ids(index_name, original_ids)

    async def get(self, index_name: str, uid: str) -> Document:
        """Get a document by its ID"""
        try:
            # Convert ID for storage lookup
            storage_id = self._convert_id_for_storage(uid)

            # Use the client property to ensure lazy loading
            points = self.client.retrieve(
                collection_name=index_name, ids=[storage_id], with_payload=True, with_vectors=False
            )

            if not points:
                raise ValueError(f"Document with ID {uid} not found")

            point = points[0]
            payload = point.payload or {}

            # Extract metadata - LangChain stores our metadata nested under 'metadata' key
            raw_metadata = payload.get("metadata", {})

            # If metadata is nested, unwrap it; otherwise use the payload directly (excluding page_content)
            if isinstance(raw_metadata, dict) and raw_metadata:
                metadata = raw_metadata
            else:
                metadata = {k: v for k, v in payload.items() if k not in ["page_content"]}

            return Document(
                content=payload.get("page_content", ""),
                metadata=metadata,
                uid=uid,  # Return original ID, not storage ID
                embedding=None,
                summary=metadata.get("summary", ""),
            )
        except Exception as e:
            logger.error(f"Error retrieving document {uid}: {e}")
            raise

    async def get_by_ids(self, index_name: str, ids: List[str]) -> List[Document]:
        """Get documents by their IDs"""
        if not ids:
            logger.warning("No IDs provided for Qdrant get operation.")
            return []

        try:
            # Convert IDs for storage lookup
            storage_ids = self._convert_ids_for_storage(ids)

            points = await run_in_executor(
                None,
                self.client.retrieve,
                collection_name=index_name,
                ids=storage_ids,  # Keep order, don't deduplicate
                with_payload=True,
                with_vectors=False,
            )

            # Create mapping from storage ID to point
            storage_id_to_point = {str(point.id): point for point in points}

            documents = []
            # Maintain the original order
            for i, original_id in enumerate(ids):
                storage_id = storage_ids[i]
                point = storage_id_to_point.get(storage_id)

                if point is None:
                    logger.warning(f"Document with ID {original_id} not found")
                    continue

                payload = point.payload or {}

                # Extract metadata - LangChain stores our metadata nested under 'metadata' key
                raw_metadata = payload.get("metadata", {})

                # If metadata is nested, unwrap it; otherwise use the payload directly (excluding page_content)
                if isinstance(raw_metadata, dict) and raw_metadata:
                    metadata = raw_metadata
                else:
                    metadata = {k: v for k, v in payload.items() if k not in ["page_content"]}

                doc = Document(
                    content=payload.get("page_content", ""),
                    metadata=metadata,
                    uid=original_id,  # Use original ID, not storage ID
                    embedding=None,
                    summary=metadata.get("summary", ""),
                )
                documents.append(doc)

            return documents
        except Exception as e:
            logger.error(f"Error retrieving documents by IDs: {e}")
            return []

    async def delete(self, index_name: str, ids: List[str]) -> bool:
        """Delete documents by their IDs"""
        if not ids:
            logger.warning("No IDs provided for deletion.")
            return False

        try:
            # Convert IDs for storage
            storage_ids = self._convert_ids_for_storage(ids)
            await self._qdrant_store_for_index(index_name).adelete(ids=list(set(storage_ids)))
            return True
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False

    def _is_metadata_condition_format(self, metadata_filter: Dict[str, Any]) -> bool:
        """
        判断是否为MetadataCondition格式

        Args:
            metadata_filter: 待检查的过滤条件

        Returns:
            True如果是MetadataCondition格式，False如果是简单格式
        """
        # MetadataCondition格式必须包含conditions字段
        if "conditions" in metadata_filter:
            conditions = metadata_filter.get("conditions", [])
            if isinstance(conditions, list) and len(conditions) > 0:
                # 检查第一个条件是否包含MetadataConditionItem的必需字段
                first_condition = conditions[0]
                if isinstance(first_condition, dict):
                    return "name" in first_condition and "comparison_operator" in first_condition
        return False

    def _build_metatda_filter_query(self, metadata_filter: Dict[str, Any]) -> Any:
        if not metadata_filter:
            return None

        # Convert MetadataCondition format to simple format for now
        if self._is_metadata_condition_format(metadata_filter):
            # Convert to simple format
            simple_filter = {}
            conditions = metadata_filter.get("conditions", [])
            for condition in conditions:
                names = condition.get("name", [])
                operator = condition.get("comparison_operator", "")
                value = condition.get("value", "")

                # For now, only handle 'is' operator and single field names
                if operator.lower() == "is" and len(names) == 1:
                    simple_filter[names[0]] = value

            # Use the simple format filter logic
            metadata_filter = simple_filter

        conditions = []
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        for key, value in metadata_filter.items():
            # Add 'metadata.' prefix for nested metadata fields stored by LangChain
            field_key = f"metadata.{key}"

            if isinstance(value, list):
                # Handle list values - create OR conditions for each item
                list_conditions = []
                for v in value:
                    list_conditions.append(FieldCondition(key=field_key, match=MatchValue(value=v)))
                if list_conditions:
                    # If multiple values, use should (OR)
                    if len(list_conditions) > 1:
                        conditions.append(Filter(should=list_conditions))
                    else:
                        conditions.extend(list_conditions)
            else:
                conditions.append(FieldCondition(key=field_key, match=MatchValue(value=value)))
        return Filter(must=conditions)

    async def select_on_metadata(self, index_name: str, metadata_filter: Dict[str, Any]) -> List[Document]:
        """Select documents based on metadata filter"""

        try:
            # Build Qdrant filter

            if not metadata_filter:
                return []
            filter_obj = self._build_metatda_filter_query(metadata_filter)
            scroll_result = await run_in_executor(
                None,
                self.client.scroll,
                collection_name=index_name,
                scroll_filter=filter_obj,
                limit=10000,  # Adjust as needed
                with_payload=True,
                with_vectors=True,
            )

            points = scroll_result[0]
            documents = []

            for point in points:
                payload = point.payload or {}

                # Extract metadata - LangChain stores our metadata nested under 'metadata' key
                raw_metadata = payload.get("metadata", {})

                # If metadata is nested, unwrap it; otherwise use the payload directly (excluding page_content)
                if isinstance(raw_metadata, dict) and raw_metadata:
                    metadata = raw_metadata
                else:
                    metadata = {k: v for k, v in payload.items() if k not in ["page_content"]}

                # Convert storage ID back to original format
                original_id = self._convert_id_from_storage(str(point.id))

                doc = Document(
                    content=payload.get("page_content", ""),
                    metadata=metadata,
                    uid=original_id,  # Use original ID format
                    embedding=None,
                    summary=metadata.get("summary", ""),
                )
                documents.append(doc)

            return documents

        except Exception as e:
            logger.error(f"Error in metadata selection: {e}")
            return []

    async def similarity_search_by_vector(
        self, index_name: str, embedding: List[float], k: int, filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """Perform similarity search using vector embedding"""
        try:
            filter_query = None
            if filter:
                filter_query = self._build_metatda_filter_query(filter)

            results: List[Tuple[LangchainDocument, float]] = await run_in_executor(
                None,
                self._qdrant_store_for_index(index_name).similarity_search_by_vector,
                embedding,
                k,
                filter_query,
            )

            documents = []
            for langchain_doc, score in results:
                metadata = langchain_doc.metadata or {}
                doc = Document(
                    content=langchain_doc.page_content,
                    metadata=metadata,
                    uid=getattr(langchain_doc, "id", None),
                    summary=metadata.get("summary", ""),
                )
                documents.append((doc, score))

            return documents

        except Exception as e:
            logger.error(f"Error in similarity search by vector: {e}")
            return []

    async def asimilar_search_with_scores(
        self, index_name: str, query: str, k: int, filter: Optional[Dict[str, str]] = None
    ) -> List[Tuple[Document, float]]:
        """Perform similarity search using query string"""
        try:
            if filter:
                filter = self._build_metatda_filter_query(filter)
            results = await run_in_executor(
                None, self._qdrant_store_for_index(index_name).similarity_search_with_score, query, k, filter
            )

            documents = []
            for langchain_doc, score in results:
                metadata = langchain_doc.metadata or {}
                doc = Document(
                    content=langchain_doc.page_content,
                    metadata=metadata,
                    uid=getattr(langchain_doc, "id", None),
                    summary=metadata.get("summary", ""),
                )
                # Convert distance to similarity score (higher is better)
                similarity_score = 1 / (1 + score) if score > 0 else 1.0
                documents.append((doc, similarity_score))

            return documents

        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []

    def get_vector_store(self, index_name: str) -> VectorStore:
        """Get vector store for the given index"""
        return self.create_index(index_name)

    @property
    def persist_directory(self) -> Path:
        return self._persist_directory

    def _is_snowflake_id(self, id_str: str) -> bool:
        """
        检查字符串是否是雪花ID格式（纯数字字符串）

        Args:
            id_str: 要检查的ID字符串

        Returns:
            bool: 如果是雪花ID格式返回True，否则返回False
        """
        try:
            # 雪花ID是纯数字字符串，长度通常大于等于1
            # 但为了区分真正的雪花ID，我们设置最小长度为3
            return id_str.isdigit() and len(id_str) >= 3
        except (AttributeError, TypeError):
            return False

    def _is_uuid_format(self, id_str: str) -> bool:
        """
        检查字符串是否是UUID格式

        Args:
            id_str: 要检查的ID字符串

        Returns:
            bool: 如果是UUID格式返回True，否则返回False
        """
        try:
            uuid.UUID(id_str)
            return True
        except (ValueError, TypeError):
            return False

    def _snowflake_to_uuid(self, snowflake_id: str) -> str:
        """
        将雪花ID转换为UUID格式用于Qdrant存储

        Args:
            snowflake_id: 雪花ID字符串

        Returns:
            str: UUID格式的字符串
        """
        if self._is_uuid_format(snowflake_id):
            # 如果已经是UUID格式，直接返回
            return snowflake_id

        if not self._is_snowflake_id(snowflake_id):
            # 如果不是雪花ID，也不是UUID，则使用MD5哈希生成UUID
            hash_bytes = hashlib.md5(snowflake_id.encode("utf-8")).digest()
            return str(uuid.UUID(bytes=hash_bytes))

        # 将雪花ID转换为UUID
        # 使用确定性的方法：将雪花ID转换为128位整数，然后生成UUID
        try:
            # 将雪花ID转换为整数
            snowflake_int = int(snowflake_id)

            # 如果雪花ID超过128位，使用哈希
            if snowflake_int.bit_length() > 128:
                hash_bytes = hashlib.md5(snowflake_id.encode("utf-8")).digest()
                return str(uuid.UUID(bytes=hash_bytes))

            # 将64位雪花ID扩展到128位
            # 将雪花ID放在低64位，高64位用固定前缀填充
            uuid_int = (0x550E8400E29B41D4 << 64) | snowflake_int

            return str(uuid.UUID(int=uuid_int))
        except (ValueError, OverflowError):
            # 如果转换失败，使用MD5哈希
            hash_bytes = hashlib.md5(snowflake_id.encode("utf-8")).digest()
            return str(uuid.UUID(bytes=hash_bytes))

    def _uuid_to_snowflake(self, uuid_str: str, original_mapping: Optional[Dict[str, str]] = None) -> str:
        """
        将UUID转换回雪花ID

        Args:
            uuid_str: UUID字符串
            original_mapping: 原始映射字典（可选，用于查找原始ID）

        Returns:
            str: 雪花ID字符串，如果无法转换则返回原UUID
        """
        if not self._is_uuid_format(uuid_str):
            # 如果不是UUID格式，直接返回
            return uuid_str

        try:
            uuid_obj = uuid.UUID(uuid_str)
            uuid_int = uuid_obj.int

            # 检查是否是由雪花ID转换而来的UUID（高64位是固定前缀）
            high_64 = uuid_int >> 64
            if high_64 == 0x550E8400E29B41D4:
                # 提取低64位作为雪花ID
                snowflake_int = uuid_int & 0xFFFFFFFFFFFFFFFF
                return str(snowflake_int)

            # 如果不是标准转换格式，可能是通过哈希生成的，无法反向转换
            # 在这种情况下，如果有原始映射，使用映射；否则返回UUID
            if original_mapping:
                for original_id, mapped_uuid in original_mapping.items():
                    if mapped_uuid == uuid_str:
                        return original_id

            return uuid_str
        except (ValueError, TypeError):
            return uuid_str

    def _convert_id_for_storage(self, doc_id: str) -> str:
        """
        将文档ID转换为Qdrant存储格式

        Args:
            doc_id: 原始文档ID

        Returns:
            str: 适合Qdrant存储的ID
        """
        return self._snowflake_to_uuid(doc_id)

    def _convert_id_from_storage(self, storage_id: str) -> str:
        """
        将Qdrant存储的ID转换回原始格式

        Args:
            storage_id: Qdrant存储的ID

        Returns:
            str: 原始格式的ID
        """
        return self._uuid_to_snowflake(storage_id)

    def _convert_ids_for_storage(self, doc_ids: List[str]) -> List[str]:
        """批量转换ID用于存储"""
        return [self._convert_id_for_storage(doc_id) for doc_id in doc_ids]

    def _convert_ids_from_storage(self, storage_ids: List[str]) -> List[str]:
        """批量转换ID从存储格式"""
        return [self._convert_id_from_storage(storage_id) for storage_id in storage_ids]
