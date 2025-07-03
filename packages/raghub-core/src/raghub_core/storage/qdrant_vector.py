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
            self._persist_directory: Path = (
                persist_directory if isinstance(persist_directory, Path) else Path(persist_directory)
            )
            self._vector_stores: Dict[str, VectorStore] = {}
            self._metadata_filter_convert: QdrantQueryConverter = QdrantQueryConverter()
            # Add a lock for thread-safe operations
            self._client_lock = asyncio.Lock()
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

                # Configure client with thread safety for asyncio environment
                self._client = QdrantClient(
                    path=self._persist_directory.as_posix(),
                    force_disable_check_same_thread=True,  # Essential for asyncio/threading compatibility
                )
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
        # await asyncio.sleep(0.01)  # Yield control to the event loop
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

    def _qdrant_store_for_index(self, index_name: str) -> QdrantVectorStore:
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
        """
        Add documents to the index with thread-safe operations.

        Supports namespace-prefixed document IDs such as:
        - "entity-12345", "doc-67890", "fact-abcdef"
        - "namespace:some-id", "user_profile:user_12345"

        The original doc_id is preserved in metadata['_original_id'] for recovery.
        """
        # Add summary and original ID to metadata for recovery
        for text in texts:
            if text.metadata is None:
                text.metadata = {}
            text.metadata["summary"] = text.summary
            # Store the original ID in metadata for recovery during retrieval
            text.metadata["_original_id"] = text.uid

        # Convert IDs for Qdrant storage (namespace-prefixed IDs -> UUID format)
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

        # Use lock to protect concurrent access to Qdrant client
        async with self._client_lock:
            ids = await self._qdrant_store_for_index(index_name).aadd_documents(
                documents=langchain_docs, ids=storage_ids
            )

        logger.debug(f"Added {len(ids)} documents to index '{index_name}'")
        # Return documents with their original IDs in the same order
        original_ids = [doc.uid for doc in texts]
        return await self.get_by_ids(index_name, original_ids)

    async def get(self, index_name: str, uid: str) -> Document:
        """
        Get a document by its ID with thread-safe operations.

        Supports namespace-prefixed IDs by converting them to storage format
        and recovering the original ID from metadata.
        """
        try:
            # Convert ID for storage lookup
            storage_id = self._convert_id_for_storage(uid)

            # Use lock to protect concurrent access to Qdrant client
            async with self._client_lock:
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

            # Verify the original ID matches what we expect
            stored_original_id = metadata.get("_original_id")
            if stored_original_id and stored_original_id != uid:
                logger.warning(f"ID mismatch: requested {uid}, stored {stored_original_id}")

            return Document(
                content=payload.get("page_content", ""),
                metadata=metadata,
                uid=uid,  # Return requested ID (original format)
                embedding=point.vector,
                summary=metadata.get("summary", ""),
            )
        except Exception as e:
            logger.error(f"Error retrieving document {uid}: {e}")
            raise

    async def get_by_ids(self, index_name: str, ids: List[str]) -> List[Document]:
        """Get documents by their IDs with thread-safe operations"""
        if not ids:
            logger.warning("No IDs provided for Qdrant get operation.")
            return []

        try:
            # Convert IDs for storage lookup
            storage_ids = self._convert_ids_for_storage(ids)
            logger.debug(f"Retrieving documents id {ids} with storage IDs: {storage_ids}")
            # Use lock to protect concurrent access to Qdrant client
            async with self._client_lock:
                points = await run_in_executor(
                    None,
                    self.client.retrieve,
                    collection_name=index_name,
                    ids=storage_ids,  # Keep order, don't deduplicate
                    with_payload=True,
                    with_vectors=True,
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
                    embedding=point.vector,
                    summary=metadata.get("summary", ""),
                )

                documents.append(doc)
            logger.debug(f"Retrieved {len(documents)} documents by IDs: {ids}")
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
        """Select documents based on metadata filter with thread-safe operations"""

        try:
            # Build Qdrant filter

            if not metadata_filter:
                return []
            filter_obj = self._build_metatda_filter_query(metadata_filter)

            # Use lock to protect concurrent access to Qdrant client
            async with self._client_lock:
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

                # Recover original ID from metadata, fallback to storage ID conversion
                original_id = metadata.get("_original_id")
                if not original_id:
                    original_id = self._convert_id_from_storage(str(point.id))

                doc = Document(
                    content=payload.get("page_content", ""),
                    metadata=metadata,
                    uid=original_id,  # Use recovered original ID
                    embedding=point.vector,
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
                self._qdrant_store_for_index(index_name).max_marginal_relevance_search_with_score_by_vector,
                embedding,
                k,
                filter=filter_query,
            )

            documents = []
            for langchain_doc, score in results:
                metadata = langchain_doc.metadata or {}
                # Recover original ID from metadata, fallback to ID conversion
                original_id = metadata.get("_original_id")
                if not original_id:
                    original_id = self._convert_id_from_storage(metadata.get("_id", ""))

                doc = Document(
                    content=langchain_doc.page_content,
                    metadata=metadata,
                    uid=original_id,
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
                # Recover original ID from metadata, fallback to ID conversion
                original_id = metadata.get("_original_id")
                if not original_id:
                    original_id = self._convert_id_from_storage(str(metadata.get("_id", "")))

                doc = Document(
                    content=langchain_doc.page_content,
                    metadata=metadata,
                    uid=original_id,
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

    def _is_mdhash_id(self, id_str: str) -> bool:
        """
        Check if the string is an MD5 hash ID format generated by compute_mdhash_id.

        Formats:
        - Plain MD5 hash: "a1b2c3d4e5f6..." (32 hex chars)
        - Prefixed MD5 hash: "entity-a1b2c3d4e5f6..." (prefix + "-" + 32 hex chars)

        Args:
            id_str: The ID string to check

        Returns:
            bool: True if it's an MD5 hash ID format, False otherwise
        """
        try:
            if not id_str:
                return False

            # Check if it's a plain MD5 hash (32 hex characters)
            if len(id_str) == 32 and all(c in "0123456789abcdef" for c in id_str.lower()):
                return True

            # Check if it's a prefixed MD5 hash (prefix-hash format)
            if "-" in id_str:
                parts = id_str.split("-", 1)  # Split on first dash only
                if len(parts) == 2:
                    prefix, hash_part = parts
                    # Check if the hash part is a valid MD5 hash (exactly 32 hex chars)
                    if len(hash_part) == 32 and all(c in "0123456789abcdef" for c in hash_part.lower()):
                        return True

            return False
        except (AttributeError, TypeError):
            return False

    def _is_uuid_format(self, id_str: str) -> bool:
        """
        Check if the string is in UUID format.

        Args:
            id_str: The ID string to check

        Returns:
            bool: True if it's UUID format, False otherwise
        """
        try:
            uuid.UUID(id_str)
            return True
        except (ValueError, TypeError):
            # Normal case when string is not UUID format - no need to log
            return False

    def _mdhash_to_uuid(self, mdhash_id: str) -> str:
        """
        Convert MD5 hash ID to UUID format for Qdrant storage.

        Args:
            mdhash_id: MD5 hash ID string (with or without prefix)

        Returns:
            str: UUID format string
        """
        if self._is_uuid_format(mdhash_id):
            # If already UUID format, return directly
            return mdhash_id

        # For MD5 hash IDs, we can use the hash directly to create a UUID
        if self._is_mdhash_id(mdhash_id):
            if "-" in mdhash_id:
                # Extract the hash part from prefixed format
                hash_part = mdhash_id.split("-", 1)[1]
            else:
                # Plain MD5 hash
                hash_part = mdhash_id

            try:
                # Convert MD5 hash to UUID by treating it as hex bytes
                hash_bytes = bytes.fromhex(hash_part)
                return str(uuid.UUID(bytes=hash_bytes))
            except (ValueError, TypeError):
                # Fallback: use the original mdhash_id to generate a new hash
                hash_bytes = hashlib.md5(mdhash_id.encode("utf-8")).digest()
                return str(uuid.UUID(bytes=hash_bytes))

        # For any other format, generate UUID using MD5 hash
        hash_bytes = hashlib.md5(mdhash_id.encode("utf-8")).digest()
        return str(uuid.UUID(bytes=hash_bytes))

    def _uuid_to_mdhash(self, uuid_str: str, original_mapping: Optional[Dict[str, str]] = None) -> str:
        """
        Convert UUID back to MD5 hash ID format.

        Args:
            uuid_str: UUID string
            original_mapping: Original mapping dictionary (optional, for lookup)

        Returns:
            str: MD5 hash ID string, returns original UUID if conversion is not possible
        """
        if not self._is_uuid_format(uuid_str):
            # If not UUID format, return directly
            return uuid_str

        try:
            uuid_obj = uuid.UUID(uuid_str)

            # Try to convert UUID bytes back to hex string (MD5 hash)
            try:
                hex_string = uuid_obj.bytes.hex()
                # Check if this looks like a valid MD5 hash
                if len(hex_string) == 32 and all(c in "0123456789abcdef" for c in hex_string.lower()):
                    return hex_string
            except Exception:
                pass

            # If direct conversion fails, check if we have original mapping
            if original_mapping:
                for original_id, mapped_uuid in original_mapping.items():
                    if mapped_uuid == uuid_str:
                        return original_id

            # As fallback, return the UUID string
            return uuid_str

        except (ValueError, TypeError):
            logger.warning(f"Invalid UUID format: {uuid_str}, returning original UUID.")
            return uuid_str

    def _convert_id_for_storage(self, doc_id: str) -> str:
        """
        Convert document ID to Qdrant storage format.

        Handles MD5 hash IDs generated by compute_mdhash_id, such as:
        - Plain MD5 hash: "a1b2c3d4e5f6..." -> UUID format
        - Prefixed MD5 hash: "entity-a1b2c3d4e5f6..." -> UUID format
        - Other formats: "namespace:some-id" -> UUID format via MD5 hash

        Args:
            doc_id: Original document ID, may be MD5 hash with/without prefix

        Returns:
            str: UUID format ID suitable for Qdrant storage
        """
        if not doc_id:
            return doc_id

        # Use the MD5 hash to UUID conversion method
        return self._mdhash_to_uuid(doc_id)

    def _convert_id_from_storage(self, storage_id: str) -> str:
        """
        Convert Qdrant storage ID back to original format.

        For MD5 hash IDs created by compute_mdhash_id, we can attempt to convert
        the UUID back to the original hex format. However, since prefixes are lost
        during UUID conversion, the primary recovery method is through
        metadata['_original_id'].

        Args:
            storage_id: UUID format ID from Qdrant storage

        Returns:
            str: Original format ID (returns storage_id if recovery is not possible)
        """
        if not storage_id:
            return storage_id

        # If not UUID format, return directly
        if not self._is_uuid_format(storage_id):
            return storage_id

        # Try to convert back to MD5 hash format
        return self._uuid_to_mdhash(storage_id)

    def _convert_ids_for_storage(self, doc_ids: List[str]) -> List[str]:
        """批量转换ID用于存储"""
        return [self._convert_id_for_storage(doc_id) for doc_id in doc_ids]

    def _convert_ids_from_storage(self, storage_ids: List[str]) -> List[str]:
        """批量转换ID从存储格式"""
        return [self._convert_id_from_storage(storage_id) for storage_id in storage_ids]
