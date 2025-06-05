import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document as LangchainDocument
from langchain_core.runnables.config import run_in_executor
from langchain_core.vectorstores import VectorStore
from loguru import logger
from pydantic import BaseModel
from raghub_core.embedding.base_embedding import BaseEmbedding
from raghub_core.schemas.document import Document
from raghub_core.storage.embedding_adapter import LangchainEmbeddings
from raghub_core.storage.vector import VectorStorage
from raghub_core.utils.file.project import ProjectHelper


class ChromaDBVectorStorage(VectorStorage):
    name = "chromadb"

    def __init__(
        self,
        embedder: BaseEmbedding,  # Assuming BaseEmbedding is imported from the correct module
        persist_directory: Path = ProjectHelper.get_project_root() / "cache/chroma_db",
    ):
        try:
            from chromadb import PersistentClient

            self.persist_directory = (
                persist_directory if isinstance(persist_directory, Path) else Path(persist_directory)
            )

            self._client: PersistentClient = None
            # self._vector_store_type = TypeVar("Chroma", bound=Chroma)
            self._embedder: BaseEmbedding = embedder
            logger.debug(f"ChromaDBVectorStorage: {self._embedder}")
            # self._collection_name: str = collection_name
        except ImportError:
            raise ImportError("ChromaDB is not installed. Please install it using `pip install chromadb`.")

    def _embed(self, text: str) -> List[float]:
        # Implement the logic to embed a single text
        return self._embedder.encode([text]).tolist()[0]

    async def init(self):
        from chromadb import PersistentClient

        # from chromadb import Chroma
        self._embedder.init()
        await asyncio.sleep(0.01)  # Yield control to the event loop
        self._client = PersistentClient(
            path=self.persist_directory.as_posix(),
        )

    def _chromadb_store_for_index(self, index_name: str) -> VectorStore:
        # Implement the logic to create or get a ChromaDB store for the given index name
        # For now, we'll just return the client

        if not self._client:
            self.init()
        from langchain_chroma import Chroma

        return Chroma(
            collection_name=index_name,
            embedding_function=LangchainEmbeddings(self._embedder),
            persist_directory=self.persist_directory.as_posix(),
            client=self._client,
        )

    def _metadata_values_to_string(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert metadata values to strings.
        """
        new_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, list):
                new_metadata[key] = json.dumps(value, ensure_ascii=False)
            elif isinstance(value, dict):
                new_metadata[key] = json.dumps(value, ensure_ascii=False)
            elif issubclass(value, BaseModel):
                new_metadata[key] = value.model_dump_json()
            else:
                new_metadata[key] = value
        return new_metadata

    def _string_metadata_to_values(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert metadata values from strings.
        """
        new_metadata: Dict[str, Any] = {}
        if not metadata:
            return new_metadata
        for key, value in metadata.items():
            if isinstance(value, str):
                try:
                    new_metadata[key] = json.loads(value)
                except json.JSONDecodeError:
                    new_metadata[key] = value
            else:
                new_metadata[key] = value
        return new_metadata

    async def add_documents(self, index_name: str, texts: List[Document]) -> List[Document]:
        if not self._client:
            await self.init()
        for text in texts:
            text.metadata["summary"] = text.summary
        ids = await self._chromadb_store_for_index(index_name).aadd_texts(
            texts=[doc.content for doc in texts],
            metadatas=[self._metadata_values_to_string(doc.metadata) for doc in texts],
            ids=[doc.uid for doc in texts],
        )
        logger.debug(f"Added {ids} documents to index '{index_name}'")
        return await self.get_by_ids(index_name, ids)

    async def get(self, index_name: str, uid: str) -> Document:
        # Implement the logic to retrieve documents based on a query from ChromaDB
        if not self._client:
            await self.init()
        results = await run_in_executor(
            None,
            self._chromadb_store_for_index(index_name).get,
            ids=[uid],
            include=["documents", "metadatas", "embeddings"],
        )
        return self._build_docs(results)[0]

    async def get_by_ids(self, index_name: str, ids: List[str]) -> List[Document]:
        # Implement the logic to retrieve documents by their IDs from ChromaDB
        if not self._client:
            await self.init()
        results = await run_in_executor(
            None,
            self._chromadb_store_for_index(index_name).get,
            ids=ids,
            include=["documents", "metadatas", "embeddings"],
        )
        return self._build_docs(results)

    async def delete(self, index_name: str, ids: List[str]) -> bool:
        # Implement the logic to delete documents by their IDs from ChromaDB
        if not self._client:
            await self.init()
        await self._chromadb_store_for_index(index_name).adelete(ids=list(set(ids)))
        return True

    async def select_on_metadata(self, index_name: str, metadata_filter: Dict[str, Any]) -> List[Document]:
        # Implement the logic to retrieve documents by metadata filter from ChromaDB
        if not self._client:
            await self.init()
        # {"$and": [{"color": "red"}, {"price": 4.20}]}
        query = metadata_filter
        if len(list(metadata_filter.keys())) > 1:
            query = {"$and": [{key: value} for key, value in metadata_filter.items()]}
        logger.debug(f"select_on_metadata: {query}")
        results = await run_in_executor(
            None,
            self._chromadb_store_for_index(index_name).get,
            where=query,
            include=["documents", "metadatas", "embeddings"],
        )
        return self._build_docs(results)

    def _build_docs(self, results: Dict[str, Any]) -> List[Document]:
        documents = []

        if not results["documents"]:
            return []
        for index, result in enumerate(results["documents"]):
            doc = Document(
                content=result,
                metadata=self._string_metadata_to_values(results["metadatas"][index]),
                uid=results["ids"][index],
                embedding=results["embeddings"][index],
            )
            doc.summary = doc.metadata.get("summary", "")
            documents.append(doc)
        return documents

    async def similarity_search_by_vector(
        self, index_name: str, embedding: List[float], k: int, filter: Optional[Dict[str, str]] = None
    ) -> List[Tuple[Document, float]]:
        # Implement the logic to perform cosine similarity search in ChromaDB
        if not self._client:
            await self.init()
        # query_embedding = self._embedding_function(query)
        results: List[LangchainDocument] = await run_in_executor(
            None,
            self._chromadb_store_for_index(index_name).similarity_search_by_vector_with_relevance_scores,
            embedding=embedding,
            k=k,
            filter=filter,
        )
        documents = []
        for result, score in results:
            doc = Document(content=result.page_content, metadata=result.metadata, uid=result.id)
            documents.append((doc, score))
        # This line is not correct and should be removed or corrected
        return documents

    async def asimilar_search_with_scores(
        self, index_name: str, query: str, k: int, filter: Optional[Dict[str, str]] = None
    ) -> List[Tuple[Document, float]]:
        # Implement the logic to perform cosine similarity search in ChromaDB
        if not self._client:
            await self.init()
        results: List[Tuple[LangchainDocument, float]] = await run_in_executor(
            None, self._chromadb_store_for_index(index_name).similarity_search_with_score, query, k=k, filter=filter
        )
        documents = []
        for result, score in results:
            doc = Document(
                content=result.page_content,
                metadata=result.metadata,
                summary=result.metadata.get("summary", ""),
                uid=result.id,
            )
            documents.append((doc, 1 / (1 + score)))
        # This line is not correct and should be removed or corrected
        return documents
