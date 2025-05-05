from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from deeprag_core.embedding.base_embedding import BaseEmbedding
from deeprag_core.schemas.document import Document
from deeprag_core.storage.embedding_adapter import LangchainEmbeddings
from deeprag_core.storage.vector import VectorStorage
from deeprag_core.utils.file.project import ProjectHelper
from loguru import logger

# class LangchainEmbeddings(Embeddings):
#     def __init__(self, embedder: BaseEmbedding):
#         self.embedder = embedder

#     def embed_documents(self, documents: List[str]) -> List[List[float]]:
#         ret = self.embedder.encode(documents).tolist()
#         return ret

#     def embed_query(self, query: str) -> List[float]:
#         ret = self.embedder.encode_query([query]).tolist()
#         return ret


class ChromaDBVectorStorage(VectorStorage):
    name = "chromadb"

    def __init__(
        self,
        embedder: BaseEmbedding,  # Assuming BaseEmbedding is imported from the correct module
        collection_name: str,  # Assuming collection_name is a string
        persist_directory: Path = ProjectHelper.get_project_root() / "cache/chroma_db",
    ):
        try:
            from langchain_chroma import Chroma

            self.persist_directory = (
                persist_directory if isinstance(persist_directory, Path) else Path(persist_directory)
            )

            self._client: Chroma = None
            self._embedder: BaseEmbedding = embedder
            logger.debug(f"ChromaDBVectorStorage: {self._embedder}")
            self._collection_name: str = collection_name
        except ImportError:
            raise ImportError("ChromaDB is not installed. Please install it using `pip install chromadb`.")

    def _embed(self, text: str) -> List[float]:
        # Implement the logic to embed a single text
        return self._embedder.encode([text]).tolist()[0]

    def init(self):
        from langchain_chroma import Chroma

        # from chromadb import Chroma
        self._embedder.init()
        # Initialize ChromaDB client
        self._client = Chroma(
            collection_name=self._collection_name,  # 替换为实际集合名称
            embedding_function=LangchainEmbeddings(self._embedder),  # 替换为实际嵌入函数
            persist_directory=self.persist_directory.as_posix(),
            collection_metadata={"hnsw:space": "cosine"},
        )  # 替换为实际路径

    def add_documents(self, texts: List[Document]) -> List[Document]:
        # Implement the logic to add documents to ChromaDB
        # For now, we'll just return the documents as is
        # Assuming Document has attributes: content, metadata, uid
        if not self._client:
            self.init()

        self._client.add_texts(
            texts=[doc.content for doc in texts],
            metadatas=[doc.metadata for doc in texts],
            ids=[doc.uid for doc in texts],
        )
        return texts

    def get(self, uid: str) -> Document:
        # Implement the logic to retrieve documents based on a query from ChromaDB
        if not self._client:
            self.init()
        results = self._client.get(ids=[uid], include=["documents", "metadatas", "embeddings"])
        return self._build_docs(results)[0]

    def get_by_ids(self, ids: List[str]) -> List[Document]:
        # Implement the logic to retrieve documents by their IDs from ChromaDB
        if not self._client:
            self.init()
        results = self._client.get(ids=ids, include=["documents", "metadatas", "embeddings"])
        return self._build_docs(results)

    def delete(self, ids: List[str]) -> bool:
        # Implement the logic to delete documents by their IDs from ChromaDB
        if not self._client:
            self.init()
        self._client.delete(ids=ids)
        return True

    def select_on_metadata(self, metadata_filter: Dict[str, Any]) -> List[Document]:
        # Implement the logic to retrieve documents by metadata filter from ChromaDB
        if not self._client:
            self.init()
        # {"$and": [{"color": "red"}, {"price": 4.20}]}
        query = metadata_filter
        if len(list(metadata_filter.keys())) > 1:
            query = {"$and": [{key: value} for key, value in metadata_filter.items()]}
        results = self._client.get(where=query, include=["documents", "metadatas", "embeddings"])
        return self._build_docs(results)

    def _build_docs(self, results: Dict[str, Any]) -> List[Document]:
        documents = []

        if not results["documents"]:
            return []
        for index, result in enumerate(results["documents"]):
            doc = Document(
                content=result,
                metadata=results["metadatas"][index],
                uid=results["ids"][index],
                embedding=results["embeddings"][index],
            )
            documents.append(doc)
        return documents

    def similarity_search_by_vector(
        self, embedding: List[float], k: int, filter: Optional[Dict[str, str]] = None
    ) -> List[Tuple[Document, float]]:
        # Implement the logic to perform cosine similarity search in ChromaDB
        if not self._client:
            self.init()
        # query_embedding = self._embedding_function(query)
        results = self._client.similarity_search_by_vector_with_relevance_scores(
            embedding=embedding, k=k, filter=filter
        )
        documents = []
        for result, score in results:
            doc = Document(content=result.page_content, metadata=result.metadata, uid=result.id)
            documents.append((doc, score))
        # This line is not correct and should be removed or corrected
        return documents
