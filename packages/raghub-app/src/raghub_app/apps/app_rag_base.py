import traceback
from abc import abstractmethod
from typing import Any, AsyncIterator, Dict, List, Optional

from loguru import logger
from raghub_core.chat.base_chat import BaseChat
from raghub_core.rag.base_rag import BaseRAG
from raghub_core.rerank.base_rerank import BaseRerank
from raghub_core.schemas.chat_response import QAChatResponse
from raghub_core.schemas.document import Document
from raghub_core.schemas.rag_model import RetrieveResultItem
from raghub_core.utils.class_meta import SingletonRegisterMeta

from raghub_app.app_schemas.history_context import HistoryContext


class BaseRAGApp(metaclass=SingletonRegisterMeta):
    def __init__(self, app: BaseRAG):
        self.app = app

    @abstractmethod
    async def init(self):
        """
        Initialize the application.
        This method should be overridden by subclasses to provide specific initialization logic.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    async def create(self, unique_name: str):
        """
        Create an index for the application.
        Args:
            unique_name: Unique name for the index.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    async def add_documents(self, unique_name: str, texts: List[Document], lang="en") -> List[Document]:
        """
        Add documents to the application.
        Args:
            texts: List of Document objects to add.
            lang: Language of the documents (default is "en").
        Returns:
            List of Document objects that were added.
        """
        return await self.app.add_documents(unique_name, texts)

    @abstractmethod
    async def retrieve(
        self, unique_name: str, queries: List[str], retrieve_top_k=10, lang="en"
    ) -> Dict[str, List[RetrieveResultItem]]:
        """
        Retrieve documents based on queries.
        Args:
            queries: List of query strings.
            retrieve_top_k: Number of top results to retrieve (default is 10).
            lang: Language of the queries (default is "en").
        Returns:
            List of RetrieveResultItem objects containing the retrieved documents and their scores.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    async def hybrid_search(
        self,
        unique_name: str,
        queries: List[str],
        reranker: Optional[BaseRerank] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.6,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[RetrieveResultItem]]:
        """
        Perform hybrid search combining graph-based and keyword-based retrieval.
        Args:
            unique_name: Name of the index to use for retrieval.
            queries: List of query strings.
            reranker: Optional reranker instance to use for reranking results.
            similarity_threshold: Threshold for similarity scores to consider a document relevant.
            filter: Optional filter criteria for retrieval.
        Returns:
            Dict[str, List[RetrieveResultItem]]: Dictionary with queries as keys and
            lists of RetrieveResultItem as values.
        """
        return await self.app.hybrid_retrieve(unique_name, queries, reranker, top_k, similarity_threshold, filter)

    async def delete(self, unique_name: str, docs_to_delete: List[str]):
        """
        Delete documents from the GraphRAG application.
        Args:
            unique_name: Unique name of the index.
            docs_to_delete: List of document IDs to delete.
        """
        await self.app.delete(unique_name, docs_to_delete)

    async def chat(
        self,
        unique_name: str,
        question: str,
        histories: Optional[HistoryContext] = None,
        retrieve_top_k=5,
        similarity_threshold=0.6,
        lang="zh",
        prompt=None,
        llm: Optional[BaseChat] = None,
    ) -> AsyncIterator[QAChatResponse]:
        """
        Perform question answering on the GraphRAG application.
        Args:
            unique_name: Unique name of the index.
            question: The question to answer.
            retrieve_top_k: Number of top results to retrieve (default is 5).
            lang: Language of the question (default is "zh").
            prompt: Optional prompt for the LLM.
            llm: Optional LLM instance to use for answering the question. If not provided, the default LLM will be used.
        Returns:
            AsyncIterator[QAChatResponse]

        """
        try:
            history_context = ""
            if histories and histories.items:
                history_context = "\n".join([f"{item.role}: {item.content}\n" for item in histories.items])

            async for r in self.app.qa(
                unique_name, question, history_context, retrieve_top_k, similarity_threshold, prompt, llm
            ):
                yield r
        except Exception as e:
            logger.error(f"Error in QA: {e}:{traceback.format_exc()}")
            yield QAChatResponse(
                question=question,
                answer="",
                error=str(e),
                documents=[],
                source_documents=[],
                metadata={},
            )
