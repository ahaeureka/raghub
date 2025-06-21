from abc import abstractmethod
from typing import AsyncIterator, Dict, List

from raghub_core.schemas.chat_response import QAChatResponse
from raghub_core.schemas.document import Document
from raghub_core.schemas.rag_model import RetrieveResultItem
from raghub_core.utils.class_meta import SingletonRegisterMeta


class BaseRAGApp(metaclass=SingletonRegisterMeta):
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

    @abstractmethod
    async def add_documents(self, unique_name: str, texts: List[Document], lang="en") -> List[Document]:
        """
        Add documents to the application.
        Args:
            texts: List of Document objects to add.
            lang: Language of the documents (default is "en").
        Returns:
            List of Document objects that were added.
        """
        raise NotImplementedError("Subclasses should implement this method.")

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

    @abstractmethod
    async def delete(self, unique_name: str, docs_to_delete: List[str]):
        """
        Delete documents from the application.
        Args:
            docs_to_delete: List of document IDs to delete.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    async def QA(
        self, unique_name: str, question: str, retrieve_top_k=5, lang="zh", prompt=None
    ) -> AsyncIterator[QAChatResponse]:
        """
        Perform question answering on the application.
        Args:
            unique_name: Unique name for the index.
            question: The question to answer.
            retrieve_top_k: Number of top results to retrieve (default is 5).
            lang: Language of the question (default is "zh").
            prompt: Optional prompt for the LLM.
        Returns:
            AsyncGenerator of QAChatResponse containing the answers and their scores.
        """
        raise NotImplementedError("Subclasses should implement this method.")
