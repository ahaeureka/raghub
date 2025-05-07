from abc import ABCMeta, abstractmethod
from typing import List

from deeprag_core.schemas.document import Document
from deeprag_core.schemas.rag_model import RetrieveResultItem


class BaseApp(metaclass=ABCMeta):
    @abstractmethod
    def init(self):
        """
        Initialize the application.
        This method should be overridden by subclasses to provide specific initialization logic.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def add_documents(self, texts: List[Document], lang="en") -> List[Document]:
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
    def retrieve(self, queries: List[str], retrieve_top_k=10, lang="en") -> List[RetrieveResultItem]:
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
    def delete(self, docs_to_delete: List[str]):
        """
        Delete documents from the application.
        Args:
            docs_to_delete: List of document IDs to delete.
        """
        raise NotImplementedError("Subclasses should implement this method.")
