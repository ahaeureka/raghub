from abc import abstractmethod
from typing import List

from deeprag_core.schemas.document import Document
from deeprag_core.schemas.rag_model import RetrieveResultItem
from deeprag_core.utils.class_meta import SingletonRegisterMeta


class BaseRAG(metaclass=SingletonRegisterMeta):
    """
    Base class for Retrieval-Augmented Generation (RAG) systems.
    """

    @abstractmethod
    def create(self, index_name: str):
        """
        Create a new index in the RAG system.
        The index name should be unique.
        Args:
            index_name (str): Name of the index to create.
        Returns:
            None
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    def add_documents(self, unique_name: str, texts: List[Document]) -> List[Document]:
        """
        Add documents to the RAG system.
        Args:
            unique_name (str): Name of the index to which documents will be added.
            texts (List[Document]): List of documents to add.
        Returns:
            List[Document]: List of added documents.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    def delete(self, unique_name: str, doc_ids: List[str] | str) -> None:
        """
        Delete documents from the RAG system.
        Args:
            unique_name (str): Name of the index from which documents will be deleted.
            doc_ids (List[str]|str): List of document IDs to delete.
        Returns:
            None
        """
        raise NotImplementedError("This method 'delete' should be overridden by subclasses.")

    # @abstractmethod
    def init(self) -> None:
        raise NotImplementedError("This method 'init' should be overridden by subclasses.")

    @abstractmethod
    def retrieve(self, unique_name: str, queries: List[str], retrieve_top_k: int = 10) -> List[RetrieveResultItem]:
        """
        Retrieve documents based on the provided queries.
        Args:
            unique_name (str): Name of the index to use for retrieval.
            queries (List[str]): List of query strings.
            retrieve_top_k (int): Number of top documents to retrieve.
        Returns:
            List[RetrieveResultItem]: List of retrieved documents and their corresponding scores.
        """
        raise NotImplementedError("This method 'retrieve' should be overridden by subclasses.")
