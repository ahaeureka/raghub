from abc import ABC, abstractmethod
from typing import List

from deeprag_core.schemas.document import Document
from deeprag_core.schemas.rag_model import RetrieveResultItem


class BaseRAG(ABC):
    """
    Base class for Retrieval-Augmented Generation (RAG) systems.
    """

    @abstractmethod
    def add_documents(self, texts: List[Document]) -> List[Document]:
        raise NotImplementedError("This method should be overridden by subclasses.")

    # @abstractmethod
    # def generate(self, query:str, retrieved_docs:List[Document])->str:
    #     raise NotImplementedError("This method should be overridden by subclasses.")

    # @abstractmethod
    # def update(self, texts:List[Document])->None:
    #     raise NotImplementedError("This method should be overridden by subclasses.")

    # @abstractmethod
    # def delete(self, doc_ids:List[str])->None:
    #     raise NotImplementedError("This method 'delete' should be overridden by subclasses.")

    # @abstractmethod
    def init(self) -> None:
        raise NotImplementedError("This method 'init' should be overridden by subclasses.")

    @abstractmethod
    def retrieve(self, queries: List[str], retrieve_top_k: int = 10) -> List[RetrieveResultItem]:
        raise NotImplementedError("This method 'retrieve' should be overridden by subclasses.")
