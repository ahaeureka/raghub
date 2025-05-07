from abc import abstractmethod
from typing import List, Optional, Type, TypeVar

from raghub_core.utils.class_meta import SingletonRegisterMeta
from sqlmodel import SQLModel

TSQLModel = TypeVar("TSQLModel", bound=SQLModel)


class SearchEngineStorage(metaclass=SingletonRegisterMeta):
    """
    SearchEngineStorage is a singleton class that manages the storage of search engine data.
    It provides methods to add, retrieve, and delete search engine data.
    """

    @abstractmethod
    def init(self):
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def create_index(self, index_name: str, index_mapping: Optional[dict] = None):
        """
        Create an index in the search engine.
        Args:
            index_name (str): The name of the index to create.
            index_mapping (Optional[dict]): Optional mapping for the index.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def insert_document(self, index_name: str, documents: list[SQLModel]):
        """
        Insert documents into the search engine index.
        Args:
            index_name (str): The name of the index to insert documents into.
            documents (list[SQLModel]): A list of SQLModel objects representing the documents to insert.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def get_documents(
        self, index_name: str, query: dict, model_cls: Type[TSQLModel], filter_deleted: bool = True
    ) -> list[TSQLModel]:
        """
        Retrieve documents from the search engine index based on a query.
        Args:
            index_name (str): The name of the index to query.
            query (dict): The query to execute.
            model_cls (Type[SQLModel]): The SQLModel class representing the document structure.
            filter_deleted (bool): Whether to filter out soft-deleted documents (default: True).
        Returns:
            list[SQLModel]: A list of SQLModel objects representing the retrieved documents.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def delete_documents(self, index_name: str, keys: List[str], soft_deleted: bool = True):
        """
        Delete documents from the search engine index.
        Args:
            index_name (str): The name of the index to delete documents from.
            keys (List[str]): A list of keys representing the documents to delete.
            soft_deleted (bool): Whether to perform a soft delete (default: True).
        """
        raise NotImplementedError("Subclasses should implement this method.")
