from abc import abstractmethod
from typing import Dict, List, Optional, Set, Tuple

from deeprag_app.app_schemas.hipporag_models import OpenIEInfo
from deeprag_core.utils.class_meta import SingletonRegisterMeta


class HipporagStorage(metaclass=SingletonRegisterMeta):
    """
    HipporagStorage is a singleton class that manages the storage of OpenIE data.
    It provides methods to add, retrieve, and delete OpenIE data.
    """

    @abstractmethod
    def init(self):
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def save_openie_info(self, openie_info: List[OpenIEInfo]):
        """
        Save OpenIE information to the storage.
        Args:
            openie_info (List[OpenIEInfo]): A list of OpenIEInfo objects to be saved.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def get_openie_info(self, keys: List[str]) -> List[OpenIEInfo]:
        """
        Retrieve OpenIE information from the storage.
        Args:
            keys (List[str]): A list of keys to retrieve OpenIE information.
        Returns:
            List[OpenIEInfo]: A list of OpenIEInfo objects retrieved from the storage.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def delete_openie_info(self, keys: List[str]):
        """
        Delete OpenIE information from the storage.
        Args:
            keys (List[str]): A list of keys to delete OpenIE information.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def set_ent_node_to_chunk_ids(self, ent_node_id: str, ent_node_to_chunk_ids: List[str]):
        """
        Set the mapping of entity node ID to chunk IDs in the cache.
        Args:
            ent_node_id (str): The entity node ID.
            ent_node_to_chunk_ids (List[str]): A list of chunk IDs associated with the entity node ID.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def get_ent_node_to_chunk_ids(self, ent_node_id: str) -> Optional[List[str]] | None:
        """
        Retrieve the mapping of entity node ID to chunk IDs from the cache.
        Args:
            ent_node_id (str): The entity node ID.
        Returns:
            Optional[List[str]]: A list of chunk IDs associated with the entity node ID, or None if not found.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def set_node_to_node_stats(self, from_node_key: str, to_node_key: str, stats: float):
        """
        Set the node-to-node statistics in the cache.
        Args:
            from_node_key (str): The key of the source node.
            to_node_key (str): The key of the target node.
            stats (float): The statistics value to be set.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def get_node_to_node_stats(self, from_node_key: str, to_node_key: str) -> Optional[float]:
        """
        Retrieve the node-to-node statistics from the cache.
        Args:
            from_node_key (str): The key of the source node.
            to_node_key (str): The key of the target node.
        Returns:
            Optional[float]: The statistics value, or 0.0 if not found.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def set_triples_to_docs(self, triples: Dict[str, Set[str]]):
        """
        Set the mapping of triples to documents in the cache.
        Args:
            triples (Dict[str, Set[str]]): A dictionary mapping triples to sets of document IDs.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def get_docs_from_triples(self, triples: Tuple[str, str, str]) -> List[str]:
        """
        Retrieve the documents associated with a given triple from the cache.
        Args:
            triples (Tuple[str, str, str]): A tuple representing the triple.
        Returns:
            List[str]: A list of document IDs associated with the triple.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def delete_ent_node_to_chunk_ids(self, ent_node_ids: List[str]):
        """
        Delete the mapping of entity node IDs to chunk IDs from the cache.
        Args:
            ent_node_ids (List[str]): A list of entity node IDs to be deleted.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def delete_node_to_node_stats(self, from_node_key: str, to_node_key: str):
        """
        Delete the node-to-node statistics from the cache.
        Args:
            from_node_key (str): The key of the source node.
            to_node_key (str): The key of the target node.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def delete_triples_to_docs(self, triples: List[Tuple[str, str, str]]):
        """
        Delete the mapping of triples to documents from the cache.
        Args:
            triples (List[Tuple[str, str, str]]): A list of tuples representing the triples to be deleted.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def get_ent_node_to_chunk_cache_key(self, ent_node_id: str) -> str:
        """
        Generate the cache key for the mapping of entity node ID to chunk IDs.
        Args:
            ent_node_id (str): The entity node ID.
        Returns:
            str: The cache key for the mapping.
        """
        raise NotImplementedError("Subclasses should implement this method.")
