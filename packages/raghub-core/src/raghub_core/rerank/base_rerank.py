from abc import abstractmethod
from typing import Dict, List

from raghub_core.schemas.document import Document
from raghub_core.utils.class_meta import RegisterABCMeta


class BaseRerank(metaclass=RegisterABCMeta):
    """
    Base class for reranking models.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    async def rerank(self, query: str, documents: List[Document]) -> Dict[str, float]:
        """
        Rerank the given documents based on the query.

        Args:
            query (str): The search query.
            documents (list): List of documents to rerank.

        Returns:
            Dict[str, float]: A dictionary with document IDs as keys and their reranked scores as values.
        """
        raise NotImplementedError("Rerank method must be implemented by subclasses.")
