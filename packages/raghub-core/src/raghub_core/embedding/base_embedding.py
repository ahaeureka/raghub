from abc import abstractmethod
from typing import List, Optional

import numpy as np
from raghub_core.utils.class_meta import SingletonRegisterMeta


class BaseEmbedding(metaclass=SingletonRegisterMeta):
    name = ""

    @abstractmethod
    def encode(self, texts: List[str], instruction: Optional[str] = None) -> np.ndarray:
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    def encode_query(self, query: str, instruction: Optional[str] = None) -> np.ndarray:
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    async def aencode(self, texts: List[str], instruction: Optional[str] = None) -> np.ndarray:
        """
        Asynchronously encode a list of texts into embeddings.
        Args:
            texts (List[str]): A list of texts to encode.
            instruction (Optional[str]): An optional instruction for the encoding process.
        Returns:
            np.ndarray: The encoded embeddings as a NumPy array.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    async def aencode_query(self, query: List[str], instruction: Optional[str] = None) -> np.ndarray:
        """
        Asynchronously encode a single query into an embedding.
        Args:
            query (str): The query to encode.
            instruction (Optional[str]): An optional instruction for the encoding process.
        Returns:
            np.ndarray: The encoded embedding as a NumPy array.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def init(self):
        """
        Initialize the embedding model. This method should be called before using the model.
        """
        pass
