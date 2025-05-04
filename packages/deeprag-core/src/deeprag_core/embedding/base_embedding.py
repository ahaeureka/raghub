from abc import abstractmethod
from typing import List, Optional

import numpy as np
from deeprag_core.utils.class_meta import SingletonRegisterMeta


class BaseEmbedding(metaclass=SingletonRegisterMeta):
    name = ""

    @abstractmethod
    def encode(self, texts: List[str], instruction: Optional[str] = None) -> np.ndarray:
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    def encode_query(self, query: str, instruction: Optional[str] = None) -> np.ndarray:
        raise NotImplementedError("This method should be overridden by subclasses.")

    def init(self):
        """
        Initialize the embedding model. This method should be called before using the model.
        """
        pass
