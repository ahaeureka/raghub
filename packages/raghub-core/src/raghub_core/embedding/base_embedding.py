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

    def similarity(self, src: str, dst: str) -> float:
        embeddings = self.encode([src, dst])
        return self.cosine_similarity(embeddings[0], embeddings[1])

    async def asimilarity(self, src: str, dst: str) -> float:
        embeddings = await self.aencode([src, dst])
        return self.cosine_similarity(embeddings[0], embeddings[1])

    def cosine_similarity(self, src: np.ndarray, dst: np.ndarray) -> float:
        # 计算点积
        dot_product = np.dot(src, dst)
        # 计算 L2 范数
        norm_src = np.linalg.norm(src)
        norm_dst = np.linalg.norm(dst)
        # 处理分母为零的情况（避免数值错误）
        if norm_src == 0 or norm_dst == 0:
            return 0.0
        # 返回余弦相似度
        return dot_product / (norm_src * norm_dst)
