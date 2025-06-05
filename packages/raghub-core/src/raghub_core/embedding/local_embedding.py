from abc import abstractmethod
from typing import List, Optional

import numpy as np
import torch
from langchain_core.runnables.config import run_in_executor
from raghub_core.embedding.base_embedding import BaseEmbedding


class LocalEmbedding(BaseEmbedding):
    def __init__(self, model_name_or_path: str, batch_size: int = 32, n_dims: Optional[int] = None):
        super().__init__()
        self._model_name_or_path = model_name_or_path
        self._batch_size = batch_size
        self._model = self._load_model()
        self._n_dims = n_dims

    @abstractmethod
    def _load_model(self):
        raise NotImplementedError("This method should be overridden by subclasses.")


class SentenceTransformersEmbedding(LocalEmbedding):
    name = "sentence_transformers"

    # def __init__(self, models_dir: str, model_name: str, batch_size: int = 32):
    #     super().__init__()
    #     from sentence_transformers import SentenceTransformer
    #     self._model = SentenceTransformer(os.path.join(models_dir, model_name))
    #     self._batch_size = batch_size
    def _load_model(self):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self._model_name_or_path)
        return self._model

    def encode(self, texts: List[str], instruction: Optional[str] = None) -> np.ndarray:
        normalize_embeddings = True if self._n_dims is None else False
        embeddings = self._model.encode(
            texts, batch_size=self._batch_size, normalize_embeddings=normalize_embeddings, prompt=instruction
        )
        if normalize_embeddings:
            return embeddings
        from sklearn.preprocessing import normalize

        return normalize(embeddings[:, : self._n_dims])

    def encode_queries(self, query: str) -> np.ndarray:
        return self.encode([query])[0]


class BGEEmbedding(LocalEmbedding):
    name = "beg"

    def _load_model(self):
        from FlagEmbedding import FlagModel

        self._model = FlagModel(
            self._model_name_or_path,
            query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
            use_fp16=torch.cuda.is_available(),
        )
        return self._model

    def encode(self, texts: List[str], instruction: Optional[str] = None) -> np.ndarray:
        ress = []
        for i in range(0, len(texts), self._batch_size):
            ress.extend(self._model.encode(texts[i : i + self._batch_size], instruction=instruction).tolist())
        return np.array(ress)

    def encode_query(self, text: str, instruction: str = "为这个句子生成表示以用于检索相关文章：") -> np.ndarray:
        results = self._model.encode_queries([text], instruction=instruction)
        return np.array(results.tolist()[0])

    async def aencode(self, texts: List[str], instruction: Optional[str] = None) -> np.ndarray:
        ress = []
        for i in range(0, len(texts), self._batch_size):
            ress.extend(
                await run_in_executor(
                    None, self._model.encode, texts[i : i + self._batch_size], instruction=instruction
                ).tolist()
            )
        return np.array(ress)

    async def aencode_query(
        self, texts: List[str], instruction: str = "为这个句子生成表示以用于检索相关文章："
    ) -> np.ndarray:
        ress = []
        for i in range(0, len(texts), self._batch_size):
            ress.extend(
                await run_in_executor(
                    None, self._model.encode_queries, texts[i : i + self._batch_size], instruction=instruction
                ).tolist()
            )
        return np.array(ress)
