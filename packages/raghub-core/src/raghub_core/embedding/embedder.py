from typing import List, Literal, Optional

import numpy as np
from raghub_core.embedding.base_embedding import BaseEmbedding


class Embbedder(BaseEmbedding):
    name = "Embbedder"

    def __init__(
        self,
        model: str,
        provider: Literal["openai-proxy-embedding", "sentence_transformers", "beg"] = "beg",
        batch_size: int = 32,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        n_dims: Optional[int] = None,
    ):
        self.model_name = model
        self.batch_size = batch_size
        self.base_url = base_url
        self.api_key = api_key
        self.n_dims = n_dims
        self.embbedder_type = provider
        self.embbedder: Optional[BaseEmbedding] = None

    def init(self):
        if self.embbedder_type == "openai-proxy-embedding":
            from raghub_core.embedding.openai_embedding import OpenAIEmbedding

            self.embbedder = OpenAIEmbedding(
                model_name=self.model_name, base_url=self.base_url, api_key=self.api_key, batch_size=self.batch_size
            )

        elif self.embbedder_type == "sentence_transformers":
            from raghub_core.embedding.local_embedding import SentenceTransformersEmbedding

            self.embbedder = SentenceTransformersEmbedding(
                model_name=self.model_name, batch_size=self.batch_size, n_dims=self.n_dims
            )
        elif self.embbedder_type == "bge":
            from raghub_core.embedding.local_embedding import BGEEmbedding

            self.embbedder = BGEEmbedding(model_name=self.model_name, batch_size=self.batch_size, n_dims=self.n_dims)
        else:
            raise ValueError(f"Unsupported embedding type: {self.embbedder_type}")
        self.embbedder.init()

    def encode(self, texts: List[str], instruction: Optional[str] = None) -> np.ndarray:
        if self.embbedder is None:
            raise ValueError("Embedder is not initialized. Please call `initialize` method first.")
        return self.embbedder.encode(texts, instruction)

    def encode_query(self, query: str, instruction: Optional[str] = None) -> np.ndarray:
        if self.embbedder is None:
            raise ValueError("Embedder is not initialized. Please call `initialize` method first.")
        return self.embbedder.encode_query(query, instruction)
