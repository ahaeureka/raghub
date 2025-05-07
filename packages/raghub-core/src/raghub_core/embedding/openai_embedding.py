from typing import List, Optional

import numpy as np
from openai import OpenAI
from raghub_core.embedding.base_embedding import BaseEmbedding
from raghub_core.embedding.embedding_helper import EmbeddingHelper


class OpenAIEmbedding(BaseEmbedding):
    name = "openai-proxy-embedding"

    def __init__(
        self, api_key, model_name="text-embedding-ada-002", base_url="https://api.openai.com/v1", batch_size: int = 5
    ):
        super().__init__()
        if not base_url:
            base_url = "https://api.openai.com/v1"
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self._batch_size = batch_size

    def encode(self, texts: List[str], instruction: Optional[str] = None) -> np.ndarray:
        # OpenAI requires batch size <=16
        texts = [EmbeddingHelper.truncate(t, 8191) for t in texts]
        ress = []
        for i in range(0, len(texts), self._batch_size):
            res = self.client.embeddings.create(input=texts[i : i + self._batch_size], model=self.model_name)
            ress.extend([d.embedding for d in res.data])
        return np.array(ress)

    def encode_query(self, text: str, instruction: Optional[str] = None) -> np.ndarray:
        return self.encode([text], instruction)[0]
