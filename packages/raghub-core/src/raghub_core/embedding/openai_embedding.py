import asyncio
from typing import List, Optional

import numpy as np
from openai import AsyncOpenAI, OpenAI
from raghub_core.embedding.base_embedding import BaseEmbedding


class OpenAIEmbedding(BaseEmbedding):
    name = "openai-proxy-embedding"

    def __init__(
        self,
        api_key,
        model_name="text-embedding-ada-002",
        base_url="https://api.openai.com/v1",
        batch_size: int = 5,
        n_dims: Optional[int] = None,
    ):
        super().__init__()
        if not base_url:
            base_url = "https://api.openai.com/v1"
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self._async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self._batch_size = batch_size
        self._n_dims = n_dims

    @property
    def embedding_dim(self):
        return self._n_dims

    def encode(self, texts: List[str], instruction: Optional[str] = None) -> np.ndarray:
        # OpenAI requires batch size <=16
        # texts = [EmbeddingHelper.truncate(t, 8191) for t in texts]
        ress = []
        for i in range(0, len(texts), self._batch_size):
            res = self.client.embeddings.create(input=texts[i : i + self._batch_size], model=self.model_name)
            ress.extend([d.embedding for d in res.data])
        return np.array(ress)

    def encode_query(self, text: str, instruction: Optional[str] = None) -> np.ndarray:
        return self.encode([text], instruction)[0]

    async def aencode(self, texts: List[str], instruction: Optional[str] = None) -> np.ndarray:
        ress = []
        tasks = []
        for i in range(0, len(texts), self._batch_size):
            tasks.append(
                self._async_client.embeddings.create(input=texts[i : i + self._batch_size], model=self.model_name)
            )
        # return np.array(ress)
        # # Create tasks
        # embedding_tasks = [(self.embed(text)) for text in batch_texts]
        # Process embedding in parallel
        batch_results = await asyncio.gather(*tasks, return_exceptions=False)
        for batch_result in batch_results:
            ress.extend([d.embedding for d in batch_result.data])
        return np.array(ress)

    async def aencode_query(self, texts: List[str], instruction: Optional[str] = None) -> np.ndarray:
        if self.model_name.lower().startswith("bbai") and instruction:
            texts = [instruction + text for text in texts]
        return await self.aencode(texts, instruction)

    def similarity(self, src: str, dst: str) -> float:
        embeddings = self.encode([src, dst])
        return self.cosine_similarity(embeddings[0], embeddings[1])

    async def asimilarity(self, src: str, dst: str) -> float:
        embeddings = await self.aencode([src, dst])
        return self.cosine_similarity(embeddings[0], embeddings[1])
