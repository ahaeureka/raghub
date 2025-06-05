from typing import List

from langchain_core.embeddings import Embeddings
from langchain_core.runnables.config import run_in_executor
from raghub_core.embedding.base_embedding import BaseEmbedding


class LangchainEmbeddings(Embeddings):
    def __init__(self, embedder: BaseEmbedding):
        self.embedder = embedder

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        ret = self.embedder.encode(documents).tolist()
        return ret

    def embed_query(self, query: str) -> List[float]:
        ret = self.embedder.encode_query(query).tolist()
        return ret

    async def aembed_documents(self, texts) -> List[List[float]]:
        return await run_in_executor(None, self.embed_documents, texts).tolist()

    async def aembed_query(self, text: str) -> List[float]:
        return await run_in_executor(None, self.embed_query, text).tolist()
