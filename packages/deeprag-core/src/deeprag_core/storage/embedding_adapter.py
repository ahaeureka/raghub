from typing import List

from deeprag_core.embedding.base_embedding import BaseEmbedding
from langchain_core.embeddings import Embeddings


class LangchainEmbeddings(Embeddings):
    def __init__(self, embedder: BaseEmbedding):
        self.embedder = embedder

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        ret = self.embedder.encode(documents).tolist()
        return ret

    def embed_query(self, query: str) -> List[float]:
        ret = self.embedder.encode_query(query).tolist()
        return ret
