from typing import Dict, List

import httpx
from raghub_core.rerank.base_rerank import BaseRerank
from raghub_core.schemas.document import Document


class RerankProxy(BaseRerank):
    """
    Proxy class for reranking models.
    This class is used to initialize and manage reranking models.
    """

    name = "RerankProxy"
    description = "Proxy class for reranking models. This class is used to initialize and manage reranking models."

    def __init__(self, apikey: str, model_name: str, base_url: str):
        super().__init__(model_name)
        self._apikey = apikey
        self._base_url = base_url
        self._model = model_name

    async def rerank(self, query: str, documents: List[Document]) -> Dict[str, float]:
        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"Bearer {self._apikey}", "Content-Type": "application/json"}
            payload = {"query": query, "documents": [doc.content for doc in documents], "model": self._model}
            response = await client.post(self._base_url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            reranked_docs = {}

            for doc in data.get("results", []):
                reranked_docs[documents[doc.get("index")].uid] = doc.get("score", 0.0)
            # Sort the documents by score in descending order
            reranked_docs = dict(sorted(reranked_docs.items(), key=lambda item: item[1], reverse=True))
            return reranked_docs
