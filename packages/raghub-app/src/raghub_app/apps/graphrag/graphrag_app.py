from typing import Dict, List

import numpy as np
from raghub_core.chat.base_chat import BaseChat
from raghub_core.embedding.base_embedding import BaseEmbedding
from raghub_core.rag.base_rag import BaseGraphRAGDAO
from raghub_core.rag.graphrag.graphrag_impl import GraphRAGImpl
from raghub_core.rag.graphrag.operators import DefaultGraphRAGOperators
from raghub_core.schemas.graph_model import GraphRAGRetrieveResultItem
from raghub_core.schemas.rag_model import RetrieveResultItem
from raghub_core.storage.graph import GraphStorage
from raghub_core.storage.structed_data import StructedDataStorage
from raghub_core.storage.vector import VectorStorage
from raghub_core.utils.class_meta import ClassFactory

from raghub_app.apps.app_rag_base import BaseRAGApp
from raghub_app.config.config_models import APPConfig


class GraphRAG(BaseRAGApp):
    name = "graphrag_app"

    def __init__(self, config: APPConfig):
        self._llm = ClassFactory.get_instance(
            config.rag.llm.provider,
            BaseChat,
            model_name=config.rag.llm.model,
            api_key=config.rag.llm.api_key,
            base_url=config.rag.llm.base_url,
            temperature=config.rag.llm.temperature,
            timeout=config.rag.llm.timeout,
        )  # Assuming BaseChat is imported from the correct module

        self._embedder = ClassFactory.get_instance(
            "Embbedder",
            BaseEmbedding,
            model=config.rag.embbeding.model,
            provider=config.rag.embbeding.provider,
            batch_size=config.rag.embbeding.batch_size,
            base_url=config.rag.embbeding.base_url,
            api_key=config.rag.embbeding.api_key,
            n_dims=config.rag.embbeding.n_dims,
        )
        embbeddr_store_config = config.vector_storage.model_dump()
        embbeddr_store_config.update(config.search_engine.model_dump())
        self._embedd_store: VectorStorage = ClassFactory.get_instance(
            config.vector_storage.provider, VectorStorage, embedder=self._embedder, **embbeddr_store_config
        )
        self._graph_store: GraphStorage = ClassFactory.get_instance(
            config.graph.provider, GraphStorage, **config.graph.model_dump()
        )

        self._db = ClassFactory.get_instance(
            config.database.provider,
            StructedDataStorage,
            **config.database.model_dump(),
        )
        self._dao = ClassFactory.get_instance(
            config.graphrag.dao_provider,
            BaseGraphRAGDAO,
            embedding_store=self._embedd_store,
            graph_store=self._graph_store,
            db=self._db,
        )
        self.app = GraphRAGImpl(self._llm, self._dao, DefaultGraphRAGOperators(self._llm, self._embedd_store))
        super().__init__(self.app)

    async def init(self):
        """
        Initialize the GraphRAG application.
        """
        self._embedder.init()
        self.app.init()
        await self._embedd_store.init()
        await self._graph_store.init()
        await self._db.init()

    async def create(self, label: str):
        """
        Create a new index in the GraphRAG application.
        """
        await self._embedd_store.create_index(label)

    async def retrieve(
        self, unique_name: str, queries: List[str], retrieve_top_k=5, lang="zh"
    ) -> Dict[str, List[RetrieveResultItem]]:
        items: Dict[str, GraphRAGRetrieveResultItem] = await self.app.retrieve(unique_name, queries, retrieve_top_k)
        retrieval_items: Dict[str, List[RetrieveResultItem]] = {}
        for query, item in items.items():
            vectors = await self._embedder.aencode_query([query])
            its: List[RetrieveResultItem] = []
            for doc in item.docs:
                similar = self._embedder.cosine_similarity(
                    vectors[0], np.array(doc.embedding, dtype=float) if doc.embedding is not None else None
                )
                metadata = {
                    "graph": item.graph.model_dump(),
                    "context": item.context,
                }
                its.append(
                    RetrieveResultItem(
                        document=doc,
                        score=similar,
                        query=item.query,
                        metadata=metadata,
                    )
                )
            retrieval_items[query] = its
        return retrieval_items
