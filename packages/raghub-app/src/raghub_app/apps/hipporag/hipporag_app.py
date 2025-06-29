from typing import Dict, List

from raghub_core.chat.base_chat import BaseChat
from raghub_core.embedding import BaseEmbedding
from raghub_core.rag.hipporag.hipporag_impl import HippoRAGImpl
from raghub_core.rag.hipporag.hipporag_storage import HipporagStorage
from raghub_core.rerank.base_rerank import BaseRerank
from raghub_core.schemas.rag_model import RetrieveResultItem
from raghub_core.storage.graph import GraphStorage
from raghub_core.storage.vector import VectorStorage
from raghub_core.utils.class_meta import ClassFactory

from raghub_app.apps.app_rag_base import BaseRAGApp
from raghub_app.config.config_models import APPConfig


class HippoRAG(BaseRAGApp):
    name = "hipporag_app"

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
        self.config = config
        self._db = ClassFactory.get_instance(
            config.hipporag.storage_provider,
            HipporagStorage,
            db_config=config.database,
            cache_config=config.cache,
            search_engine_config=config.search_engine,
        )
        hipporag_config = config.hipporag.model_dump()
        hipporag_config["embedding_prefix"] = config.rag.embbeding.embedding_key_prefix
        hipporag_config["graph_path"] = config.graph.graph_path
        hipporag_config.pop("storage_provider", None)
        self._rerank = ClassFactory.get_instance(
            config.rag.rerank.provider, BaseRerank, **config.rag.rerank.model_dump()
        )
        self.hipporag = HippoRAGImpl(
            self._llm,
            self._rerank,
            self._embedder,
            self._embedd_store,
            self._graph_store,
            self._db,
            **hipporag_config,
        )
        super().__init__(self.hipporag)

    async def create(self, unique_name: str):
        """
        Creates a new index in the vector store and graph store.

        Args:
            unique_name : str
                The unique name for the index to be created.
        """
        await self.hipporag.create(unique_name)

    async def retrieve(
        self, unique_name: str, queries: List[str], retrieve_top_k=10, lang="en"
    ) -> Dict[str, List[RetrieveResultItem]]:
        """
        Retrieves documents based on the provided queries using a combination of dense
        passage retrieval and graph search.
        Args:
            unique_name : str
                The unique name for the index to be used for retrieval.
            queries : List[str]
                A list of query strings for which documents need to be retrieved.
            retrieve_top_k : int, optional
                The number of top documents to retrieve. Defaults to 10.
            lang : str, optional
                The language of the queries. Defaults to "en".
            link_top_k : int, optional
                The number of top facts to consider for graph search. Defaults to 5.
        Returns:
            Dict[str, List[RetrieveResultItem]]
            A dictionary where keys are query strings and values are lists of RetrieveResultItem objects,
        """
        return await self.hipporag.retrieve(
            unique_name,
            queries=queries,
            retrieve_top_k=retrieve_top_k,
            lang=lang,
        )

    async def init(self):
        """
        Initializes the HippoRAG instance by loading the database and embedding store.
        Returns:
            None
        """
        await self.hipporag.init()
