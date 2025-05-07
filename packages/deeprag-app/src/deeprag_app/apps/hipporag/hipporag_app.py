from typing import List

from deeprag_app.apps.app_base import BaseApp
from deeprag_app.config.config_models import APPConfig
from deeprag_core.chat.base_chat import BaseChat
from deeprag_core.embedding import BaseEmbedding
from deeprag_core.rag.hipporag.hipporag_impl import HippoRAGImpl
from deeprag_core.rag.hipporag.hipporag_storage import HipporagStorage
from deeprag_core.schemas.document import Document
from deeprag_core.schemas.rag_model import RetrieveResultItem
from deeprag_core.storage.graph import GraphStorage
from deeprag_core.storage.vector import VectorStorage
from deeprag_core.utils.class_meta import ClassFactory
from loguru import logger


class HippoRAG(BaseApp):
    name = "hipporag"

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
        super().__init__()

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
        logger.debug(f"Storage Provider store: {self.config.hipporag.storage_provider}")
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
        self.hipporag = HippoRAGImpl(
            self._llm,
            self._embedder,
            self._embedd_store,
            self._graph_store,
            self._db,
            **hipporag_config,
        )

    def create(self, unique_name: str):
        """
        Creates a new index in the vector store and graph store.

        Args:
            unique_name : str
                The unique name for the index to be created.
        """
        self.hipporag.create(unique_name)

    def add_documents(self, unique_name: str, texts: List[Document], lang="en") -> List[Document]:
        """
        Adds documents to the vector store and graph store.

        Args:
            unique_name : str
                The unique name for the index to which documents will be added.
            texts : List[Document]
                A list of Document objects to be added to the vector store and graph store.
            lang : str
                The language of the documents. Defaults to "en".
        """
        return self.hipporag.add_documents(unique_name, texts, lang=lang)

    def retrieve(self, unique_name: str, queries: List[str], retrieve_top_k=10, lang="en") -> List[RetrieveResultItem]:
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
            List[RetrieveResultItem]
                A list of RetrieveResultItem objects containing the retrieved documents and their corresponding scores.
        """
        return self.hipporag.retrieve(
            unique_name,
            queries=queries,
            retrieve_top_k=retrieve_top_k,
            lang=lang,
        )

    def delete(self, index_name: str, docs_to_delete: List[str]):
        """
        Deletes documents and their associated triples from the database, embedding store, and graph store.
        Args:
            index_name : str
                The unique name for the index from which documents will be deleted.
            docs_to_delete : List[str]
                A list of document IDs to be deleted from the database, embedding store, and graph store.
        Returns:
            None
        """
        self.hipporag.delete(index_name, docs_to_delete)

    def init(self):
        """
        Initializes the HippoRAG instance by loading the database and embedding store.
        Returns:
            None
        """
        self.hipporag.init()
