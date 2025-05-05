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

        self._embedd_store: VectorStorage = ClassFactory.get_instance(
            config.vector_storage.provider, VectorStorage, embedder=self._embedder, **config.vector_storage.model_dump()
        )
        self._graph_store: GraphStorage = ClassFactory.get_instance(
            config.graph.provider, GraphStorage, **config.graph.model_dump()
        )
        self.config = config
        db_config = config.database.model_dump()
        db_config["db_url"] = config.database.url
        db_config["cache_dir"] = config.cache.cache_dir
        db_config["db_provider"] = config.database.provider
        db_config["cache_provider"] = config.cache.provider
        self._db = ClassFactory.get_instance(
            config.hipporag.storage_provider,
            HipporagStorage,
            database_config=config.database,
            cache_config=config.cache,
            **db_config,
        )

        self.hipporag = HippoRAGImpl(
            self._llm,
            self._embedder,
            self._embedd_store,
            self._graph_store,
            self._db,
            config.hipporag.dspy_file_path,
            **config.hipporag.model_dump(),
        )

    def add_documents(self, texts: List[Document], lang="en") -> List[Document]:
        """
        Adds documents to the vector store and graph store.

        Args:
            texts : List[Document]
                A list of Document objects to be added to the vector store and graph store.
            lang : str
                The language of the documents. Defaults to "en".
        """

        return self.hipporag.add_documents(texts, lang=lang)

    def retrieve(self, queries: List[str], retrieve_top_k=10, lang="en") -> List[RetrieveResultItem]:
        """
        Retrieves documents based on the provided queries using a combination of dense
        passage retrieval and graph search.
        Args:
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
            queries=queries,
            retrieve_top_k=retrieve_top_k,
            lang=lang,
        )

    def delete(self, docs_to_delete: List[str]):
        """
        Deletes documents and their associated triples from the database, embedding store, and graph store.
        Args:
            docs_to_delete : List[str]
                A list of document IDs to be deleted from the database, embedding store, and graph store.
        Returns:
            None
        """
        self.hipporag.delete(docs_to_delete)

    def init(self):
        """
        Initializes the HippoRAG instance by loading the database and embedding store.
        Returns:
            None
        """
        self.hipporag.init()
