from abc import abstractmethod
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from raghub_core.chat.base_chat import BaseChat
from raghub_core.schemas.chat_response import QAChatResponse
from raghub_core.schemas.document import Document
from raghub_core.schemas.graph_model import GraphCommunity, GraphEdge, GraphModel, GraphVertex, QueryIndentationModel
from raghub_core.schemas.hipporag_models import OpenIEInfo
from raghub_core.schemas.rag_model import RetrieveResultItem
from raghub_core.utils.class_meta import SingletonRegisterMeta


class BaseRAG(metaclass=SingletonRegisterMeta):
    """
    Base class for Retrieval-Augmented Generation (RAG) systems.
    """

    @abstractmethod
    async def create(self, index_name: str):
        """
        Create a new index in the RAG system.
        The index name should be unique.
        Args:
            index_name (str): Name of the index to create.
        Returns:
            None
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    async def add_documents(self, unique_name: str, texts: List[Document]) -> List[Document]:
        """
        Add documents to the RAG system.
        Args:
            unique_name (str): Name of the index to which documents will be added.
            texts (List[Document]): List of documents to add.
        Returns:
            List[Document]: List of added documents.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    async def delete(self, unique_name: str, doc_ids: List[str] | str) -> None:
        """
        Delete documents from the RAG system.
        Args:
            unique_name (str): Name of the index from which documents will be deleted.
            doc_ids (List[str]|str): List of document IDs to delete.
        Returns:
            None
        """
        raise NotImplementedError("This method 'delete' should be overridden by subclasses.")

    @abstractmethod
    def init(self) -> None:
        raise NotImplementedError("This method 'init' should be overridden by subclasses.")

    @abstractmethod
    async def retrieve(
        self, unique_name: str, queries: List[str], retrieve_top_k: int = 10
    ) -> Dict[str, List[RetrieveResultItem]]:
        """
        Retrieve documents based on the provided queries.
        Args:
            unique_name (str): Name of the index to use for retrieval.
            queries (List[str]): List of query strings.
            retrieve_top_k (int): Number of top documents to retrieve.
        Returns:
            List[RetrieveResultItem]: List of retrieved documents and their corresponding scores.
        """
        raise NotImplementedError("This method 'retrieve' should be overridden by subclasses.")

    @abstractmethod
    async def qa(
        self, unique_name: str, query: str, top_k: int = 5, prompt: Optional[str] = None, llm: Optional[BaseChat] = None
    ) -> AsyncIterator[QAChatResponse]:
        """
        Perform question answering on the RAG system.
        Args:
            unique_name (str): Name of the index to use for the question answering.
            query (str): The question to answer.
            top_k (int): Number of top results to retrieve.
            prompt (Optional[str]): Optional prompt for the LLM.
            llm (Optional[BaseChat]): Optional LLM instance to use for generating answers.
        Returns:
            AsyncIterator[QAChatResponse]: Iterator of QAChatResponse containing the answers and their scores.
        """
        raise NotImplementedError("This method 'qa' should be overridden by subclasses.")


class BaseGraphRAGDAO(metaclass=SingletonRegisterMeta):
    """
    Base class for Graph Storage in RAG systems.
    """

    @abstractmethod
    async def add_documents(self, index_name: str, documents: List[Document]) -> List[Document]:
        """
        Add documents to the graph storage system.
        Args:
            index_name (str): Name of the index to which documents will be added.
            documents (List[Document]): List of documents to add.
        Returns:
            List[Document]: List of added documents.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    async def aselect_vertices_group_by_graph(
        self, index_name: str, filter: Dict[str, Any]
    ) -> Dict[str, List[GraphVertex]]:
        """
        Select vertices grouped by graph from the graph storage system.
        Args:
            index_name (str): Name of the index to use for selection.
            filter (Dict[str, Any]): Filter criteria for selecting vertices.
        Returns:
            Dict[str, List[GraphVertex]]: Dictionary with graph names as keys and lists of vertices as values.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    async def create(self, index_name: str):
        """
        Create a new index in the graph storage system.
        The index name should be unique.
        Args:
            index_name (str): Name of the index to create.
        Returns:
            None
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    async def add_virtices(self, unique_name: str, texts: List[GraphVertex]):
        """
        Add vertices to the graph storage system.
        Args:
            unique_name (str): Name of the index to which vertices will be added.
            texts (List[Document]): List of documents to add as vertices.
        Returns:
            List[Document]: List of added vertices.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    async def delete(self, unique_name: str, doc_ids: List[str] | str) -> None:
        """
        Delete vertices from the graph storage system.
        Args:
            unique_name (str): Name of the index from which vertices will be deleted.
            doc_ids (List[str]|str): List of vertex IDs to delete.
        Returns:
            None
        """
        raise NotImplementedError("This method 'delete' should be overridden by subclasses.")

    @abstractmethod
    async def init(self) -> None:
        """
        Initialize the graph storage system.
        Returns:
            None
        """
        raise NotImplementedError("This method 'init' should be overridden by subclasses.")

    @abstractmethod
    async def add_edges(self, unique_name: str, edges: List[GraphEdge]):
        """
        Add edges to the graph storage system.
        Args:
            unique_name (str): Name of the index to which edges will be added.
            edges (List[Document]): List of documents to add as edges.
        Returns:
            List[GraphEdge]: List of added edges.
        """
        raise NotImplementedError("This method 'add_edges' should be overridden by subclasses.")

    @abstractmethod
    async def save_openie_info(self, unique_name: str, openie_info: List[OpenIEInfo]) -> List[Document]:
        """
        Save OpenIE information to the graph storage system.
        Args:
            unique_name (str): Name of the index to which OpenIE information will be saved.
            openie_info (List[Document]): List of OpenIE information to save.
        Returns:
            List[Document]: List of saved OpenIE information.
        """
        raise NotImplementedError("This method 'save_openie_info' should be overridden by subclasses.")

    @abstractmethod
    async def similar_search_with_scores(
        self, index_name: str, query: str, top_k: int = 10, filter: Optional[Dict[str, str]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Perform a similar search with scores in the graph storage system.
        Args:
            index_name (str): Name of the index to use for the search.
            query (str): Query string for the search.
            top_k (int): Number of top results to return.
            filter (Optional[Dict[str, str]]): Optional filter for the search.
        Returns:
            List[Tuple[Document, float]]: List of tuples containing documents and their corresponding scores.
        """
        raise NotImplementedError("This method 'similar_search_with_scores' should be overridden by subclasses.")

    @abstractmethod
    async def discover_communities(self, label: str) -> List[GraphCommunity]:
        """
        Discover communities in the graph storage system.
        Returns:
            List[str]: List of discovered communities.
        """
        raise NotImplementedError("This method 'discover_communities' should be overridden by subclasses.")

    @abstractmethod
    async def get_community(self, lable, community_id: str) -> GraphCommunity:
        """
        Get a community from the graph storage system.
        Args:
            lable (str): Label of the community.
            community_id (str): ID of the community to retrieve.
        Returns:
            List[Document]: List of documents in the community.
        """
        raise NotImplementedError("This method 'get_community' should be overridden by subclasses.")

    @abstractmethod
    async def explore_trigraph(self, index_name: str, entities: List[str]) -> GraphModel:
        """
        Explore a trigraph in the graph storage system.
        Args:
            index_name (str): Name of the index to use for exploration.
            entities (List[str]): List of entities for exploration.
            top_k (int): Number of top results to return.
            score_threshold (Optional[float]): Optional score threshold for filtering results.
        Returns:
            GraphModel: The explored trigraph model.
        """
        raise NotImplementedError("This method 'explore_trigraph' should be overridden by subclasses.")

    @abstractmethod
    async def search_communities(
        self, label: str, query: str, top_k: int = 5, similar_threshold=0.55
    ) -> List[GraphCommunity]:
        """
        Search for communities in the graph storage system.
        Args:
            label (str): The label of the communities to search.
            query (str): The query string to search for.
            top_k (int): The number of top results to return.
        Returns:
            List[GraphCommunity]: List of communities matching the query.
        """
        raise NotImplementedError("This method 'search_communities' should be overridden by subclasses.")

    @abstractmethod
    async def search_graph_by_indent(self, index_name: str, indent: QueryIndentationModel) -> Optional[GraphModel]:
        """
        Search for a graph based on the provided indentation model.
        Args:
            index_name (str): Name of the index to use for the search.
            indent (QueryIndentationModel): The indentation model to use for searching.
        Returns:
            Optional[GraphModel]: The found graph model or None if not found.
        """
        raise NotImplementedError("This method 'search_graph_by_indent' should be overridden by subclasses.")

    @abstractmethod
    async def get_verteices_by_ids(self, index_name: str, ids: List[str]) -> List[GraphVertex]:
        """
        Get vertices by their IDs from the graph storage system.
        Args:
            index_name (str): Name of the index to use for retrieval.
            ids (List[str]): List of vertex IDs to retrieve.
        Returns:
            List[GraphVertex]: List of retrieved vertices.
        """
        raise NotImplementedError("This method 'get_verteices_by_ids' should be overridden by subclasses.")

    @abstractmethod
    async def get_docs_by_entities(self, index_name: str, entities: List[str]) -> List[Document]:
        """
        Get documents by their entities from the graph storage system.
        Args:
            index_name (str): Name of the index to use for retrieval.
            entities (List[str]): List of entity names to retrieve documents for.
        Returns:
            List[Document]: List of retrieved documents.
        """
        raise NotImplementedError("This method 'get_docs_by_entities' should be overridden by subclasses.")
