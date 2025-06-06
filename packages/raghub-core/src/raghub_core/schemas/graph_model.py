from enum import StrEnum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel
from raghub_core.schemas.document import Document
from sqlmodel import Field, SQLModel


class RelationType(StrEnum):
    """
    Enum for relation types in a graph.
    """

    RELATION = "RELATED"
    INCLUDE = "INCLUDE"


class Namespace(StrEnum):
    """
    Enum for namespace types in a graph.
    """

    ENTITY = "entity"
    DOC = "doc"
    FACT = "fact"
    CONTEXT = "context"
    PASSAGE = "passage"


class GraphVertex(BaseModel):
    """
    A vertex in a graph.

    Attributes:
        id (int): The unique identifier for the vertex.
        name (str): The name of the vertex.
        description (str): A description of the vertex.
    """

    name: str = Field(..., description="The name of the vertex.")
    description: Optional[Dict[str, str]] = Field(default=[], description="A summary of the vertex.")
    metadata: Dict[str, Any] = Field(default={}, description="The metadata associated with the vertex.")
    uid: str = Field(..., description="The unique identifier for the vertex.")
    content: str = Field(..., description="The content of the vertex.")
    embedding: Optional[List[float]] = Field(default=None, description="The embedding of the vertex.")
    doc_id: Optional[List[str]] = Field(default=None, description="The document ID associated with the vertex.")
    namespace: Optional[Literal["entity", "doc", "fact"]] = Field(
        default=None, description="The namespace of the vertex."
    )
    label: Optional[str] = Field(default=None, description="The label of the vertex, if applicable.")
    index: Optional[int] = Field(
        default=None, description="The index of the vertex in the graph, if applicable."
    )  # Optional index for the vertex, e.g., for ordering in a list


class GraphEdge(SQLModel):
    """
    An edge in a graph.

    Attributes:
        id (int): The unique identifier for the edge.
        source (int): The source vertex ID.
        target (int): The target vertex ID.
        weight (float): The weight of the edge.
    """

    __tablename__ = "graph_edge"
    source: str = Field(..., description="The source vertex ID.")
    target: str = Field(..., description="The target vertex ID.")
    source_content: str = Field(..., description="The content of the source vertex.")
    target_content: str = Field(..., description="The content of the target vertex.")
    weight: float = Field(..., description="The weight of the edge.")
    relation_type: RelationType = Field(..., description="The relation type of the edge.")
    relation: Optional[str] = Field(default=None, description="The relation of the edge, if applicable.")
    description: Optional[Dict[str, str]] = Field(default="", description="A summary of the edge.")
    edge_metadata: Dict[str, Any] = Field(
        default={}, alias="metadata", description="The metadata associated with the edge."
    )
    uid: str = Field(..., description="The unique identifier for the edge.")
    label: Optional[str] = Field(
        default=None, description="The label of the edge, if applicable."
    )  # Optional label for the edge, e.g., 'relation', 'include'


class GraphModel(SQLModel):
    vertices: List[GraphVertex] = Field(default_factory=list, description="A list of vertices in the graph.")
    edges: List[GraphEdge] = Field(default_factory=list, description="A list of edges in the graph.")


class GraphCommunity(SQLModel):
    """
    A community in a graph.

    Attributes:
        id (int): The unique identifier for the community.
        name (str): The name of the community.
        description (str): A description of the community.
    """

    cid: str = Field(..., description="The unique identifier for the community.")
    name: str = Field(..., description="The name of the community.")
    graph: GraphModel = Field(default_factory=list, description="A list of graphs in the community.")
    summary: Optional[str] = Field(default="", description="The summary for the community.")


class SearchIndentationCategory(StrEnum):
    """
    Enum for search indentation types.
    """

    SINGLE_ENTITY_SEARCH = "SingleEntitySearch"
    ONE_HOP_ENTITY_SEARCH = "OneHopEntitySearch"
    ONE_HOP_RELATION_SEARCH = "OneHopRelationSearch"
    TWO_HOP_ENTITY_SEARCH = "TwoHopEntitySearch"
    FREESTYLE_QUESTION = "FreestyleQuestion"


class QueryIndentationModel(BaseModel):
    # {{"category": "SingleEntitySearch", entities": ["TuGraph"], "relations": []}}
    category: SearchIndentationCategory = Field(
        ...,
        description="The category of the query, indicating the type of search to perform.",
    )
    entities: List[str] = Field(
        default_factory=list,
        description="A list of entities involved in the query.",
    )
    relations: List[str] = Field(
        default_factory=list,
        description="A list of relations involved in the query.",
    )


class GraphRAGRetrieveResultItem(BaseModel):
    """
    A single item in the retrieval result for a graph-based RAG system.

    Attributes:
        graph (GraphModel): The retrieved vertex.
        score (float): The score associated with the retrieved vertex.
        metadata (Optional[Dict[str, Any]]): Additional metadata for the retrieved item.
        query (str): The original query used for retrieval.
    """

    graph: Optional[GraphModel] = Field(None, description="The retrieved graph.")
    subgraph: str = Field(
        default=None, description="The subgraph associated with the retrieved graph."
    )  # Optional subgraph for the retrieval, e.g., a specific part of the graph
    docs: List[Document] = Field(
        default_factory=list, description="List of documents associated with the retrieved graph."
    )
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for the retrieved item.")
    query: str = Field(..., description="The original query used for retrieval.")
    context: Optional[str] = Field(
        default=None, description="Contextual information related to the retrieval."
    )  # Optional context for the retrieval, e.g., a summary or explanation
