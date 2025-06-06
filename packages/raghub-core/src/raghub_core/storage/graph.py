from abc import abstractmethod
from typing import Any, Dict, List, Optional

from raghub_core.schemas.document import Document
from raghub_core.schemas.graph_model import GraphCommunity, GraphEdge, GraphModel, GraphVertex
from raghub_core.utils.class_meta import SingletonRegisterMeta


class GraphStorage(metaclass=SingletonRegisterMeta):
    name = ""

    # _registry: Dict[str, Type["GraphStorage"]] = {}  # 注册表
    @abstractmethod
    async def init(self):
        raise NotImplementedError("Subclasses should implement this `init` method.")

    @abstractmethod
    async def aadd_new_edges(self, label: str, edges: List[GraphEdge]):
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    async def aadd_graph_edges(self, label: str, edges: List[GraphEdge]):
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    async def aget_by_ids(self, label: str, ids: List[str]) -> List[Document]:
        raise NotImplementedError("Subclasses should implement this method.")

    # @abstractmethod
    # def vs(self, name: str = "") -> List[str] | Dict[str, List[str]]:
    #     raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    async def aadd_vertices(self, label: str, nodes: List[GraphVertex]):
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    async def aadd_graph_vertices(self, label: str, nodes: List[GraphVertex]):
        """
        Add graph vertices to the storage.

        Args:
            label (str): The label of the vertices.
            nodes (List[GraphVertex]): A list of GraphVertex objects to be added.

        Returns:
            None
        """
        raise NotImplementedError("Subclasses should implement this method.")

    # @abstractmethod
    # def edge_seq(self, name: str = "") -> List[str] | Dict[str, List[str]]:
    #     raise NotImplementedError("Subclasses should implement this method.")

    # @abstractmethod
    # async def avertices_count(self, label: str) -> int:
    #     raise NotImplementedError("Subclasses should implement this `vertices_count` method.")

    @abstractmethod
    async def apersonalized_pagerank(
        self,
        label: str,
        vertices_with_weight: Dict[str, float],
        damping: float = 0.85,
        top_k: int = 10,
        **kwargs: Any,
    ) -> Dict[str, float]:
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    async def aupdate_vertices(self, label: str, nodes: List[GraphVertex]) -> List[GraphVertex]:
        """
        Update existing vertices in the graph.

        Args:
            label (str): The label of the vertices to be updated.
            nodes (List[GraphVertex]): A list of GraphVertex objects to be updated.

        Returns:
            List[GraphVertex]: A list of updated GraphVertex objects.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    async def aupdate_edges(self, label: str, edges: List[GraphEdge]) -> List[GraphEdge]:
        """
        Update existing edges in the graph.

        Args:
            label (str): The label of the edges to be updated.
            edges (List[GraphEdge]): A list of GraphEdge objects to be updated.

        Returns:
            List[GraphEdge]: A list of updated GraphEdge objects.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    async def aupsert_virtices(self, unique_name: str, vertices: List[GraphVertex]) -> List[GraphVertex]:
        """
        Upsert vertices in the graph.

        Args:
            unique_name (str): The unique name of the vertices.
            vertices (List[GraphVertex]): A list of GraphVertex objects to be upserted.

        Returns:
            List[GraphVertex]: A list of upserted GraphVertex objects.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    async def aupsert_edges(self, unique_name: str, edges: List[GraphEdge]) -> List[GraphEdge]:
        """
        Upsert edges in the graph.

        Args:
            unique_name (str): The unique name of the edges.
            edges (List[GraphEdge]): A list of GraphEdge objects to be upserted.

        Returns:
            List[GraphEdge]: A list of upserted GraphEdge objects.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    async def aselect_vertices(self, label, attrs: Dict[str, Any]) -> List[GraphVertex]:
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    async def asearch_neibors(
        self,
        vertex_id: str,
        rel_type: Optional[str] = None,
    ) -> GraphModel:
        """
        Search for neighbors of a given vertex in the graph.

        Args:
            vertex_id: The ID of the vertex to search for neighbors.
            rel_type: The type of relationship to filter the neighbors.

        Returns:
            A GraphModel containing the neighbors of the specified vertex.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    async def aselect_vertices_group_by_graph(
        self, label: str, attrs: Dict[str, Any]
    ) -> List[Dict[str, List[GraphVertex]]]:
        """
        Select vertices from the graph based on attributes and group them by a specified field.

        Args:
            label: The name of the label to use for the search.
            attrs: A dictionary of attributes to filter the vertices.
            group_by: The field to group the vertices by.

        Returns:
            A list of GraphVertex objects representing the selected vertices.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    async def aselect_edges(self, label: str, attrs: Dict[str, Any]) -> List[GraphEdge]:
        """
        Select edges from the graph based on source and target vertices.

        Args:
            label: The name of the label to use for the search.
            source: The source vertex ID.
            target: The target vertex ID.

        Returns:
            A list of GraphEdge objects representing the edges between the source and target vertices.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    async def adelete_vertices(self, label: str, keys: List[str]):
        """Delete vertices from the graph."""
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    async def discover_communities(
        self, label, resolution_parameter=0.8, beta=0.1, n_iterations=-1, **kwargs
    ) -> List[str]:
        """
        Discover communities in the graph using the Louvain method.

        Args:
            label: The label of the vertices to be clustered.
            resolution_parameter: The resolution parameter for the Louvain method.
            beta: The beta parameter for the Louvain method.
            n_iterations: The number of iterations for the Louvain method.

        Returns:
            A list of community labels for each vertex in the graph.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    async def get_community(self, label: str, community_id: str) -> GraphCommunity:
        """
        Get the community of a specific vertex.

        Args:
            label: The label of the vertices to be clustered.
            community_id: The ID of the community to retrieve.

        Returns:
            A list of vertices in the specified community.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def graph_vertices2nodes(self, vertices: List[GraphVertex]) -> List[Dict[str, Any]]:
        """
        Convert a list of GraphVertex objects to a list of dictionaries.

        Args:
            vertices: A list of GraphVertex objects.

        Returns:
            A list of dictionaries representing the vertices.
        """
        return [vertex.model_dump() for vertex in vertices]

    @abstractmethod
    async def multi_hop_search(
        self,
        label: str,
        start_nodes_id: List[str],
        relation_path: List[str] = [],
        max_hops: int = 3,
        rel_type: str = "",
        max_paths: int = 10,
        continuous=False,
    ) -> GraphModel | None:
        """
        Perform a multi-hop search in the graph.

        Args:
            label: The label of the vertices to be searched.
            namespace: The namespace of the vertices to be searched.
            start_nodes_id: A list of starting node IDs for the search.
            rel_type: The type of relationship to follow during the search.
            max_hops: The maximum number of hops to traverse in the graph.
            limit: The maximum number of results to return.

        Returns:
            A GraphModel containing the results of the multi-hop search.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    async def freestyle_search(self, label: str, entities: List[str], max_hops: int = 3) -> GraphModel | None:
        """
        Perform a freestyle search in the graph.

        Args:
            label: The label of the vertices to be searched.
            entities: A list of entities to search for.
            max_hops: The maximum number of hops to traverse in the graph.

        Returns:
            A GraphModel containing the results of the freestyle search.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def transform_logic_operators(self, attr: Dict[str, Any]) -> str:
        """
        Transform logic operators in the attribute dictionary to Cypher syntax.

        Keyword arguments can be used to filter the vertices based on their
        attributes. The name of the keyword specifies the name of the attribute
        and the filtering operator, they should be concatenated by an
        underscore (C{_}) character. Attribute names can also contain
        underscores, but operator names don't, so the operator is always the
        largest trailing substring of the keyword name that does not contain
        an underscore. Possible operators are:

          - C{eq}: equal to

          - C{ne}: not equal to

          - C{lt}: less than

          - C{gt}: greater than

          - C{le}: less than or equal to

          - C{ge}: greater than or equal to

          - C{in}: checks if the value of an attribute is in a given list

          - C{notin}: checks if the value of an attribute is not in a given
            list

        For instance, if you want to filter vertices with a numeric C{age}
        property larger than 200, you have to write:

          >>> select(age_gt=200)  # doctest: +SKIP

        Similarly, to filter vertices whose C{type} is in a list of predefined
        types:

          >>> list_of_types = ["HR", "Finance", "Management"]
          >>> select(type_in=list_of_types)  # doctest: +SKIP

        If the operator is omitted, it defaults to C{eq}. For instance, the
        following selector selects vertices whose C{cluster} property equals
        to 2:

          >>> select(cluster=2)  # doctest: +SKIP
        Args:
            attr: Dictionary of attributes with logic operators
        Returns:
        """
        if not attr:
            return ""
        conditions = []
        for key, value in attr.items():
            if "_" in key:
                attr_name, operator = key.rsplit("_", 1)
                if operator == "eq":
                    conditions.append(f"n.{attr_name} = ${key}")
                elif operator == "ne":
                    conditions.append(f"n.{attr_name} <> ${key}")
                elif operator == "lt":
                    conditions.append(f"n.{attr_name} < ${key}")
                elif operator == "gt":
                    conditions.append(f"n.{attr_name} > ${key}")
                elif operator == "le":
                    conditions.append(f"n.{attr_name} <= ${key}")
                elif operator == "ge":
                    conditions.append(f"n.{attr_name} >= ${key}")
                elif operator == "in":
                    conditions.append(f"n.{attr_name} IN ${key}")
                elif operator == "notin":
                    conditions.append(f"n.{attr_name} NOT IN ${key}")
            else:
                conditions.append(f"n.{key} = ${key}")
        if len(conditions) == 0:
            return ""
        if len(conditions) == 1:
            return conditions[0]
        return " AND ".join(conditions)
