import os
from typing import Any, Dict, List, Tuple

import igraph as ig
from deeprag_core.schemas.document import Document
from deeprag_core.storage.graph import GraphStorage
from loguru import logger


class IGraphStore(GraphStorage):
    name = "igraph"

    def __init__(self, graph_path: str, **kwargs):
        super().__init__(**kwargs)
        self._graph_path = graph_path
        self._graph: ig.Graph = None

    def init(self):
        # Initialize the IGraph storage
        if not self._graph:
            self.load_graph()
        logger.debug("Graph already loaded.")

    def load_graph(self):
        logger.debug(f"Loading graph from {self._graph_path}.")
        if self._graph_path and os.path.exists(self._graph_path):
            self._graph = ig.Graph.Read_Pickle(self._graph_path)
            logger.info(f"Graph loaded from {self._graph_path}.")
        else:
            self._graph = ig.Graph()
            logger.debug("Initialized a new empty graph.")

    def add_edges(self, edges: List[Tuple[str, str]], weights: Tuple[float]):
        if not self._graph:
            self.load_graph()
        if len(edges) != len(weights):
            raise ValueError("The number of edges and weights must be the same.")
        self._graph.add_edges(edges, attributes={"weight": weights})
        logger.info(f"Added {len(edges)} edges to the graph.")

    def vs(self, name: str = "") -> List[str] | Dict[str, List[str]]:
        """The vertex sequence of the graph"""
        if not self._graph:
            self.load_graph()
        if not name:
            return self._graph.vs
        if name in self._graph.vs:
            return self._graph.vs[name]
        return []

    def select_vertices(self, **attrs) -> List[Dict[str, Any]]:
        """Select vertices based on attributes."""
        if not self._graph:
            self.load_graph()
        if not self._graph.vs.attributes():
            logger.warning("No attributes found in the graph vertices.")
            return []
        logger.info(f"Selecting vertices with attributes: {self._graph.vs.attributes()} with {attrs}")
        ret: ig.VertexSeq = self._graph.vs.select(**attrs)

        indexs = ret.indices
        vertices = {k: ret.get_attribute_values(k) for k in ret.attributes()}
        result = [
            {"content": content, "metadata": metadata, "uid": uid, "embedding": embedding, "name": name, "index": index}
            for content, metadata, uid, embedding, name, index in zip(
                vertices["content"],
                vertices["metadata"],
                vertices["uid"],
                vertices["embedding"],
                vertices["name"],
                indexs,
            )
        ]

        return result

    def edge_seq(self, name: str = "") -> List[str] | Dict[str, List[str]]:
        """The edge sequence of the graph"""
        if not self._graph:
            self.load_graph()
        if not name:
            return self._graph.es
        if name in self._graph.es:
            return self._graph.es[name]
        return []

    def add_vertices(self, n: int, attributes: Dict[str, List[Any]]) -> None:
        """Add vertices to the graph with the given attributes."""
        self._graph.add_vertices(n, attributes=attributes)
        logger.info(f"Added {n} vertices to the graph with attributes {attributes}.")

    def personalized_pagerank(
        self,
        vertices: Any | None = None,
        directed: bool = True,
        damping: float = 0.85,
        weights: Any | None = None,
        reset: Any | None = None,
        arpack_options: Any | None = None,
        implementation: str = "prpack",
        top_k: int = 10,
    ) -> Dict[str, float]:
        """Compute the personalized PageRank of the graph."""
        if not self._graph:
            self.load_graph()
        pr_scores = self._graph.personalized_pagerank(
            vertices,
            directed=directed,
            damping=damping,
            weights=weights,
            reset=reset,  # Corrected the parameter name here
            implementation=implementation,
        )
        labeled_pairs = [(self._graph.vs[i]["name"], pr_scores[i]) for i in range(len(pr_scores))]
        sorted_pairs = sorted(labeled_pairs, key=lambda x: x[1], reverse=True)
        return dict(sorted_pairs[:top_k])

    def save(self, path: str) -> None:
        """Save the graph to a file."""
        path_dir = os.path.dirname(path)
        os.makedirs(path_dir, exist_ok=True)
        self._graph.write_pickle(path)
        logger.info(f"Graph saved to {path}.")

    def vertices_count(self) -> int:
        """Return the number of vertices in the graph."""
        return self._graph.vcount()

    def get_by_ids(self, ids: List[str]) -> List[Document]:
        """Get vertices by their IDs."""
        if not self._graph:
            self.load_graph()
        vertices = self._graph.vs.select(id_in=ids)
        documents = []
        for vertex in vertices:
            doc = Document(content=vertex["content"], metadata=vertex["metadata"], uid=vertex["id"])
            documents.append(doc)
        return documents
