import copy
import os
from typing import Any, Dict, List, Tuple

import igraph as ig
import numpy as np
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

    def _add_edges(self, edges: List[Tuple[str, str]], weights: List[float]):
        if not self._graph:
            self.load_graph()
        if len(edges) != len(weights):
            raise ValueError("The number of edges and weights must be the same.")
        self._graph.add_edges(edges, attributes={"weight": weights})
        logger.info(f"Added {len(edges)} edges to the graph.")

    def vs(self, name: str = "") -> ig.VertexSeq:
        """The vertex sequence of the graph"""
        if not self._graph:
            self.load_graph()
        if not name:
            return self._graph.vs
        if name in self._graph.vs:
            return self._graph.vs[name]
        raise ValueError(f"Vertex {name} not found in the graph.")

    def delete_vertices(self, label, keys: List[str]) -> None:
        """Delete vertices from the graph by their IDs."""
        if not self._graph:
            self.load_graph()
        if not keys:
            raise ValueError("IDs list is empty.")
        ret: ig.VertexSeq = self._graph.vs.select(name_in=keys, label=label)
        ids = ret.indices

        logger.info(f"Deleting vertices with IDs: {ids}")
        self._graph.delete_vertices(ids)
        logger.info(f"Deleted {len(ids)} vertices from the graph.")

    def select_vertices(self, label: str, attrs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Select vertices based on attributes."""
        if not self._graph:
            self.load_graph()
        if not self._graph.vs.attributes():
            logger.warning("No attributes found in the graph vertices.")
            return []
        logger.info(f"Selecting vertices with attributes: {self._graph.vs.attributes()} with {attrs}")
        attrs["label"] = label
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

    def add_vertices(self, label: str, nodes: List[Dict[str, Any]]):
        """Add vertices to the graph with the given attributes."""
        attributes: Dict[str, List[Any]] = {}
        new_nodes = copy.deepcopy(nodes)
        for node in new_nodes:
            node["label"] = label
            for k, v in node.items():
                if k not in attributes:
                    attributes[k] = []
                attributes[k].append(v)
        if len(attributes) > 0:
            n = len(next(iter(attributes.values())))
            self._graph.add_vertices(n=n, attributes=attributes)
            logger.info(f"Added {n} vertices to the graph with attributes {attributes}.")

    def personalized_pagerank(
        self,
        label: str,
        vertices_with_weight: Dict[str, float],
        damping: float = 0.85,
        top_k: int = 10,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Compute the personalized PageRank of the graph."""
        if not self._graph:
            self.load_graph()
        if not vertices_with_weight:
            logger.warning("No vertices with weights provided. Using all vertices.")
            raise ValueError("No vertices with weights provided.")
        keys = list(vertices_with_weight.keys())
        ret: ig.VertexSeq = self._graph.vs.select(name_in=keys, label=label)
        if not ret:
            return {}
        indexs = ret.indices
        vertices = {k: ret.get_attribute_values(k) for k in ret.attributes()}
        vcount = len(self._graph.vs.select(label=label).get_attribute_values("name"))
        reset_prob = np.zeros(vcount)
        key_to_index = {key: index for key, index in zip(vertices["name"], indexs)}
        for key, weight in vertices_with_weight.items():
            if key in key_to_index:
                reset_prob[key_to_index[key]] = weight
            else:
                logger.warning(f"Key {key} not found in graph vertices.")
        np_reset_prob = np.where(np.isnan(reset_prob) | (reset_prob < 0.0), 0, reset_prob)
        pr_scores = self._graph.personalized_pagerank(
            vertices=range(vcount),
            directed=False,
            damping=damping,
            weights="weight",
            reset=np_reset_prob,  # Corrected the parameter name here
            implementation="prpack",
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

    def vertices_count(self, label: str) -> int:
        """Return the number of vertices in the graph."""
        return self._graph.vs.select(label=label).vcount()

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

    def add_new_edges(self, label, node_to_node_stats: Dict[Tuple[str, str], float]):
        """
        Processes edges from `node_to_node_stats` to add them into a graph object while
        managing adjacency lists, validating edges, and logging invalid edge cases.
        """

        # 初始化邻接表（若后续需要使用）
        # self.graph_adj_list: DefaultDict[str, Dict[str, float]] = defaultdict(dict)
        # self.graph_inverse_adj_list: DefaultDict[str, Dict[str, float]] = defaultdict(dict)
        if not self._graph:
            self.load_graph()
        edge_source_node_keys = []
        edge_target_node_keys = []
        edge_metadata = []
        logger.debug(f"Adding edges to graph with {node_to_node_stats} edges.")
        for edge, weight in node_to_node_stats.items():
            if edge[0] == edge[1]:
                continue
            edge_source_node_keys.append(edge[0])
            edge_target_node_keys.append(edge[1])
            edge_metadata.append({"weight": weight})

        valid_edges: List[Tuple[str, str]] = []
        valid_weights: List[float] = []

        source_nodes = self.select_vertices(label, attrs=dict(name_in=edge_source_node_keys))
        target_nodes = self.select_vertices(label, attrs=dict(name_in=edge_target_node_keys))
        logger.info(
            f"Found {source_nodes} source nodes from {edge_source_node_keys} and {target_nodes} \
                    target nodes from {edge_target_node_keys}."
        )
        existing_node_ids = set([node["name"] for node in source_nodes + target_nodes])

        for source_node_id, target_node_id, edge_d in zip(edge_source_node_keys, edge_target_node_keys, edge_metadata):
            if source_node_id in existing_node_ids and target_node_id in existing_node_ids:
                valid_edges.append((source_node_id, target_node_id))
                valid_weights.append(edge_d["weight"])
            else:
                logger.warning(f"Edge {source_node_id} -> {target_node_id} is not valid.")
        logger.debug(f"Adding {valid_edges} edges to the graph.")
        self._add_edges(valid_edges, valid_weights)
