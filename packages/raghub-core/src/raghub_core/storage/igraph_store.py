import asyncio
import copy
import os
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import igraph as ig
import numpy as np
from langchain_core.runnables.config import run_in_executor
from loguru import logger
from raghub_core.schemas.document import Document
from raghub_core.schemas.graph_model import GraphCommunity, GraphEdge, GraphModel, GraphVertex
from raghub_core.storage.graph import GraphStorage
from raghub_core.utils.graph.graph_helper import GraphHelper


class IGraphStore(GraphStorage):
    name = "igraph"

    def __init__(self, graph_path: str, **kwargs):
        super().__init__(**kwargs)
        self._graph_path = graph_path
        self._graph: ig.Graph = None

    async def init(self):
        # Initialize the IGraph storage
        if not self._graph:
            self.load_graph()
        await asyncio.sleep(0.01)  # Yield control to the event loop
        logger.debug("Graph already loaded.")

    def load_graph(self):
        logger.debug(f"Loading graph from {self._graph_path}.")
        if self._graph_path and os.path.exists(self._graph_path):
            self._graph = ig.Graph.Read_Pickle(self._graph_path)
            logger.info(f"Graph loaded from {self._graph_path}.")
        else:
            self._graph = ig.Graph()
            logger.debug("Initialized a new empty graph.")

    def _add_edges(
        self, edges: List[Tuple[str, str]], weights: List[float], attributes: Optional[Dict[str, List[Any]]] = None
    ) -> None:
        if not self._graph:
            self.load_graph()
        if len(edges) != len(weights):
            raise ValueError("The number of edges and weights must be the same.")
        attributes = attributes or {}
        attributes["weight"] = weights
        # logger.debug(f"Adding {len(edges)} edges with attributes {attributes}.")
        self._graph.add_edges(edges, attributes=attributes)
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

    async def adelete_vertices(self, label, keys: List[str]) -> None:
        return await run_in_executor(None, self.delete_vertices, label, keys)

    def select_vertices(self, label: str, attrs: Dict[str, Any]) -> List[GraphVertex]:
        """Select vertices based on attributes."""
        if not self._graph:
            self.load_graph()
        if not self._graph.vs.attributes():
            logger.warning("No attributes found in the graph vertices.")
            return []
        ret: ig.VertexSeq = self._graph.vs.select(label=label, **attrs)

        indexs = ret.indices
        vertices = {k: ret.get_attribute_values(k) for k in ret.attributes()}
        values: List[List[Any]] = [vertices[key] for key in vertices.keys()]
        values.append(indexs)
        keys = list(vertices.keys())
        keys.append("index")
        result = [GraphVertex(**dict(zip(keys, value))) for value in zip(*values)]

        return result

    async def aselect_vertices(self, label: str, attrs: Dict[str, Any]) -> List[GraphVertex]:
        return await run_in_executor(None, self.select_vertices, label, attrs)

    def edge_seq(self, name: str = "") -> List[str] | Dict[str, List[str]]:
        """The edge sequence of the graph"""
        if not self._graph:
            self.load_graph()
        if not name:
            return self._graph.es
        if name in self._graph.es:
            return self._graph.es[name]
        return []

    def add_vertices(self, label: str, nodes: List[GraphVertex]):
        """Add vertices to the graph with the given attributes."""
        if all(isinstance(node, GraphVertex) for node in nodes):
            nodes = self.graph_vertices2nodes(nodes)
        uids = {node["uid"]: node for node in nodes if "uid" in node}
        existing_nodes = []
        if uids and "label" in self._graph.vs.attributes():
            existing_vertices: ig.VertexSeq = self._graph.vs.select(label_eq=label, uid_in=list(uids.keys()))
            vertices = {k: existing_vertices.get_attribute_values(k) for k in existing_vertices.attributes()}
            for index, existing_vertex in enumerate(existing_vertices):
                uid = vertices["uid"][index]
                metadata = vertices.get("metadata", [])[index] if "metadata" in vertices else {}
                doc_id: List[str] = vertices.get("doc_id", [])[index] if "doc_id" in vertices else []
                doc_id = doc_id if doc_id else []
                description: List[str] = vertices.get("description ", [])[index] if "description " in vertices else []
                facts: List[str] = metadata.get("facts", []) if "facts" in metadata else []
                description.extend(uids[uid].get("description", []))
                doc_id.extend(uids[uid].get("doc_id", []))
                facts.extend(uids[uid].get("facts", []))
                metadata.update(uids[uid].get("metadata", {}))
                metadata["facts"] = list(set(facts))
                vertex_dict = existing_vertex.attributes()
                vertex_dict.update(
                    {
                        "metadata": metadata,
                        "doc_id": list(set(doc_id)),
                        "description": list(set(description)),
                    }
                )
                existing_vertex.update_attributes(**vertex_dict)
                logger.debug(f"Updated existing vertex {uid}.")
                existing_nodes.append(uid)

        attributes: Dict[str, List[Any]] = {}
        new_nodes = copy.deepcopy(nodes)
        new_nodes = [node for node in new_nodes if node["uid"] not in existing_nodes]
        for node in new_nodes:
            logger.debug(f"Adding new vertex {node['uid']} named {node['name']} with label {label}.")
            node["label"] = label
            for k, v in node.items():
                if k not in attributes:
                    attributes[k] = []
                attributes[k].append(v)
        if len(attributes) > 0:
            n = len(next(iter(attributes.values())))
            self._graph.add_vertices(n=n, attributes=attributes)
            # logger.info(f"Added {n} vertices to the graph with attributes {attributes}.")
        self._save()

    async def aadd_vertices(self, label: str, nodes: List[GraphVertex]):
        return await run_in_executor(None, self.add_vertices, label, nodes)

    async def aadd_graph_vertices(self, label: str, nodes: List[GraphVertex]):
        await run_in_executor(None, self.add_vertices, label, nodes)

    def _graph_edges2edges(self, edges: List[GraphEdge]) -> List[Tuple[Dict[Tuple[str, str], float], Dict[str, Any]]]:
        """
        Convert a list of GraphEdge objects to a list of dictionaries.

        Args:
            edges: A list of GraphEdge objects.

        Returns:
            A list of dictionaries representing the edges.
        """
        return [
            (
                {
                    (edge.source, edge.target): edge.weight,
                },
                edge.model_dump(),
            )
            for edge in edges
        ]

    async def aadd_graph_edges(self, label: str, edges: List[GraphEdge]):
        """
        Add graph edges to the storage.

        Args:
            label (str): The label of the edges.
            edges (List[GraphVertex]): A list of GraphVertex objects to be added.

        Returns:
            None
        """
        if not self._graph:
            self.load_graph()
        if not edges:
            raise ValueError("No edges provided.")
        # nodes_to_nodes = {key:value for key, value in edges_nodes}
        await self.aadd_new_edges(label, edges)
        self._save()

    def _save(self):
        """Save the graph to the specified path."""
        if not self._graph:
            self.load_graph()
        if not self._graph_path:
            raise ValueError("Graph path is not set.")
        self._graph.write_pickle(self._graph_path)
        logger.info(f"Graph saved to {self._graph_path}.")

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
        vcount = len(self._graph.vs.select(label_eq=label).get_attribute_values("name"))
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

    async def apersonalized_pagerank(
        self,
        label: str,
        vertices_with_weight: Dict[str, float],
        damping: float = 0.85,
        top_k: int = 10,
        **kwargs: Any,
    ) -> Dict[str, float]:
        return await run_in_executor(
            None, self.personalized_pagerank, label, vertices_with_weight, damping, top_k, **kwargs
        )

    def save(self, path: str) -> None:
        """Save the graph to a file."""
        path_dir = os.path.dirname(path)
        os.makedirs(path_dir, exist_ok=True)
        self._graph.write_pickle(path)
        logger.info(f"Graph saved to {path}.")

    async def asave(self, path: str) -> None:
        return await run_in_executor(None, self.save, path)

    def vertices_count(self, label: str) -> int:
        """Return the number of vertices in the graph."""
        return self._graph.vs.select(label_eq=label).vcount()

    async def avertices_count(self, label: str) -> int:
        return await run_in_executor(None, self.vertices_count, label)

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

    async def aget_by_ids(self, ids: List[str]) -> List[Document]:
        return await run_in_executor(None, self.get_by_ids, ids)

    def _update_attr(self, edge: GraphEdge | GraphVertex, existing_edge: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the edge attributes with the new edge attributes.
        """
        updated_attrs = existing_edge.copy()
        for k, v in edge.model_dump().items():
            if k not in updated_attrs:
                updated_attrs[k] = v
            elif isinstance(updated_attrs[k], list) and isinstance(v, list):
                updated_attrs[k].extend(v)
                updated_attrs[k] = list(set(updated_attrs[k]))
            elif isinstance(updated_attrs[k], dict) and isinstance(v, dict):
                updated_attrs[k].update(v)
            else:
                updated_attrs[k] = v
        return updated_attrs

    def add_new_edges(self, label: str, edges: List[GraphEdge]):
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
        for e in edges:
            if e.source == e.target:
                continue
            edge_source_node_keys.append(e.source)
            edge_target_node_keys.append(e.target)
            edge_metadata.append({"weight": e.weight})
        edge_keys_to_attrs = {f"{edge.source}#{edge.target}": edge for edge in edges}
        valid_edges: List[Tuple[str, str]] = []
        valid_weights: List[float] = []

        source_nodes = self.select_vertices(label, attrs=dict(name_in=edge_source_node_keys))
        target_nodes = self.select_vertices(label, attrs=dict(name_in=edge_target_node_keys))
        existing_edge_uids = set()
        if "uid" in self._graph.es.attributes():
            existings_edges = self._graph.es.select(uid_in=[edge.uid for edge in edges])
            existing_edge_uids = set(existings_edges.get_attribute_values("uid")) if existings_edges else set()
        existing_node_ids = set([node.name for node in source_nodes + target_nodes])
        index = 0
        attrs: Dict[str, List[Any]] = {}
        for source_node_id, target_node_id, edge_d in zip(edge_source_node_keys, edge_target_node_keys, edge_metadata):
            if source_node_id in existing_node_ids and target_node_id in existing_node_ids:
                valid_edges.append((source_node_id, target_node_id))
                valid_weights.append(edge_d["weight"])
                edge: GraphEdge = edge_keys_to_attrs.get(f"{source_node_id}#{target_node_id}")
                if edge and edge.uid in existing_edge_uids:
                    # 如果边已经存在，更新边的属性
                    existing_edge = existings_edges.select(uid_eq=edge.uid)
                    if existing_edge:
                        existing_edge = existing_edge[0]
                        updated_attrs = existing_edge.attributes()
                        existing_edge.update_attributes(**self._update_attr(edge, updated_attrs))
                        logger.debug(f"Updated existing edge {edge.uid} from {source_node_id} to {target_node_id}.")
                else:
                    for k, v in edge.model_dump().items():
                        if k not in attrs:
                            attrs[k] = []
                        attrs[k].append(v)
            else:
                logger.warning(f"Edge {source_node_id} -> {target_node_id} is not valid.")
                logger.warning(
                    f"""Source node {source_node_id in existing_node_ids} or 
                    target node {target_node_id in existing_node_ids} not found in the graph."""
                )
            index += 1

        logger.debug(f"Adding {len(valid_edges)} edges to the graph.")
        self._add_edges(valid_edges, valid_weights, attrs)

    async def aadd_new_edges(self, label: str, edges: List[GraphEdge]):
        return await run_in_executor(None, self.add_new_edges, label, edges)

    async def _discover_communities(self, label, resolution_parameter=0.8, beta=0.1, n_iterations=-1, **kwargs):
        clustering = await run_in_executor(
            None,
            self._graph.community_leiden,
            resolution_parameter=resolution_parameter,
            beta=beta,
            weights="weight",
            n_iterations=n_iterations,
            **kwargs,
        )  # 使用 igraph 的 Leiden 算法
        membership = clustering.membership
        # community_nodes = {}
        self._graph.vs["cid"] = [f"community_{cid}" for cid in membership]
        for i, _ in enumerate(self._graph.vs):
            community_id = f"community_{membership[i]}"
            self._graph.vs[i].update_attributes(cid=community_id)
            logger.debug(f"Vertex {self._graph.vs[i]['uid']} assigned to community {community_id}.")
        community_ids = list(set(f"community_{cid}" for cid in membership))
        await run_in_executor(None, self._save)
        return community_ids

    async def discover_communities(
        self, label, resolution_parameter=0.8, beta=0.1, n_iterations=-1, **kwargs
    ) -> List[str]:
        """Run community discovery with Leiden using igraph."""
        communities = await self._discover_communities(label, resolution_parameter, beta, n_iterations, **kwargs)
        # community_ids = [community.cid for community in communities]
        return communities

    async def get_community(self, label: str, community_id: str) -> GraphCommunity:
        """Get vertices in a specific community."""
        if not self._graph:
            self.load_graph()
        if not community_id:
            raise ValueError("Community ID is required.")
        # self._graph.vs.select(label_eq=label)
        vs = await run_in_executor(None, self._graph.vs.select, label=label, cid=community_id)

        if len(vs) == 0:
            logger.warning(f"No vertices found in community {community_id} with label {label}.")
            return None
        vertices: List[GraphVertex] = []
        edges: List[GraphEdge] = []
        existing_vs = set()
        for vertex in vs:
            attr = vertex.attributes()
            # logger.debug(f"Processing vertex {attr} in community {community_id}.")
            if attr.get("label") == label and attr["uid"] not in existing_vs:
                vertices.append(GraphVertex(**vertex.attributes()))
                existing_vs.add(attr["uid"])
        vertices_uids = [vertex.uid for vertex in vertices]
        if not vertices_uids:
            logger.warning(f"No vertices with UID found in community {community_id} with label {label}.")
            return None
        edges_seq: ig.EdgeSeq = await run_in_executor(
            None, self._graph.es.select, source_in=vertices_uids, target_in=vertices_uids, label_eq=label
        )
        edges_id = []
        for edge in edges_seq:
            attributes = edge.attributes()
            attributes["source"] = self._graph.vs[edge.source]["uid"]
            attributes["target"] = self._graph.vs[edge.target]["uid"]
            if "uid" in attributes and attributes["uid"] in edges_id:
                continue
            edges_id.append(attributes["uid"])
            edges.append(GraphEdge(**attributes))
        # Duplicate vertices and edges to create a GraphModel

        graph = GraphModel(vertices=vertices, edges=edges)
        return GraphCommunity(
            cid=community_id,
            name=f"Community {community_id}",
            graph=graph,
        )

    async def aupdate_vertices(self, label: str, nodes: List[GraphVertex]) -> List[GraphVertex]:
        """Update vertices in the graph with the given attributes."""
        if not self._graph:
            self.load_graph()
        if not nodes:
            raise ValueError("No nodes provided for update.")
        uids = {node.uid: node for node in nodes}
        if not uids:
            logger.warning("No valid nodes provided for update. Ensure nodes have 'uid' attributes.")
            return []
        if not self._graph.vs.attributes():
            logger.warning("No attributes found in the graph vertices.")
            return []
        updated_keys = []
        if uids and "label" in self._graph.vs.attributes():
            existing_vertices: ig.VertexSeq = await run_in_executor(
                None, self._graph.vs.select, label_eq=label, uid_in=list(uids.keys())
            )
            vertices = {k: existing_vertices.get_attribute_values(k) for k in existing_vertices.attributes()}
            for index, existing_vertex in enumerate(existing_vertices):
                uid = vertices["uid"][index]
                vertex_dict = existing_vertex.attributes()
                attr = self._update_attr(uids[uid], vertex_dict)
                existing_vertices[index].update_attributes(**attr)
                logger.debug(f"Updated existing vertex {uid}.")
                updated_keys.append(uid)
        if not updated_keys:
            logger.warning("No vertices were updated. Ensure the nodes exist in the graph.")
            return []
        nodes = []
        for key in updated_keys:
            node = uids.get(key)
            if node:
                nodes.append(node)
        if not nodes:
            logger.warning("No vertices were updated. Ensure the nodes exist in the graph.")
            return []
        logger.info(f"Updated {len(nodes)} vertices in the graph.")
        # Save the graph after updating vertices
        await run_in_executor(None, self._save)
        return nodes

    async def aupdate_edges(self, label: str, edges: List[GraphEdge]) -> List[GraphEdge]:
        """Update edges in the graph with the given attributes."""
        if not self._graph:
            self.load_graph()
        if not edges:
            raise ValueError("No edges provided for update.")
        updated_keys = []
        edge_keys_to_attrs = {f"{edge.source}#{edge.relation}#{edge.target}": edge for edge in edges}
        if edge_keys_to_attrs and "label" in self._graph.es.attributes():
            existing_edges: ig.EdgeSeq = await run_in_executor(
                None, self._graph.es.select, label_eq=label, uid_in=list(edge_keys_to_attrs.keys())
            )
            for index, existing_edge in enumerate(existing_edges):
                attr = existing_edge.attributes()
                edge_key = f"{attr['source']}#{attr['relation']}#{attr['target']}"
                if edge_key in edge_keys_to_attrs:
                    attr = self._update_attr(edge_keys_to_attrs[edge_key], existing_edge.attributes())
                    existing_edges[index].update_attributes(**attr)
                    logger.debug(f"Updated existing edge {edge_key}.")
                    updated_keys.append(edge_key)
        if not updated_keys:
            logger.warning("No edges were updated. Ensure the edges exist in the graph.")
            return []
        edges = []
        for edge_key in updated_keys:
            edge = edge_keys_to_attrs.get(edge_key)
            if edge:
                edges.append(edge)
        if not edges:
            logger.warning("No edges were updated. Ensure the edges exist in the graph.")
            return []
        logger.info(f"Updated {len(edges)} edges in the graph.")
        await run_in_executor(None, self._save)
        return edges

    async def aselect_edges(self, label: str, attrs: Dict[str, Any]) -> List[GraphEdge]:
        """Select an edge by source and target."""
        if not self._graph:
            self.load_graph()
        if "label" not in self._graph.es.attributes():
            logger.warning("No label attribute found in the graph edges.")
            return []
        ret: ig.EdgeSeq = await run_in_executor(None, self._graph.es.select, label_eq=label, **attrs)
        if not ret:
            return []
        edges: List[GraphEdge] = []
        for index, edge in enumerate(ret):
            attributes = edge.attributes()
            edges.append(GraphEdge(**attributes))
        return edges

    def _get_edge_type(self, edge_idx: int) -> str:
        """获取边的类型"""
        edge = self._graph.es[edge_idx]
        return edge.attributes().get("relation_type", "")

    def _get_edge_relation(self, edge_idx: int) -> str:
        """获取边的关系"""
        edge = self._graph.es[edge_idx]
        return edge.attributes().get("relation", "")

    def _bfs_multi_hop(
        self,
        start_nodes: List[str],
        relation_path: List[str],
        max_hops: int = 3,
        rel_type: Optional[str] = None,
        avoid_cycles: bool = True,
        max_paths: int = 5,
        continuous=False,
    ) -> GraphModel:
        """
        使用BFS进行多跳搜索

        Args:
            start_nodes: 起始节点列表
            relation_path: 关系路径
            max_hops: 最大跳数
            rel_type: 关系类型过滤
            avoid_cycles: 是否避免循环
            max_paths: 最大路径数

        Returns:
            搜索结果
        """
        verteices: List[GraphVertex] = []
        edges: List[GraphEdge] = []
        existing_vs = set()
        existing_es = set()
        # results = {
        #     "paths": [],
        #     "end_nodes": set(),
        #     # 'statistics': {
        #     #     'total_paths': 0,
        #     #     'unique_end_nodes': 0,
        #     #     'path_lengths': defaultdict(int)
        #     # }
        # }

        # BFS队列：(当前节点, 路径, 已访问节点集合, 当前跳数, 路径中的关系)
        queue: deque = deque()

        # 初始化队列
        for start_node in start_nodes:
            start_node_index = self._graph.vs.find(uid=start_node).index
            if avoid_cycles:
                visited = {start_node_index}
            else:
                visited = set()
            queue.append((start_node_index, [start_node_index], visited, 0, []))

        path_count = 0
        max_hops = max_hops if not continuous else len(relation_path) + 1
        while queue and path_count < max_paths:
            current_node, path, visited, hop_count, path_relations = queue.popleft()

            if hop_count >= max_hops:
                if len(path) > 1:  # 确保路径长度大于1
                    # results["paths"].append(
                    #     {
                    #         "path": path.copy(),
                    #         "relations": path_relations.copy(),
                    #         "length": len(path) - 1,
                    #         "end_node": path[-1],
                    #     }
                    # )
                    # results["end_nodes"].add(path[-1])
                    path_count += 1
                continue

            # 获取当前节点的所有邻居
            # current_node_index = self._graph.vs.find(uid=).index
            neighbors = self._graph.neighbors(current_node)
            incident_edges = self._graph.incident(current_node)

            for neighbor, edge_idx in zip(neighbors, incident_edges):
                # 避免循环
                if avoid_cycles and neighbor in visited:
                    continue

                # 获取边的类型
                edge_type = self._get_edge_type(edge_idx)

                # 关系类型过滤
                if rel_type and edge_type != rel_type:
                    continue

                # 关系路径过滤
                if relation_path and hop_count < len(relation_path):
                    relation = self._get_edge_relation(edge_idx)
                    if not continuous:
                        if relation not in relation_path:
                            continue
                    else:
                        # 如果是连续关系，检查是否在关系路径中
                        if relation != relation_path[hop_count]:
                            continue

                # 构建新路径
                new_path = path + [neighbor]
                new_path_relations = path_relations + [edge_type]

                if avoid_cycles:
                    new_visited = visited | {neighbor}
                else:
                    new_visited = visited.copy()
                vertex = self._graph.vs[neighbor]
                edge = self._graph.es[edge_idx]
                # 检查是否已经存在该顶点和边
                if vertex.attributes()["uid"] not in existing_vs:
                    existing_vs.add(vertex.attributes()["uid"])
                    # 仅添加新的顶点
                    verteices.append(GraphVertex(**vertex.attributes()))
                if edge.attributes()["uid"] not in existing_es:
                    existing_es.add(edge.attributes()["uid"])
                    # 仅添加新的边
                    edges.append(
                        GraphEdge(
                            **edge.attributes(),
                        )
                    )
                # 如果已经达到指定的关系路径长度，记录结果
                if relation_path and len(new_path_relations) == len(relation_path):
                    # results["paths"].append(
                    #     {
                    #         "path": new_path.copy(),
                    #         "relations": new_path_relations.copy(),
                    #         "length": len(new_path) - 1,
                    #         "end_node": neighbor,
                    #     }
                    # )
                    # results["end_nodes"].add(neighbor)
                    path_count += 1
                    if path_count >= max_paths:
                        break
                else:
                    # 继续搜索
                    queue.append((neighbor, new_path, new_visited, hop_count + 1, new_path_relations))

        return GraphModel(vertices=verteices, edges=edges)

    async def multi_hop_search(
        self,
        label: str,
        start_nodes_id: List[str],
        relation_path: List[str] = [],
        max_hops: int = 3,
        rel_type: Optional[str] = None,
        max_paths: int = 5,
        continuous=False,
    ) -> Dict[str, Any] | None:
        """
        多跳实体搜索
        给定起始实体和关系路径，搜索与起始实体存在多跳关系的所有实体

        Args:
            label: 节点标签
            namespace: 目标命名空间
            start_nodes_id: 起始节点ID列表
            relation_path: 关系路径列表，指定每一跳的关系类型
            rel_type: 通用关系类型（include or relation ）
            avoid_cycles: 是否避免环路
            max_paths: 最大路径数量限制
            limit: 返回结果限制
        """
        # avoid_cycles: bool = (True,)
        if not self._graph:
            self.load_graph()
        graph = await run_in_executor(
            None,
            self._bfs_multi_hop,
            start_nodes=start_nodes_id,
            relation_path=relation_path,
            max_hops=max_hops if not relation_path else len(relation_path),
            rel_type=rel_type,
            avoid_cycles=True,
            max_paths=max_paths,
            continuous=continuous,
        )

        graph.vertices = [v for v in graph.vertices if v.label == label]
        graph.edges = [e for e in graph.edges if e.label == label]
        if not graph.vertices:
            logger.warning(f"No vertices found with label '{label}' in the graph.")
            return None
        return graph

    async def freestyle_search(self, label: str, entities: List[str], max_hops: int = 3) -> GraphModel | None:
        """
        策略5: FreestyleQuestion
        搜索所有相关实体及以其为中心的子图
        entities: 查询相关的实体列表
        max_hops: 最大跳数，默认为2
        """
        uids = [GraphHelper.generate_vertex_id(entity) for entity in entities]
        valid_entities = await self.aselect_vertices(label, {"uid_in": uids})

        # 收集所有相关顶点
        relevant_vertices = set()

        for entity in valid_entities:
            vertex_id = entity.index
            relevant_vertices.add(entity.index)

            # 添加实体详细信息
            # entity_details[entity] = self.single_entity_search(entity)

            # 进行多跳搜索
            current_level = {vertex_id}
            visited = {vertex_id}

            for hop in range(max_hops):
                next_level = set()

                for v in current_level:
                    # 获取邻居（出边和入边）

                    neighbors = set(self._graph.neighbors(v, mode="out"))
                    neighbors.update(set(self._graph.neighbors(v, mode="in")))

                    for neighbor in neighbors:
                        if neighbor not in visited:
                            next_level.add(neighbor)
                            relevant_vertices.add(neighbor)
                            visited.add(neighbor)

                current_level = next_level
                if not current_level:
                    break

        # 构建子图
        subgraph = await run_in_executor(None, self._graph.induced_subgraph, list(relevant_vertices))

        # 提取子图中的实体和关系
        subgraph_entities: List[GraphVertex] = []

        for v_id in relevant_vertices:
            vertex = self._graph.vs[v_id]
            subgraph_entities.append(GraphVertex(**vertex.attributes()))
        edges: List[GraphEdge] = []
        # 提取子图中的边
        for edge in subgraph.es:
            e = GraphEdge(**edge.attributes())
            edges.append(e)
        # 创建GraphModel对象
        subgraph_entities = [v for v in subgraph_entities if v.label == label]
        if not subgraph_entities:
            logger.warning(f"No vertices found with label '{label}' in the subgraph.")
            return None
        graph_model = GraphModel(vertices=subgraph_entities, edges=edges)
        return graph_model

    def _format_path_string(self, path_entities: List[str], relations: List[str]) -> str:
        """格式化路径字符串"""
        if len(path_entities) < 2:
            return str(path_entities)

        path_parts = [path_entities[0]]
        for i in range(len(relations)):
            if i + 1 < len(path_entities):
                relation_str = str(relations[i])
                if hasattr(relations[i], "value"):
                    relation_str = relations[i]
                path_parts.extend([f"--[{relation_str}]-->", path_entities[i + 1]])

        return " ".join(path_parts)

    async def aupsert_virtices(self, unique_name: str, vertices: List[GraphVertex]) -> List[GraphVertex]:
        """
        Add vertices to the graph storage system.
        Args:
            unique_name (str): Name of the index to which vertices will be added.
            vertices (List[GraphVertex]): List of documents to add as vertices.
        Returns:
            List[GraphVertex]: List of added vertices.
        """
        # Add vertices to the graph storage
        existing_vertices: List[GraphVertex] = await self.aselect_vertices(
            unique_name, dict(uid_in=[v.uid for v in vertices])
        )
        filter_existing_vertices = (
            [v for v in vertices if v.uid in [ev.uid for ev in existing_vertices]] if existing_vertices else []
        )
        if filter_existing_vertices:
            await self.aupdate_vertices(unique_name, filter_existing_vertices)
        if not existing_vertices:
            logger.warning(f"No existing vertices found for index: {unique_name}")
            return await self.aadd_graph_vertices(unique_name, vertices)
        # Filter out existing vertices
        new_vertices = [v for v in vertices if v.uid not in [ev.uid for ev in existing_vertices]]
        if not new_vertices:
            logger.warning(f"No new vertices to add for index: {unique_name}")
            return existing_vertices
        # upsert existing vertices
        return await self.aadd_graph_vertices(unique_name, new_vertices)

    async def aupsert_edges(self, unique_name: str, edges: List[GraphEdge]) -> List[GraphEdge]:
        """
        Add edges to the graph storage system.
        Args:
            unique_name (str): Name of the index to which edges will be added.
            edges (List[GraphEdge]): List of documents to add as edges.
        Returns:
            List[GraphEdge]: List of added edges.
        """
        existing_edges: List[GraphEdge] = await self.aselect_edges(
            unique_name, dict(uid_in=[edge.uid for edge in edges])
        )
        filter_existing_edges = (
            [edge for edge in edges if edge.uid in [ee.uid for ee in existing_edges]] if existing_edges else []
        )
        if filter_existing_edges:
            await self.aupdate_edges(unique_name, filter_existing_edges)
        if not existing_edges:
            logger.warning(f"No existing edges found for index: {unique_name}")
            return await self.aadd_graph_edges(unique_name, edges)
        new_edges = [edge for edge in edges if edge.uid not in [ee.uid for ee in existing_edges]]
        if not new_edges:
            logger.warning(f"No new edges to add for index: {unique_name}")
            return existing_edges
        # Add edges to the graph storage
        return await self.aadd_graph_edges(unique_name, new_edges)
