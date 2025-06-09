import asyncio
import json
from typing import Any, AsyncIterable, Dict, List, Optional, Tuple

from loguru import logger
from raghub_core.schemas.document import Document
from raghub_core.schemas.graph_model import GraphCommunity, GraphEdge, GraphModel, GraphVertex, RelationType
from raghub_core.storage.graph import GraphStorage


class Neo4jGraphStorage(GraphStorage):
    name = "neo4j"

    def __init__(
        self,
        url: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._uri = url
        self._user = username
        self._password = password
        self._graph = database
        try:
            from neo4j import AsyncDriver
        except ImportError:
            raise ImportError("Neo4j is not installed. Please install it using `pip install neo4j`.")

        self._driver: Optional[AsyncDriver] = None

    async def init(self):
        """Initialize Neo4j driver connection."""
        if not self._driver:
            try:
                from neo4j import AsyncGraphDatabase

                await asyncio.sleep(0.01)  # Yield control to the event loop
                self._driver = AsyncGraphDatabase.driver(self._uri, auth=(self._user, self._password))
                logger.debug("Neo4j driver initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize Neo4j driver: {e}")
                raise

    async def close(self):
        """Close Neo4j driver connection."""
        if self._driver:
            await self._driver.close()
            logger.debug("Neo4j driver connection closed.")

    async def aadd_new_edges(self, label: str, edges: List[GraphEdge]):
        """Add new edges to the graph."""
        if not self._driver:
            raise ValueError("Neo4j driver is not initialized. Call init() first.")
        try:
            async with self._driver.session() as session:
                # 按关系类型分组
                from collections import defaultdict

                edges_by_type = defaultdict(list)

                for edge in edges:
                    relation_type = edge.relation_type
                    # 确保关系类型在枚举中定义
                    if relation_type in [rt.value for rt in RelationType]:
                        edges_by_type[relation_type.value].append(edge)
                    else:
                        # 默认使用 RELATED 类型
                        edges_by_type[RelationType.RELATION.value].append(edge)

                # 为每种关系类型执行单独的查询
                for relation_type, type_edges in edges_by_type.items():
                    query = ""
                    if relation_type == RelationType.INCLUDE.value:
                        query = """
                        UNWIND $edges AS edge
                        MATCH (a {uid: edge.source}), (b {uid: edge.target})
                        CREATE (a)-[r:CONTAINS {
                            weight: edge.weight,
                            source_content: edge.source_content,
                            target_content: edge.target_content,
                            relation: edge.relation,
                            description: edge.description,
                            metadata: edge.metadata,
                            uid: edge.uid,
                            label: edge.label,
                            source: edge.source,
                            target: edge.target,
                            relation_type: edge.relation_type
                        }]->(b)
                        RETURN r
                        """
                    elif relation_type == RelationType.RELATION.value:
                        query = """
                        UNWIND $edges AS edge
                        MATCH (a {uid: edge.source}), (b {uid: edge.target})
                        CREATE (a)-[r:RELATED_TO {
                            weight: edge.weight,
                            source_content: edge.source_content,
                            target_content: edge.target_content,
                            relation: edge.relation,
                            description: edge.description,
                            metadata: edge.metadata,
                            uid: edge.uid,
                            label: edge.label,
                            source: edge.source,
                            target: edge.target,
                            relation_type: edge.relation_type
                        }]->(b)
                        RETURN r
                        """

                    edge_dicts = self._dumps(type_edges)
                    r = await session.run(query, edges=edge_dicts)
                    logger.debug(f"Adding edges result for relation type {relation_type}: {r}:{type(r)}")
                    # async for record in r:
                    #     # Log each created edge
                    #     logger.debug(f"Created edge: {record['r']} with relation type {relation_type}")
                    logger.info(f"Successfully added {len(edges)} edges to the graph")

        except Exception as e:
            logger.error(f"Error adding edges: {e}")
            raise

    async def aadd_graph_edges(self, label: str, edges: List[GraphEdge]):
        """Add graph edges (same as aadd_new_edges for Neo4j)."""
        await self.aadd_new_edges(label, edges)

    async def aget_by_ids(self, label: str, ids: List[str]) -> List[Document]:
        """Get documents by their IDs."""
        if not self._driver:
            raise ValueError("Neo4j driver is not initialized. Call init() first.")

        query = f"""
        UNWIND $ids AS id
        MATCH (n:{label} {{uid: id}})
        RETURN properties(n) as props
        """

        try:
            async with self._driver.session() as session:
                result = await session.run(query, ids=ids)
                documents = []
                async for record in result:
                    vertex_props = record["props"]
                    doc = self._vertex_to_doc(vertex_props)
                    documents.append(doc)
                return documents
        except Exception as e:
            logger.error(f"Error getting documents by IDs: {e}")
            raise

    def _dumps(self, nodes: List[GraphVertex] | List[GraphEdge]) -> List[Dict[str, Any]]:
        """Convert metadata dictionary to a string representation."""
        dump_nodes = []
        for node in nodes:
            v: Dict[str, Any] = {}
            for k, val in node.model_dump().items():
                if not isinstance(val, (str, float, int, bool)):
                    # Convert list to a string representation
                    v[k] = json.dumps(val, ensure_ascii=False)
                else:
                    v[k] = val
            dump_nodes.append(v)
        # print(f"ret={ret}")
        return dump_nodes

    def _loads(self, nodes: List[Dict[str, Any]]) -> List[GraphVertex] | List[GraphEdge]:
        """Convert metadata string representation back to a dictionary."""
        if not nodes:
            return []
        for node in nodes:
            # Convert string representation back to dictionary
            for k, v in node.items():
                if isinstance(v, str):
                    try:
                        node[k] = json.loads(v)
                    except json.JSONDecodeError:
                        # If it fails, keep it as a string
                        pass
        if "relation_type" in nodes[0]:
            # If the first node has a relation_type, treat as edges
            return [GraphEdge(**node) for node in nodes]
        return [GraphVertex(**node) for node in nodes]

    async def aadd_vertices(self, label: str, nodes: List[GraphVertex]):
        """Add vertices to the graph."""
        if not self._driver:
            raise ValueError("Neo4j driver is not initialized. Call init() first.")
        if not nodes:
            raise ValueError("Empty list of vertices provided")

        query = f"""
        UNWIND $nodes AS node
        MERGE (n:{label} {{uid: node.uid}})
        ON CREATE SET
            n.name = node.name, 
            n.content = node.content, 
            n.namespace = node.namespace,
            n.uid = node.uid, 
            n.doc_id = node.doc_id,
            n.description = node.description,
            n.label = node.label,
            n.index = node.index,
            n.embedding = node.embedding,
            n.metadata = node.metadata
        ON MATCH SET
            n.content = node.content,
            n.description = node.description,
            n.metadata = node.metadata
        """

        try:
            async with self._driver.session() as session:
                await session.run(query, nodes=self._dumps(nodes))
            logger.info(f"Added {len(nodes)} vertices")
        except Exception as e:
            logger.error(f"Error adding vertices: {e}")
            raise

    async def aadd_graph_vertices(self, label: str, nodes: List[GraphVertex]):
        """Add graph vertices (same as aadd_vertices for Neo4j)."""
        await self.aadd_vertices(label, nodes)

    async def apersonalized_pagerank(
        self,
        label: str,
        vertices_with_weight: Dict[str, float],
        damping: float = 0.85,
        top_k: int = 10,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Calculate personalized PageRank scores for top-k vertices."""
        if not self._driver:
            raise ValueError("Neo4j driver is not initialized. Call init() first.")

        try:
            # Check if graph exists
            async with self._driver.session() as session:
                graph_name = f"{label}_ppr_graph"
                await session.run(
                    f"CALL gds.graph.drop('{graph_name}', false)",
                )
                create_graph_query = f"""
                MATCH (source:{label})-[r]->(target:{label})
                RETURN gds.graph.project(
                $graph_name,
                  source,
                  target,
                {{ relationshipProperties: r {{ .weight }} }},
                {{ undirectedRelationshipTypes: ['*'] }}
                )"""
                await session.run(create_graph_query, graph_name=graph_name)
                logger.info(f"Graph '{graph_name}' created successfully.")

                query = f"""
                UNWIND $pairs AS pair
                MATCH (n:{label} {{uid: pair.name}})
                WITH collect([n, pair.weight]) AS sourceNodes
                CALL gds.pageRank.stream(
                    $graphName,
                    {{
                        maxIterations: 20,
                        dampingFactor: $damping,
                        sourceNodes: sourceNodes
                    }}
                )
                YIELD nodeId, score
                RETURN gds.util.asNode(nodeId).uid AS name, score
                ORDER BY score DESC, name ASC
                LIMIT $topK
                """
                logger.debug(f"Running personalized PageRank vertices_with_weight: {vertices_with_weight}")
                result = await session.run(
                    query,
                    pairs=[{"name": name, "weight": value} for name, value in vertices_with_weight.items()],
                    damping=damping,
                    topK=top_k,
                    graphName=graph_name,
                )
                return {record["name"]: record["score"] async for record in result}

        except Exception as e:
            logger.error(f"Error in personalized_pagerank: {e}")
            raise

    async def aupdate_vertices(self, label: str, nodes: List[GraphVertex]) -> List[GraphVertex]:
        """Update existing vertices in the graph."""
        if not self._driver:
            raise ValueError("Neo4j driver is not initialized. Call init() first.")

        query = f"""
        UNWIND $nodes AS node
        MATCH (n:{label} {{uid: node.uid}})
        SET n.name = node.name,
            n.content = node.content,
            n.description = node.description,
            n.metadata = node.metadata,
            n.namespace = node.namespace,
            n.doc_id = node.doc_id,
            n.label = node.label,
            n.index = node.index,
            n.embedding = node.embedding
        RETURN properties(n) as props
        """

        try:
            async with self._driver.session() as session:
                result = await session.run(query, nodes=self._dumps(nodes))
                updated_vertices = []
                async for record in result:
                    updated_vertices.append(self._loads([record["props"]])[0])
                logger.info(f"Updated {len(updated_vertices)} vertices")
                return updated_vertices
        except Exception as e:
            logger.error(f"Error updating vertices: {e}")
            raise

    async def aupdate_edges(self, label: str, edges: List[GraphEdge]) -> List[GraphEdge]:
        """Update existing edges in the graph."""
        if not self._driver:
            raise ValueError("Neo4j driver is not initialized. Call init() first.")

        updated_edges = []
        try:
            async with self._driver.session() as session:
                for edge in edges:
                    query = """
                    MATCH ()-[r]->()
                    WHERE r.uid = $edge.uid
                    SET r.weight = $edge.weight,
                        r.description = $edge.description,
                        r.metadata = $edge.edge_metadata,
                        r.label = $edge.label_param

                    RETURN properties(r) as props
                    """
                    e = self._dumps([edge])[0]
                    result = await session.run(
                        query,
                        edge=e,
                    )

                    async for record in result:
                        # Reconstruct GraphEdge from properties
                        props = record["props"]
                        updated_edge = self._loads([props])[0]
                        updated_edges.append(updated_edge)

                logger.info(f"Updated {len(updated_edges)} edges")
                return updated_edges
        except Exception as e:
            logger.error(f"Error updating edges: {e}")
            raise

    async def aupsert_virtices(self, unique_name: str, vertices: List[GraphVertex]) -> List[GraphVertex]:
        """Upsert vertices in the graph."""
        existing_vertices: List[GraphVertex] = await self.aselect_vertices(
            unique_name, dict(uid_in=[v.uid for v in vertices])
        )
        ready_updated_vertices = (
            [v for v in vertices if v.uid in [ev.uid for ev in existing_vertices]] if existing_vertices else []
        )
        if ready_updated_vertices:
            ready_updated_vertices_dict = {v.uid: v for v in ready_updated_vertices}
            existing_dict = {v.uid: v for v in existing_vertices}
            updated_dicts = self.update_graph_attr(ready_updated_vertices_dict, existing_dict)
            await self.aupdate_vertices(unique_name, [GraphVertex(**v) for v in updated_dicts])
        if not existing_vertices:
            logger.warning(f"No existing vertices found for index: {unique_name} by uids: {[v.uid for v in vertices]}")
            return await self.aadd_graph_vertices(unique_name, vertices)
        # Filter out existing vertices
        new_vertices = [v for v in vertices if v.uid not in [ev.uid for ev in existing_vertices]]
        if not new_vertices:
            logger.warning(f"No new vertices to add for index: {unique_name}")
            return existing_vertices
        # upsert existing vertices
        return await self.aadd_graph_vertices(unique_name, new_vertices)

    async def aupsert_edges(self, unique_name: str, edges: List[GraphEdge]) -> List[GraphEdge]:
        """Upsert edges in the graph."""
        if not self._driver:
            raise ValueError("Neo4j driver is not initialized. Call init() first.")

        try:
            existing_edges: List[GraphEdge] = await self.aselect_edges(unique_name, dict(uid_in=[v.uid for v in edges]))
            ready_updated_vertices = (
                [v for v in edges if v.uid in [ev.uid for ev in existing_edges]] if existing_edges else []
            )
            if ready_updated_vertices:
                ready_updated_vertices_dict = {v.uid: v for v in ready_updated_vertices}
                existing_dict = {v.uid: v for v in existing_edges}
                updated_dicts = self.update_graph_attr(ready_updated_vertices_dict, existing_dict)
                await self.aupdate_edges(unique_name, [GraphEdge(**v) for v in updated_dicts])
            if not existing_edges:
                logger.warning(f"No existing vertices found for index: {unique_name}")
                return await self.aadd_graph_edges(unique_name, edges)
            # Filter out existing vertices
            new_edges = [v for v in edges if v.uid not in [ev.uid for ev in existing_edges]]
            if not new_edges:
                logger.warning(f"No new vertices to add for index: {unique_name}")
                return existing_edges
            # upsert existing vertices
            return await self.aadd_graph_edges(unique_name, new_edges)
            # async with self._driver.session() as session:
            #     for edge in edges:
            #         relation_type = edge.relation_type.value
            #         query = f"""
            #     // 首先尝试找到具有相同uid的现有关系
            #     OPTIONAL MATCH ()-[existing_r:`{relation_type}`]->()
            #     WHERE existing_r.uid = $uid

            #     // 获取源节点和目标节点
            #     MATCH (a:{unique_name} {{uid: $source}}), (b:{unique_name} {{uid: $target}})

            #     // 如果找到现有关系，更新它
            #     FOREACH (r IN CASE WHEN existing_r IS NOT NULL THEN [existing_r] ELSE [] END |
            #         SET r.weight = $weight,
            #             r.source_content = $source_content,
            #             r.target_content = $target_content,
            #             r.relation = $relation,
            #             r.description = $description,
            #             r.metadata = $edge_metadata,
            #             r.label = $label_param
            #     )

            #     // 如果没有找到现有关系，创建新的
            #     FOREACH (ignore IN CASE WHEN existing_r IS NULL THEN [1] ELSE [] END |
            #         CREATE (a)-[r:`{relation_type}`]->(b)
            #         SET r.weight = $weight,
            #             r.source_content = $source_content,
            #             r.target_content = $target_content,
            #             r.relation = $relation,
            #             r.description = $description,
            #             r.metadata = $edge_metadata,
            #             r.uid = $uid,
            #             r.label = $label_param
            #     )

            #     RETURN CASE WHEN existing_r IS NOT NULL THEN 'updated' ELSE 'created' END as action
            #     """

            #         ret = await session.run(
            #             query,
            #             source=edge.source,
            #             target=edge.target,
            #             weight=edge.weight,
            #             source_content=edge.source_content,
            #             target_content=edge.target_content,
            #             relation=edge.relation,
            #             description=edge.description,
            #             edge_metadata=edge.edge_metadata,
            #             uid=edge.uid,
            #             label_param=edge.label,
            #         )

            #     logger.info(f"Upserted {[dict(r) async for r in ret]} edges")
            #     return edges
        except Exception as e:
            logger.error(f"Error upserting edges: {e}")
            raise

    async def _prop_to_model(self, results: AsyncIterable, node_or_rel="n") -> List[GraphVertex] | List[GraphEdge]:
        """Convert properties dictionary to GraphVertex or GraphEdge."""
        props = [dict(record) async for record in results]
        return self._loads(
            [record[f"properties({node_or_rel})"] for record in props if f"properties({node_or_rel})" in record]
        )

    async def aselect_vertices(self, label, attrs: Dict[str, Any]) -> List[GraphVertex]:
        """Select vertices based on attributes."""
        query = f"MATCH (n:{label}) RETURN properties(n)"
        if attrs:
            conditions = self.transform_logic_operators(attrs)
            if conditions:
                query = f"MATCH (n:{label}) WHERE {conditions} RETURN properties(n)"

        if not self._driver:
            raise ValueError("Neo4j driver is not initialized. Call init() first.")

        try:
            async with self._driver.session() as session:
                logger.debug(f"Running query to select vertices: {query}")
                result = await session.run(query)
                return await self._prop_to_model(result)
        except Exception as e:
            logger.error(f"Error selecting vertices: {e}")
            raise

    async def asearch_neibors(
        self,
        vertex_id: str,
        rel_type: Optional[str] = None,
    ) -> GraphModel:
        """Search for neighbors of a given vertex in the graph."""
        if not self._driver:
            raise ValueError("Neo4j driver is not initialized. Call init() first.")

        # Build relationship filter
        rel_filter = f":`{rel_type}`" if rel_type else ""

        query = f"""
        MATCH (center {{uid: $vertex_id}})-[r{rel_filter}]-(neighbor)
        RETURN 
            properties(center) as center_props,
            properties(neighbor) as neighbor_props,
            properties(r) as edge_props,
            type(r) as rel_type,
            startNode(r) = center as is_outgoing
        """

        try:
            async with self._driver.session() as session:
                result = await session.run(query, vertex_id=vertex_id)

                vertices = {}
                edges = []

                async for record in result:
                    center_props = record["center_props"]
                    neighbor_props = record["neighbor_props"]
                    edge_props = record["edge_props"]

                    # Add vertices to collection
                    center_vertex = self._loads([center_props])[0]
                    neighbor_vertex = self._loads([neighbor_props])[0]
                    vertices[center_vertex.uid] = center_vertex
                    vertices[neighbor_vertex.uid] = neighbor_vertex

                    edge = self._loads([edge_props])[0]
                    edges.append(edge)

                return GraphModel(vertices=list(vertices.values()), edges=edges)

        except Exception as e:
            logger.error(f"Error searching neighbors: {e}")
            raise

    async def aselect_vertices_group_by_graph(
        self, label: str, attrs: Dict[str, Any]
    ) -> List[Dict[str, List[GraphVertex]]]:
        """Select vertices from the graph based on attributes and group them by a specified field."""
        if not self._driver:
            raise ValueError("Neo4j driver is not initialized. Call init() first.")

        # For this implementation, we'll group by a common field like 'namespace' or 'doc_id'
        group_by_field = attrs.pop("group_by", "namespace")  # Default group by namespace

        query = f"MATCH (n:{label})"
        if attrs:
            conditions = self.transform_logic_operators(attrs)
            if conditions:
                query += f" WHERE {conditions}"

        query += f" RETURN n.{group_by_field} as group_key, collect(properties(n)) as vertices"

        try:
            async with self._driver.session() as session:
                result = await session.run(query, **attrs)
                grouped_results = []

                async for record in result:
                    group_key = record["group_key"]
                    vertex_props_list = record["vertices"]

                    vertices = [GraphVertex(**props) for props in vertex_props_list]
                    grouped_results.append({group_key: vertices})

                return grouped_results
        except Exception as e:
            logger.error(f"Error selecting vertices grouped by graph: {e}")
            raise

    async def aselect_edges(self, label: str, attrs: Dict[str, Any]) -> List[GraphEdge]:
        """Select edges from the graph based on attributes."""
        if not self._driver:
            raise ValueError("Neo4j driver is not initialized. Call init() first.")

        query = f"MATCH (n:{label})-[r]-() "
        if attrs:
            conditions = self.transform_logic_operators(attrs, "r")
            if conditions:
                query += f"WHERE {conditions} "
        query += """
        RETURN properties(r)
        """

        try:
            async with self._driver.session() as session:
                result = await session.run(query, **attrs)
                edges = await self._prop_to_model(result, "r")
                if not edges:
                    return []

                return edges
        except Exception as e:
            logger.error(f"Error selecting edges: {e}")
            raise

    async def adelete_vertices(self, label: str, keys: List[str]):
        """Delete vertices from the graph."""
        if not keys:
            raise ValueError("Empty list of vertex keys provided")

        query = f"""
        UNWIND $keys AS key
        MATCH (n:{label} {{uid: key}})
        DETACH DELETE n
        """

        if not self._driver:
            raise ValueError("Neo4j driver is not initialized. Call init() first.")

        try:
            async with self._driver.session() as session:
                await session.run(query, keys=keys)
            logger.info(f"Deleted {len(keys)} vertices")
        except Exception as e:
            logger.error(f"Error deleting vertices: {e}")
            raise

    async def query(self, query: str, **kwargs) -> Tuple[List[GraphVertex], List[GraphEdge]]:
        """Execute a raw Cypher query."""
        if not self._driver:
            raise ValueError("Neo4j driver is not initialized. Call init() first.")

        try:
            async with self._driver.session(database=self._graph) as session:
                white_prop_list = kwargs.pop("white_prop_list", [])
                result = await session.run(query, **kwargs)
                return self._get_nodes_edges_from_queried_data(
                    [dict(record) async for record in result],
                    white_prop_list=white_prop_list,
                )
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise

    def _get_nodes_edges_from_queried_data(
        self,
        data: List[Dict[str, Any]],
        white_prop_list: List[str] = [],
    ) -> Tuple[List[GraphVertex], List[GraphEdge]]:
        """Format the query data.

        Args:
            data: The data to be formatted.
            white_prop_list: The white list of properties.

        Returns:
            Tuple[List[Vertex], List[Edge]]: The formatted vertices and edges.
        """

        # Remove id, src_id, dst_id and name from the white list
        # to avoid duplication in the initialisation of the vertex and edge
        _white_list = white_prop_list

        from neo4j import graph

        # Parse the data to nodes and relationships
        vertices = []
        edges = []
        for record in data:
            for value in record.values():
                if isinstance(value, graph.Node):
                    assert value._properties.get("uid")
                    vertices.append(dict(**value._properties))

                elif isinstance(value, graph.Relationship):
                    for node in value.nodes:  # num of nodes is 2
                        assert node and node._properties
                        vertex = dict(
                            **node._properties,
                        )
                        vertices.append(vertex)

                        assert value.nodes and value.nodes[0] and value.nodes[1]
                        edge = dict(
                            **value._properties,
                        )
                        edges.append(edge)
                elif isinstance(value, graph.Path):
                    for rel in value.relationships:
                        for node in rel.nodes:  # num of nodes is 2
                            assert node and node._properties
                            vertex = dict(
                                **node._properties,
                            )
                            vertices.append(vertex)

                            assert rel.nodes and rel.nodes[0] and rel.nodes[1]
                            edge = dict(
                                **node._properties,
                            )
                            edges.append(edge)

        return self._loads(vertices), self._loads(edges)

    async def discover_communities(
        self, label: str, resolution_parameter: float = 0.8, beta: float = 0.1, n_iterations: int = -1, **kwargs
    ) -> List[str]:
        """Run community discovery with leiden algorithm."""
        if not self._driver:
            raise ValueError("Neo4j driver is not initialized. Call init() first.")
        try:
            # Prepare leiden parameters
            leiden_config: Dict[str, Any] = {"writeProperty": "communityId", "randomSeed": 19}

            # Add optional parameters if provided
            if resolution_parameter != 0.8:
                leiden_config["gamma"] = resolution_parameter

            if beta != 0.1:
                leiden_config["theta"] = beta

            if n_iterations > 0:
                leiden_config["maxLevels"] = n_iterations

            # Add any additional parameters from kwargs
            leiden_config.update(kwargs)

            community_ids: List[str] = []

            # Execute leiden community detection
            async with self._driver.session(database=self._graph) as session:
                # Check if graph exists
                graph_name = f"{label}_community_graph"
                await session.run(
                    f"CALL gds.graph.drop('{graph_name}', false)",
                )

                # Create graph if it doesn't exist
                await session.run(
                    f"""
                    MATCH (source:{label})-[r]->(target:{label})
                    RETURN gds.graph.project(
                    $graph_name,
                      source,
                      target,
                      {{}},
                      {{ undirectedRelationshipTypes: ['*'] }}
                        )
                    """,
                    graph_name=graph_name,
                )
                logger.info(f"Graph '{graph_name}' created successfully.")
                # Run leiden algorithm
                logger.info(f"Running leiden community discovery with config: {leiden_config}")
                leiden_result = await session.run(
                    """
                    CALL gds.leiden.write($graph_name, $config)
                    YIELD communityCount, nodePropertiesWritten
                    RETURN communityCount, nodePropertiesWritten
                    """,
                    graph_name=graph_name,
                    config=leiden_config,
                )

                # Get the result
                record = await leiden_result.single()
                if record and record["communityCount"] > 0:
                    logger.info(
                        f"Found {record['communityCount']} communities, updated {record['nodePropertiesWritten']} nodes"
                    )

                    # Query community IDs directly using standard Cypher
                    community_query = f"""
                    MATCH (n:{label}) 
                    WHERE n.communityId IS NOT NULL 
                    RETURN collect(DISTINCT n.communityId) AS communityIds
                    """

                    community_result = await session.run(community_query)
                    community_record = await community_result.single()

                    if community_record:
                        community_ids = community_record["communityIds"]
                        logger.info(f"Retrieved {community_ids} unique community IDs")
                        return community_ids

                return []

        except Exception as e:
            logger.error(f"Error discovering communities: {e}")
            raise RuntimeError(f"Failed to discover communities: {str(e)}") from e

    async def get_community(self, label: str, community_id: str) -> GraphCommunity:
        """Get community."""
        query = f"MATCH (n:{label}) WHERE n.communityId = {community_id} OPTIONAL MATCH (n)-[r]-(m) RETURN n, r, m"

        vertices, edges = await self.query(query)
        s = set()
        vertices = [v for v in vertices if v.uid not in s and not s.add(v.uid)]  # type:ignore[func-returns-value]
        edges = [e for e in edges if e.uid not in s and not s.add(e.uid)]  # type:ignore[func-returns-value]
        logger.debug(f"Retrieved {len(vertices)} vertices and {len(edges)} edges for community {community_id}")

        return GraphCommunity(
            name=str(community_id), cid=str(community_id), graph=GraphModel(vertices=vertices, edges=edges)
        )

    def _build_non_continuous_query(
        self, label: str, max_hops: int, relation_path: List[str], rel_type: str, max_paths: int
    ) -> str:
        """构建非连续模式的Cypher查询"""

        # 构建关系过滤条件
        filters = ["true"]

        if relation_path:
            relation_list = "', '".join(relation_path)
            filters.append(f"rel.relation IN ['{relation_list}']")

        if rel_type:
            filters.append("rel.relation_type = $rel_type")

        where_condition = " AND ".join(filters)

        query = f"""
        MATCH (start:{label})
        WHERE start.uid IN $start_nodes
        MATCH path = (start)-[r*1..{max_hops}]->(end:{label})
        WHERE ALL(rel IN r WHERE {where_condition})
        WITH path
        LIMIT $max_paths
        UNWIND nodes(path) as node
        UNWIND relationships(path) as rel
        RETURN DISTINCT 
            properties(node) as vertex_props,
            properties(rel) as edge_props
        """

        return query

    def _build_continuous_query(
        self, label: str, max_hops: int, relation_path: List[str], rel_type: str, max_paths: int
    ) -> str:
        """构建连续模式的Cypher查询"""

        # 构建路径模式，每一跳使用指定的relation
        path_pattern = "start"
        where_conditions = []

        for i, relation in enumerate(relation_path[:max_hops]):
            path_pattern += f"-[r{i}]->(n{i}:{label})"
            where_conditions.append(f"r{i}.relation = '{relation}'")
            if rel_type:
                where_conditions.append(f"r{i}.relation_type = $rel_type")

        # 如果relation_path长度不足max_hops，剩余的跳用通用模式
        for i in range(len(relation_path), max_hops):
            path_pattern += f"-[r{i}]->(n{i}:{label})"
            if rel_type:
                where_conditions.append(f"r{i}.relation_type = $rel_type")

        where_clause = " AND ".join(where_conditions) if where_conditions else ""

        query = f"""
        MATCH (start:{label})
        WHERE start.uid IN $start_nodes
        MATCH path = (start){path_pattern.replace("start", "")}
        {f"WHERE {where_clause}" if where_clause else ""}
        WITH path
        LIMIT $max_paths
        UNWIND nodes(path) as node
        UNWIND relationships(path) as rel
        RETURN DISTINCT 
            properties(node) as vertex_props,
            properties(rel) as edge_props
        """

        return query

    async def multi_hop_search(
        self,
        label: str,
        start_nodes_id: List[str],
        relation_path: List[str] = [],
        max_hops: int = 3,
        rel_type: str = "",
        max_paths: int = 10,
        continuous: bool = False,
    ) -> GraphModel | None:
        """
        Perform a multi-hop search in the graph using Neo4j.

        Args:
            label: The label of nodes to search for
            start_nodes_id: List of starting node IDs
            relation_path: List of relations to follow
            max_hops: Maximum number of hops
            rel_type: Relation type filter
            max_paths: Maximum number of paths to return
            continuous: If True, relation_path corresponds to each hop's relation value

        Returns:
            GraphModel containing vertices and edges found in the search
        """
        if not self._driver:
            raise ValueError("Neo4j driver is not initialized. Call init() first.")

        async def run_query(tx, query, parameters):
            logger.debug(f"Running multi-hop search query: {query} with parameters: {parameters}")
            result = await tx.run(query, parameters)
            return [record async for record in result]

        # 构建Cypher查询
        if continuous and relation_path:
            # continuous模式：relation_path对应每一跳的relation取值
            query = self._build_continuous_query(label, max_hops, relation_path, rel_type, max_paths)
        else:
            # 非continuous模式：relation从relation_path中选择
            query = self._build_non_continuous_query(label, max_hops, relation_path, rel_type, max_paths)

        parameters = {"start_nodes": start_nodes_id, "max_hops": max_hops, "max_paths": max_paths}

        # 如果有rel_type参数，添加到查询参数中
        if rel_type:
            parameters["rel_type"] = rel_type

        # 执行查询
        async with self._driver.session() as session:
            records = await session.execute_read(run_query, query, parameters)

        if not records:
            return None
        # properties(node) as vertex_props,
        # properties(rel) as edge_props,
        # 解析结果并构建GraphModel
        vertices = self._loads([r["vertex_props"] for r in records])
        edges = self._loads([r["edge_props"] for r in records])
        return GraphModel(vertices=vertices, edges=edges)

    async def _query_node_ids(self, label: str, names: List[str], weights: List[float]) -> List[List[Any]]:
        """Query node IDs for PageRank calculation."""
        query_get_ids = f"""
            UNWIND $pairs AS pair
            MATCH (n:{label} {{uid: pair.name}})
            RETURN collect([elementId(n), pair.weight]) AS sourceNodes
            """
        if not self._driver:
            raise ValueError("Neo4j driver is not initialized. Call init() first.")

        async with self._driver.session() as session:
            result = await session.run(
                query_get_ids,
                pairs=[{"name": name, "weight": weight} for name, weight in zip(names, weights)],
            )
            record = await result.single()
            if not record or not record["sourceNodes"]:
                logger.warning("No source nodes found for the given names.")
                return []
            source_nodes = record["sourceNodes"]
            logger.debug(f"Source nodes: {source_nodes}")
            return source_nodes

    def save(self, path: str) -> None:
        """Neo4j data is automatically persisted in database"""
        logger.info("Neo4j graph data is automatically persisted in database")

    def _vertex_to_doc(self, vertex: Dict[str, Any]) -> Document:
        """
        Convert a vertex dictionary to a Document object.

        Args:
            vertex: Dictionary representing vertex properties

        Returns:
            Document object
        """
        return Document(
            content=vertex.get("content", ""),
            metadata={
                "namespace": vertex.get("namespace"),
                "openie_idx": vertex.get("openie_idx"),
                "entities": vertex.get("entities"),
                "facts": vertex.get("facts"),
                "name": vertex.get("name"),
            },
            uid=vertex.get("uid", ""),
            embedding=vertex.get("embedding"),
        )

    async def freestyle_search(self, label: str, entities: List[str], max_hops: int = 3) -> GraphModel | None:
        """Perform a freestyle search in the graph."""
        if not self._driver:
            raise ValueError("Neo4j driver is not initialized. Call init() first.")

        try:
            async with self._driver.session() as session:
                # Search for vertices that contain any of the entities
                query = f"""
                UNWIND $entities AS entity
                MATCH (n:{label})
                WHERE n.content CONTAINS entity OR n.name CONTAINS entity
                WITH DISTINCT n
                MATCH path = (n)-[*1..{max_hops}]-(connected)
                WITH DISTINCT nodes(path) as path_nodes, relationships(path) as path_rels
                UNWIND path_nodes AS node
                WITH DISTINCT node, path_rels
                UNWIND path_rels AS rel
                RETURN 
                    DISTINCT properties(node) as vertex_props,
                    properties(rel) as edge_props,
                """

                result = await session.run(query, entities=entities)

                vertices = []
                edges = []

                async for record in result:
                    vertex_props = record["vertex_props"]
                    edge_props = record["edge_props"]

                    if vertex_props:
                        vertex = GraphVertex(**self._loads([vertex_props])[0])
                        vertices.append(vertex)

                    if edge_props:
                        edge = GraphEdge(
                            **self._loads([edge_props])[0],
                        )
                        edges.append(edge)

                if vertices or edges:
                    return GraphModel(vertices=vertices, edges=edges)
                return None

        except Exception as e:
            logger.error(f"Error in multi-hop search: {e}")
            raise
