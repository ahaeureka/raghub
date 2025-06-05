import asyncio
from typing import Any, Dict, List, Optional

from loguru import logger
from raghub_core.schemas.document import Document
from raghub_core.schemas.graph_model import GraphEdge, GraphVertex
from raghub_core.storage.graph import GraphStorage


class Neo4jGraphStorage(GraphStorage):
    name = "neo4j"

    def __init__(self, url: str, username: Optional[str] = None, password: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self._uri = url
        self._user = username
        self._password = password
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
        """
        Add new edges to the graph.

        Args:
            node_to_node_stats: Dictionary mapping (source, target) tuples to weights
        """
        if not self._driver:
            raise ValueError("Neo4j driver is not initialized. Call init() first.")
        try:
            async with self._driver.session() as session:
                tasks = []
                for edge in edges:
                    relation_type = edge.relation_type.value
                    query = f"""
                    MATCH (a:{label} {{name: $source}}), (b:{label} {{name: $target}})
                    CREATE (a)-[r:`{relation_type}` {{
                        weight: $weight,
                        source_content: $source_content,
                        target_content: $target_content,
                        relation: $relation,
                        description: $description,
                        metadata: $edge_metadata,
                        uid: $uid,
                        label: $label
                    }}]->(b)
                    RETURN r
                    """

                    tasks.append(
                        session.run(
                            query,
                            source=edge.source,
                            target=edge.target,
                            weight=edge.weight,
                            source_content=edge.source_content,
                            target_content=edge.target_content,
                            relation=edge.relation,
                            description=edge.description,
                            metadata=edge.edge_metadata,
                            uid=edge.uid,
                            label=edge.label,
                        )
                    )
                await asyncio.gather(*tasks)
                logger.info(f"Successfully added {len(edges)} edges to the graph")

        except Exception as e:
            logger.error(f"Error adding edges: {e}")
            raise

    async def aadd_vertices(self, label: str, nodes: List[GraphVertex]):
        """
        Add vertices to the graph.

        Args:
            label: Label for the vertices (e.g., "Hipporag")
            nodes: List of dictionaries representing vertex properties
        """
        if not self._driver:
            raise ValueError("Neo4j driver is not initialized. Call init() first.")
        if not nodes:
            raise ValueError("Empty list of vertices provided")
        # attrs = [f"n.{k}=" for k in list(nodes[0].model_dump().keys())]
        query = f"""
    UNWIND $nodes AS node
    MERGE (n:{label} {{name: node.uid}})
    ON CREATE SET
        n.name=node.uid, 
        n.content=node.content, 
        n.namespace=node.namespace,
        n.openie_idx=node.metadata.openie_idx,
        n.entities=node.metadata.entities,
        n.facts=node.metadata.facts,
        n.uid=node.uid, 
        n.doc_id=node.doc_id,
        n.description=node.description,
        n.label=node.label,
        n.index=node.index,
        n.embedding=node.embedding
    """

        try:
            async with self._driver.session() as session:
                await session.run(query, nodes=[n.model_dump() for n in nodes])
            logger.info(f"Added {len(nodes)} vertices")
        except Exception as e:
            logger.error(f"Error adding vertices: {e}")
            raise

    async def aadd_graph_vertices(self, label: str, nodes: List[GraphVertex]):
        pass

    async def aselect_vertices(self, label, attrs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Select vertices based on attributes.

        Args:
            attrs: Key-value pairs of vertex attributes to match
        """
        query = f"MATCH (n:{label}) RETURN properties(n)"
        if attrs:
            conditions = self.transform_logic_operators(attrs)
            if conditions:
                query = f"MATCH (n:{label}) WHERE {conditions} RETURN properties(n)"
        if not self._driver:
            raise ValueError("Neo4j driver is not initialized. Call init() first.")
        try:
            async with self._driver.session() as session:
                result = []
                if not attrs:
                    result = await session.run(query)
                else:
                    result = await session.run(query, **attrs)
                return [record["properties(n)"] for record in result]
        except Exception as e:
            logger.error(f"Error selecting vertices: {e}")
            raise

    async def adelete_vertices(self, label: str, keys: List[str]) -> None:
        """
        Delete vertices with specified names.

        Args:
            keys: List of vertex names to delete
        """
        if not keys:
            raise ValueError("Empty list of vertex names provided")

        query = f"""
        UNWIND $names AS name
        MATCH (n:{label} {{name: name}})
        DETACH DELETE n
        """
        if not self._driver:
            raise ValueError("Neo4j driver is not initialized. Call init() first.")
        try:
            async with self._driver.session() as session:
                await session.run(query, names=keys)
            logger.info(f"Deleted {len(keys)} vertices")
        except Exception as e:
            logger.error(f"Error deleting vertices: {e}")
            raise

    async def _query_node_ids(self, label: str, names: List[str], weights: List[float]) -> List[List[Any]]:
        query_get_ids = f"""
            UNWIND $pairs AS pair
            MATCH (n:{label} {{name: pair.name}})
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

    async def apersonalized_pagerank(
        self,
        label: str,
        vertices_with_weight: Dict[str, float],
        damping: float = 0.5,
        top_k: int = 20,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """
        Calculate personalized PageRank scores for top-k vertices.
        Returns:
            Dictionary of {vertex_name: score}
        """
        if not self._driver:
            raise ValueError("Neo4j driver is not initialized. Call init() first.")

        # 获取图名称、节点标签和关系类型（支持自定义）
        # graph_name = kwargs.get("graph_name", "hipporag")
        # node_label = kwargs.get("node_label", "Hipporag")  # 默认节点标签
        # relationship_type = kwargs.get("relationship_type", "RELATIONSHIP")  # 默认关系类型

        try:
            # 检查图是否已存在
            async with self._driver.session() as session:
                result = await session.run(
                    "CALL gds.graph.list() YIELD graphName RETURN graphName",
                )
                existing_graphs = [record["graphName"] for record in result]
                logger.debug(f"Existing graphs: {existing_graphs}")
                graph_name = f"{label}Graph"
                if graph_name not in existing_graphs:
                    logger.info(f"Graph '{label}' does not exist. Creating...")
                    # 创建图投影（基于节点标签和关系类型）
                    create_graph_query = f"""
                    MATCH (source:{label})-[r:RELATED]->(target:{label})
                    RETURN gds.graph.project('{graph_name}', 
                    source,
                    target,
                    {{ relationshipProperties: r {{ .weight }} }}
                    )"""
                    ret = await session.run(
                        create_graph_query,
                    )
                    logger.info(f"Graph projection created: {[record for record in ret]}")
                    logger.info(f"Graph '{label}' created successfully.")

            query = f"""
    UNWIND $pairs AS pair
    MATCH (n:{label} {{name: pair.name}})
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
    RETURN gds.util.asNode(nodeId).name AS name, score
    ORDER BY score DESC, name ASC
    LIMIT $topK
"""
            # logger.debug(f"Executing personalized PageRank with query: {source_nodes}")
            async with self._driver.session() as session:
                result = await session.run(
                    query,
                    dict(
                        damping=damping,
                        pairs=[{"name": name, "weight": value} for name, value in vertices_with_weight.items()],
                        topK=top_k,
                        graphName=graph_name,
                    ),
                )
                return {record["name"]: record["score"] for record in result}

        except Exception as e:
            logger.error(f"Error in personalized_pagerank: {e}")
            raise

    def save(self, path: str) -> None:
        """Neo4j data is automatically persisted in database"""
        logger.info("Neo4j graph data is automatically persisted in database")

    async def vertices_count(self, label) -> int:
        """Return number of vertices in the graph"""
        if self._driver is None:
            raise ValueError("Neo4j driver is not initialized. Call init() first.")
        try:
            async with self._driver.session() as session:
                result = await session.run(f"MATCH (n:{label}) RETURN count(n)")
                return await result.single()[0]
        except Exception as e:
            logger.error(f"Error counting vertices: {e}")
            raise

    def _vertex_to_doc(self, vertex: Dict[str, Any]) -> Document:
        """
        Convert a vertex dictionary to a Document object.

        Args:
            vertex: Dictionary representing vertex properties

        Returns:
            Document object
        """
        return Document(
            content=vertex["content"],
            metadata={
                "namespace": vertex["namespace"],
                "openie_idx": vertex["openie_idx"],
                "entities": vertex["entities"],
                "facts": vertex["facts"],
                "name": vertex["name"],
            },
            uid=vertex["uid"],
            embedding=vertex["embedding"],
        )

    async def aaget_by_ids(self, ids: List[str]) -> List[Document]:
        """
        Retrieve documents by their IDs.

        Args:
            ids: List of vertex names to retrieve

        Returns:
            List of Document objects
        """
        if not self._driver:
            raise ValueError("Neo4j driver is not initialized. Call init() first.")
        try:
            async with self._driver.session() as session:
                result = await session.run(
                    """
                    UNWIND $ids AS id
                    MATCH (n {name: id})
                    RETURN {
                        content: n.content,
                        metadata: n.metadata,
                        uid: n.uid,
                        name: n.name,
                        embedding: n.embedding,
                        namespace: n.namespace,
                        openie_idx: n.openie_idx,
                        entities: n.entities,
                        facts: n.facts,
                    } AS doc
                    """,
                    ids=ids,
                )
                return [Document(self._vertex_to_doc(record["doc"])) for record in result]
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            raise
