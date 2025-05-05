from typing import Any, Dict, List, Optional, Tuple

from deeprag_core.schemas.document import Document
from deeprag_core.storage.graph import GraphStorage
from loguru import logger


class Neo4jGraphStorage(GraphStorage):
    name = "neo4j"

    def __init__(self, uri: str, user: str, password: str, **kwargs):
        super().__init__(**kwargs)
        self._uri = uri
        self._user = user
        self._password = password
        try:
            from neo4j import Driver
        except ImportError:
            raise ImportError("Neo4j is not installed. Please install it using `pip install neo4j`.")

        self._driver: Optional[Driver] = None

    def init(self):
        """Initialize Neo4j driver connection."""
        if not self._driver:
            try:
                from neo4j import GraphDatabase

                self._driver = GraphDatabase.driver(self._uri, auth=(self._user, self._password))
                logger.debug("Neo4j driver initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize Neo4j driver: {e}")
                raise

    def close(self):
        """Close Neo4j driver connection."""
        if self._driver:
            self._driver.close()
            logger.debug("Neo4j driver connection closed.")

    def add_new_edges(self, node_to_node_stats: Dict[Tuple[str, str], float]):
        """
        Add new edges to the graph.

        Args:
            node_to_node_stats: Dictionary mapping (source, target) tuples to weights
        """
        if not self._driver:
            raise ValueError("Neo4j driver is not initialized. Call init() first.")
        if not node_to_node_stats:
            raise ValueError("Empty dictionary of edges provided")

        query = """
        UNWIND $edges AS edge
        MATCH (a {name: edge[0]}), (b {name: edge[1]})
        CREATE (a)-[:RELATED {weight: edge[2]}]->(b)
        """

        try:
            with self._driver.session() as session:
                session.run(query, edges=[(src, tgt, weight) for (src, tgt), weight in node_to_node_stats.items()])
            logger.info(f"Added {len(node_to_node_stats)} edges")
        except Exception as e:
            logger.error(f"Error adding edges: {e}")
            raise

    def add_vertices(self, nodes: List[Dict[str, Any]]):
        """
        Add vertices to the graph.

        Args:
            nodes: List of dictionaries representing vertex properties
        """
        if not self._driver:
            raise ValueError("Neo4j driver is not initialized. Call init() first.")
        if not nodes:
            raise ValueError("Empty list of vertices provided")

        query = """
        UNWIND $nodes AS node
        CREATE (n:Node {name: node.name, content: node.content, metadata: node.metadata, 
        uid: node.uid, embedding: node.embedding})
        """

        try:
            with self._driver.session() as session:
                session.run(query, nodes=nodes)
            logger.info(f"Added {len(nodes)} vertices")
        except Exception as e:
            logger.error(f"Error adding vertices: {e}")
            raise

    def select_vertices(self, attrs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Select vertices based on attributes.

        Args:
            attrs: Key-value pairs of vertex attributes to match
        """
        query = "MATCH (n) RETURN properties(n)"
        if attrs:
            conditions = self.transform_logic_operators(attrs)
            if conditions:
                query = f"MATCH (n) WHERE {conditions} RETURN properties(n)"
        if not self._driver:
            raise ValueError("Neo4j driver is not initialized. Call init() first.")
        try:
            with self._driver.session() as session:
                result = []
                if not attrs:
                    result = session.run(query)
                else:
                    result = session.run(query, **attrs)
                return [record["properties(n)"] for record in result]
        except Exception as e:
            logger.error(f"Error selecting vertices: {e}")
            raise

    def delete_vertices(self, keys: List[str]) -> None:
        """
        Delete vertices with specified names.

        Args:
            keys: List of vertex names to delete
        """
        if not keys:
            raise ValueError("Empty list of vertex names provided")

        query = """
        UNWIND $names AS name
        MATCH (n {name: name})
        DETACH DELETE n
        """
        if not self._driver:
            raise ValueError("Neo4j driver is not initialized. Call init() first.")
        try:
            with self._driver.session() as session:
                session.run(query, names=keys)
            logger.info(f"Deleted {len(keys)} vertices")
        except Exception as e:
            logger.error(f"Error deleting vertices: {e}")
            raise

    def personalized_pagerank(
        self,
        vertices_with_weight: Dict[str, float],
        damping: float = 0.5,
        top_k: int = 10,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """
        Calculate personalized PageRank scores for top-k vertices.
        Returns:
            Dictionary of {vertex_name: score}
        """
        # Create graph projection if not exists
        if not self._driver:
            raise ValueError("Neo4j driver is not initialized. Call init() first.")
        try:
            graph_name = kwargs.get("graph_name", None)
            sourceNodes = [[key, weight] for key, weight in vertices_with_weight.items()]
            if not graph_name:
                raise ValueError("Graph name is required for graph projection")
            query = """
            UNWIND $graphName AS graphName
            UNWIND $damping AS damping
            UNWIND $sources AS sources
            UNWIND $topK AS topK
        CALL gds.pageRank.stream(graphNmae, {
          maxIterations: 20,
          dampingFactor: damping,
          sourceNodes: sources
        })
        YIELD nodeId, score
        RETURN gds.util.asNode(nodeId).name AS name, score
        ORDER BY score DESC, name ASC
        LIMIT topK
        """
            if self._driver is None:
                raise ValueError("Neo4j driver is not initialized. Call init() first.")
            with self._driver.session() as session:
                result = session.run(
                    query,
                    graphName=graph_name,
                    damping=damping,
                    sources=sourceNodes,
                    topK=top_k,
                )

                logger.info(f"Graph projection '{graph_name}' created successfully.")
                return {record["name"]: record["score"] for record in result}
        except Exception as e:
            logger.error(f"Error creating graph projection: {e}")
            raise

    def save(self, path: str) -> None:
        """Neo4j data is automatically persisted in database"""
        logger.info("Neo4j graph data is automatically persisted in database")

    def vertices_count(self) -> int:
        """Return number of vertices in the graph"""
        if self._driver is None:
            raise ValueError("Neo4j driver is not initialized. Call init() first.")
        try:
            with self._driver.session() as session:
                result = session.run("MATCH (n) RETURN count(n)")
                return result.single()[0]
        except Exception as e:
            logger.error(f"Error counting vertices: {e}")
            raise

    def get_by_ids(self, ids: List[str]) -> List[Document]:
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
            with self._driver.session() as session:
                result = session.run(
                    """
                    UNWIND $ids AS id
                    MATCH (n {name: id})
                    RETURN {
                        content: n.content,
                        metadata: n.metadata,
                        uid: n.uid,
                        name: n.name,
                        embedding: n.embedding
                    } AS doc
                    """,
                    ids=ids,
                )
                return [Document(**record["doc"]) for record in result]
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            raise
