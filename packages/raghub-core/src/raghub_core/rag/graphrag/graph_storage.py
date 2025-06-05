from typing import Dict, List, Optional, Tuple

from loguru import logger
from raghub_core.rag.base_rag import BaseGraphRAGStorage
from raghub_core.schemas.document import Document
from raghub_core.schemas.graph_model import (
    GraphCommunity,
    GraphEdge,
    GraphModel,
    GraphVertex,
    QueryIndentationModel,
    RelationType,
    SearchIndentationCategory,
)
from raghub_core.storage.graph import GraphStorage
from raghub_core.storage.structed_data import StructedDataStorage
from raghub_core.storage.vector import VectorStorage
from raghub_core.utils.graph.graph_helper import GraphHelper


class GraphRAGStorage(BaseGraphRAGStorage):
    """
    GraphRAGStorage is a concrete implementation of the BaseGraphRAGStorage class.
    It provides methods to manage graph storage in a RAG system.
    """

    def __init__(
        self,
        embedding_store: VectorStorage,
        graph_store: GraphStorage,
        db: StructedDataStorage,
    ):
        """
        Initialize the GraphRAGStorage with a specific storage backend.
        Args:
            storage (BaseGraphRAGStorage): The storage backend to use.
        """
        self.embedding_store = embedding_store
        self.graph_store = graph_store
        self.db = db
        self._entities_index = "{}_entities"
        self._doc_index = "{}_docs"

    async def create(self, index_name: str):
        """
        Create a new index in the graph storage system.
        The index name should be unique.
        Args:
            index_name (str): Name of the index to create.
        Returns:
            None
        """
        # await self.graph_store.create_new_index(index_name)
        pass

    async def add_documents(self, index_name: str, documents: List[Document]) -> List[Document]:
        """
        Add documents to the graph storage system.
        Args:
            index_name (str): Name of the index to which documents will be added.
            documents (List[Document]): List of documents to add.
        Returns:
            List[Document]: List of added documents.
        """
        # Add documents to the graph storage
        exsiting = await self.embedding_store.get_by_ids(index_name, [doc.uid for doc in documents])
        new_docs = {doc.uid: doc.model_dump() for doc in documents}
        if exsiting:
            exsiting_docs = {doc.uid: doc.model_dump() for doc in exsiting}
            for uid, _ in exsiting_docs.items():
                if uid in new_docs:
                    exsiting_docs[uid].update(new_docs[uid])
                    new_docs[uid] = exsiting_docs[uid]

        documents = [Document(**doc) for doc in new_docs.values()]

        return await self.embedding_store.add_documents(index_name, documents)

    async def add_virtices(self, unique_name: str, texts: List[GraphVertex]):
        """
        Add vertices to the graph storage system.
        Args:
            unique_name (str): Name of the index to which vertices will be added.
            texts (List[Document]): List of documents to add as vertices.
        Returns:
            List[Document]: List of added vertices.
        """
        # Add vertices to the graph storage
        existing_vertices: List[GraphVertex] = await self.graph_store.aselect_vertices(
            unique_name, dict(uid_in=[v.uid for v in texts])
        )
        filter_existing_vertices = (
            [v for v in texts if v.uid in [ev.uid for ev in existing_vertices]] if existing_vertices else []
        )
        if filter_existing_vertices:
            await self.graph_store.aupdate_vertices(unique_name, filter_existing_vertices)
        if not existing_vertices:
            logger.warning(f"No existing vertices found for index: {unique_name}")
            return await self.graph_store.aadd_graph_vertices(unique_name, texts)
        # Filter out existing vertices
        new_vertices = [v for v in texts if v.uid not in [ev.uid for ev in existing_vertices]]
        if not new_vertices:
            logger.warning(f"No new vertices to add for index: {unique_name}")
            return existing_vertices
        # upsert existing vertices
        return await self.graph_store.aadd_graph_vertices(unique_name, new_vertices)

    async def delete(self, unique_name: str, doc_ids: List[str] | str) -> None:
        """
        Delete vertices from the graph storage system.
        Args:
            unique_name (str): Name of the index from which vertices will be deleted.
            doc_ids (List[str]|str): List of vertex IDs to delete.
        Returns:
            None
        """
        pass

    async def init(self) -> None:
        """
        Initialize the graph storage system.
        Returns:
            None
        """
        await self.graph_store.init()
        self.embedding_store.init()
        await self.db.init()

    async def add_edges(self, unique_name: str, edges: List[GraphEdge]):
        """
        Add edges to the graph storage system.
        Args:
            unique_name (str): Name of the index to which edges will be added.
            edges (List[Document]): List of documents to add as edges.
        Returns:
            List[GraphEdge]: List of added edges.
        """
        existing_edges: List[GraphEdge] = await self.graph_store.aselect_edges(
            unique_name, dict(uid_in=[edge.uid for edge in edges])
        )
        filter_existing_edges = (
            [edge for edge in edges if edge.uid in [ee.uid for ee in existing_edges]] if existing_edges else []
        )
        if filter_existing_edges:
            await self.graph_store.aupdate_edges(unique_name, filter_existing_edges)
        if not existing_edges:
            logger.warning(f"No existing edges found for index: {unique_name}")
            return await self.graph_store.aadd_graph_edges(unique_name, edges)
        new_edges = [edge for edge in edges if edge.uid not in [ee.uid for ee in existing_edges]]
        if not new_edges:
            logger.warning(f"No new edges to add for index: {unique_name}")
            return existing_edges
        # Add edges to the graph storage
        return await self.graph_store.aadd_graph_edges(unique_name, new_edges)

    async def similar_search_with_scores(
        self, index_name: str, query: str, top_k: int = 10, filter: Optional[Dict[str, str]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Perform a similarity search in the graph storage system.
        Args:
            index_name (str): Name of the index to search in.
            query (str): The query string to search for.
            top_k (int): The number of top results to return.
        Returns:
            List[Document]: List of documents matching the query.
        """
        return await self.embedding_store.asimilar_search_with_scores(index_name, query, top_k, filter)

    async def discover_communities(self, label: str) -> List[str]:
        """
        Discover communities in the graph storage system.
        Args:
            None
        Returns:
            None
        """
        return await self.graph_store.discover_communities(label)

    async def get_community(self, lable, community_id: str) -> List[Document]:
        """
        Get a community from the graph storage system.
        Args:
            community_id (str): The ID of the community to retrieve.
        Returns:
            Optional[GraphVertex]: The community object, or None if not found.
        """
        return await self.graph_store.get_community(lable, community_id)

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
        results = await self.embedding_store.asimilar_search_with_scores(label, query, top_k)
        logger.debug(f"Found {[similar for _, similar in results]} communities for query: {query}")
        return [
            GraphCommunity(cid=doc.uid, summary=doc.content, name=doc.uid)
            for doc, score in results
            if score >= similar_threshold
        ]

    # async def search_graph_by_triple(self, index_name: str, triple: Triple) -> Optional[GraphModel]:
    #     start_nodes = [compute_mdhash_id(triple.subject, Namespace.ENTITY.value)]
    #     graph = await self.graph_store.multi_hop_search(
    #         index_name, Namespace.ENTITY.value, start_nodes, RelationType.INCLUDE.value
    #     )
    #     if not graph:
    #         return None
    #     for edge in graph.edges:
    #         if edge.relation and set(triple.predicate).intersection(set(edge.relation)):
    #             return graph

    async def search_graph_by_indent(self, index_name: str, indent: QueryIndentationModel) -> Optional[GraphModel]:
        """
        Search the graph by indent.

        给定一个问题，请分析并归类到以下类别之一：
        1. SingleEntitySearch：搜索给定实体的详细信息
        2. OneHopEntitySearch：给定一个实体和一个关系，搜索与该实体存在该关系的所有实体
        3. OneHopRelationSearch：给定两个实体，搜索它们之间的关系
        4. TwoHopEntitySearch：给定一个实体和一个关系，将该关系拆分为两个连续关系，搜索与给定实体存在两跳关系的所有实体
        5. FreestyleQuestion：不属于以上四类的问题。搜索所有相关实体及以其为中心的两跳子图
        同时以JSON格式返回可能用于查询生成的实体和关系。参考示例如下：
        ---------------------
        示例：
        问题：介绍TuGraph
        返回：
        {{"category": "SingleEntitySearch", "entities": ["TuGraph"], "relations": []}}
        问题：谁向TuGraph提交了代码
        返回：
        {{"category": "OneHopEntitySearch", "entities": ["TuGraph"], "relations": ["提交"]}}
        问题：Alex和TuGraph之间是什么关系
        返回：
        {{"category": "OneHopRelationSearch", "entities": ["Alex", "TuGraph"], "relations": []}}
        问题：Bob的同事是谁
        返回：
        {{"category": "TwoHopEntitySearch", "entities": ["Bob"], "relations": ["任职于"]}}
        问题：分别介绍TuGraph和DB-GPT
        返回：
        {{"category": "FreestyleQuestion", "entities": ["TuGraph", "DBGPT"], "relations": []}}
        ---------------------
        """
        results: List[GraphModel] = []
        if indent.category == SearchIndentationCategory.SINGLE_ENTITY_SEARCH:
            for entity in indent.entities:
                results.append(await self._single_entity_search(index_name, entity))
        elif indent.category == SearchIndentationCategory.ONE_HOP_ENTITY_SEARCH:
            for entity in indent.entities:
                logger.debug(
                    f"Performing one-hop entity search for entity: {entity} with relations: {indent.relations}"
                )
                results.append(await self._one_hop_entity_search(index_name, entity, indent.relations))
            # return await self._one_hop_entity_search(index_name, indent.entities[0], indent.relations)
        elif indent.category == SearchIndentationCategory.ONE_HOP_RELATION_SEARCH:
            return await self._one_hop_relation_search(index_name, indent.entities)
        elif indent.category == SearchIndentationCategory.TWO_HOP_ENTITY_SEARCH:
            for entity in indent.entities:
                logger.debug(
                    f"Performing two-hop entity search for entity: {entity} with relations: {indent.relations}"
                )
                results.append(await self._two_hop_entity_search(index_name, entity, indent.relations))
        elif indent.category == SearchIndentationCategory.FREESTYLE_QUESTION:
            return await self._freestyle_question_search(index_name, indent.entities)
        else:
            logger.warning(f"Unknown search category: {indent.category}")
            return None
        # Combine results from all entities
        combined_vertices = []
        combined_edges = []
        if not any(results):
            logger.warning(f"No results found for indent: {indent}")
            return None
        for result in results:
            if not result:
                logger.warning("Empty result found, skipping.")
                continue
            combined_vertices.extend(result.vertices)
            combined_edges.extend(result.edges)
        # Create a new GraphModel with combined results
        if not combined_vertices:
            logger.warning(f"No vertices found for indent: {indent}")
            return None
        return GraphModel(vertices=combined_vertices, edges=combined_edges)

    async def _single_entity_search(self, index_name: str, entity: str) -> GraphModel:
        """
        Perform a single entity search in the graph storage system.
        Args:
            index_name (str): Name of the index to search in.
            entity (str): The entity to search for.
        Returns:
            GraphModel: The graph model containing the entity and its related information.
        """
        entities = await self.graph_store.aselect_vertices(index_name, name_in=entity)

        if not entities:
            return GraphModel(vertices=[], edges=[])
        return GraphModel(vertices=entities, edges=[])

    async def _one_hop_entity_search(self, index_name: str, entity: str, relations: List[str]) -> GraphModel:
        """
        Perform a one-hop entity search in the graph storage system.
        Args:
            index_name (str): Name of the index to search in.
            entity (str): The entity to search for.
            relation (str): The relation to follow from the entity.
        Returns:
            GraphModel: The graph model containing the entity and its related information.
        """
        logger.debug(f"Performing one-hop entity search for entity: {entity} with relations: {relations}")
        return await self.graph_store.multi_hop_search(
            index_name,
            start_nodes_id=[GraphHelper.generate_vertex_id(entity)],
            rel_type=RelationType.RELATION.value,
            relation_path=relations,
            max_hops=1,
        )

    async def _one_hop_relation_search(self, index_name: str, entities: List[str]) -> GraphModel:
        """
        Perform a one-hop relation search in the graph storage system.
        Args:
            index_name (str): Name of the index to search in.
            entities (List[str]): The entities to search for.
        Returns:
            GraphModel: The graph model containing the entities and their relations.
        """
        source = GraphHelper.generate_vertex_id(entities[0])
        target = GraphHelper.generate_vertex_id(entities[1])
        edges = await self.graph_store.aselect_edges(
            index_name,
            dict(source=source, target=target),
        )
        source = await self.graph_store.aselect_vertices(index_name, uid_eq=source)
        target = await self.graph_store.aselect_vertices(index_name, uid_eq=target)
        return GraphModel(vertices=[GraphVertex(**source[0]), GraphVertex(**target[0])], edges=edges)

    async def _two_hop_entity_search(self, index_name: str, entity: str, relations: List[str]) -> GraphModel:
        """
        Perform a two-hop entity search in the graph storage system.
        Args:
            index_name (str): Name of the index to search in.
            entity (str): The entity to search for.
            relations (List[str]): The relations to follow from the entity.
        Returns:
            GraphModel: The graph model containing the entity and its related information.
        """
        start_nodes_id = [GraphHelper.generate_vertex_id(entity)]
        return await self.graph_store.multi_hop_search(
            index_name,
            start_nodes_id=start_nodes_id,
            rel_type=RelationType.RELATION.value,
            relation_path=relations,
            max_hops=len(relations) + 1,  # +1 for the starting entity
        )

    async def _freestyle_question_search(self, index_name: str, entities: List[str]) -> GraphModel:
        """
        Perform a freestyle question search in the graph storage system.
        Args:
            index_name (str): Name of the index to search in.
            entities (List[str]): The entities to search for.
        Returns:
            GraphModel: The graph model containing the entities and their related information.
        """
        return await self.graph_store.freestyle_search(index_name, entities, rel_type=RelationType.RELATION.value)

    async def explore_trigraph(self, index_name: str, entities: List[str]) -> GraphModel:
        # logger.debug(f"Found {similar_entities} similar entities for keywords: {keywords}")
        graph = await self.graph_store.multi_hop_search(index_name, entities, rel_type=RelationType.RELATION.value)
        if not graph:
            None
        # graph.vertices = [v for v in graph.vertices if v.namespace == Namespace.ENTITY.value]
        return graph
        # Create a GraphModel from the similar entities

    async def get_verteices_by_ids(self, index_name: str, ids: List[str]) -> List[GraphVertex]:
        """
        Get vertices by their IDs from the graph storage system.
        Args:
            index_name (str): Name of the index to search in.
            ids (List[str]): List of vertex IDs to retrieve.
        Returns:
            List[GraphVertex]: List of vertices matching the given IDs.
        """
        if not ids:
            logger.warning("No IDs provided for vertex retrieval.")
            return []
        return await self.graph_store.aselect_vertices(index_name, uid_in=ids)

    async def save_openie_info(self, unique_name, openie_info):
        pass
        # return await super().save_openie_info(unique_name, openie_info)

    async def get_docs_by_entities(self, index_name: str, entities: List[str]) -> List[Document]:
        """
        Get documents by their entities from the graph storage system.
        Args:
            index_name (str): Name of the index to search in.
            entities (List[str]): List of entity names to retrieve documents for.
        Returns:
            List[Document]: List of documents matching the given entities.
        """
        if not entities:
            logger.warning("No entities provided for document retrieval.")
            return []
        entitiy_ids = [GraphHelper.generate_vertex_id(entity) for entity in entities]
        ents = await self.embedding_store.get_by_ids(self._entities_index.format(index_name), list(set(entitiy_ids)))
        if not ents:
            logger.warning(f"No documents found for entities:{index_name}: {entities}")
            return []
        docs_ids = [ent.metadata["doc_id"] for ent in ents]
        docs = await self.embedding_store.get_by_ids(self._doc_index.format(index_name), list(set(docs_ids)))
        if not docs:
            logger.warning(f"No documents found for entity IDs: {docs_ids}")
            return []
        return docs
