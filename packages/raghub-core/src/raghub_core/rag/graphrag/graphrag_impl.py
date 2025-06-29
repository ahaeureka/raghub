"""
GraphRAG implementation for RAG (Retrieval-Augmented Generation) using a graph database.

# This implementation is inspired by the DBGPT graphrag module.
# Original source code: https://github.com/eosphoros-ai/DB-GPT/blob/main/packages/dbgpt-ext/src/dbgpt_ext/storage/knowledge_graph/knowledge_graph.py
"""

import asyncio
import json
from typing import Any, AsyncIterator, Dict, List, Optional

from loguru import logger
from raghub_core.chat.base_chat import BaseChat
from raghub_core.operators.graph.query_indent_det import QueryIndentDetectionOperator
from raghub_core.prompts.graph_qa import DefaultGraphRAGQAPromptBuilder, GraphRAGQAPromptBuilder
from raghub_core.rag.base_rag import BaseGraphRAGDAO, BaseRAG
from raghub_core.rag.graphrag.operators import GraphRAGOperators
from raghub_core.schemas.chat_response import ChatResponse, QAChatResponse
from raghub_core.schemas.document import Document
from raghub_core.schemas.graph_extract_model import GraphExtractOperatorOutputModel
from raghub_core.schemas.graph_model import (
    GraphCommunity,
    GraphEdge,
    GraphModel,
    GraphRAGRetrieveResultItem,
    GraphVertex,
    Namespace,
    QueryIndentationModel,
    RelationType,
)
from raghub_core.schemas.keywords_model import KeywordsOperatorOutputModel
from raghub_core.schemas.rag_model import RetrieveResultItem
from raghub_core.schemas.summarize_model import SummarizeOperatorOutputModel
from raghub_core.utils.graph.graph_helper import GraphHelper
from raghub_core.utils.misc import compute_mdhash_id, detect_language, duplicate_filter


class GraphRAGImpl(BaseRAG):
    """
    GraphRAG implementation.
    Reference:
    """

    def __init__(
        self,
        llm: BaseChat,
        dao: BaseGraphRAGDAO,
        operators: GraphRAGOperators,
        qa_prompt_builder: GraphRAGQAPromptBuilder = DefaultGraphRAGQAPromptBuilder(),
    ):
        self.llm = llm
        self.storage = dao
        self._topk = 5
        self._score_threshold = 0.5
        self._max_chunks_once_load = 10
        self._max_threads = 4
        self._operators = operators
        self._context_history_index = "{}_context_history"
        self._entities_index = "{}_entities"
        self._doc_index = "{}_docs"
        self._communities_index = "{}_communities"
        self.lock = asyncio.Lock()
        self.qa_prompt_builder = qa_prompt_builder

    def init(self):
        pass

    async def create(self, index_name):
        pass

    async def delete(self, unique_name: str, doc_ids: List[str] | str) -> None:
        await self.storage.delete(
            unique_name,
            [doc_ids] if isinstance(doc_ids, str) else doc_ids,
        )

    def _to_virtices(
        self, index_name: str, entities: List[Document], doc: Document, entities_facts: Dict[str, List[str]]
    ) -> List[GraphVertex]:
        vertices = []
        for entity in entities:
            metadata = entity.metadata
            metadata["facts"] = entities_facts.get(entity.content, [])
            vertices.append(
                GraphVertex(
                    uid=entity.uid,
                    name=entity.uid,
                    content=entity.content,
                    description={doc.uid: entity.summary},
                    metadata=entity.metadata,
                    namespace=Namespace.ENTITY.value,
                    embedding=entity.embedding,
                    label=index_name,
                    doc_id=[metadata.get("doc_id")] if "doc_id" in metadata else [],
                )
            )
        vertices.append(
            GraphVertex(
                uid=doc.uid,
                name=doc.uid,
                content=doc.content,
                description={},
                metadata=doc.metadata,
                namespace=Namespace.DOC.value,
                label=index_name,
                embedding=doc.embedding,
                doc_id=[],
            )
        )
        return vertices

    def _to_edges(self, index_name: str, result: GraphExtractOperatorOutputModel, doc_id: str) -> List[GraphEdge]:
        edges: List[GraphEdge] = []
        # for index, output in enumerate(results):
        edges.extend(
            [
                GraphEdge(
                    uid=GraphHelper.generate_edge_id(index_name, edge[0], edge[1], edge[2]),
                    source=GraphHelper.generate_vertex_id(index_name, edge[0]),
                    target=GraphHelper.generate_vertex_id(index_name, edge[2]),
                    source_content=edge[0],
                    target_content=edge[2],
                    weight=1.0,
                    relation_type=RelationType.RELATION,
                    relation=edge[1],
                    description={doc_id: edge[3]},
                    label=index_name,
                    edge_metadata={
                        "doc_id": doc_id,
                        "relation_type": edge[1],
                        "summary": [edge[3]],
                    },
                )
                for edge in result.triples
            ]
        )

        return edges

    def _build_entity_chunk_edge(self, index_name: str, entities: List[Document], doc: Document) -> List[GraphEdge]:
        edges: List[GraphEdge] = []
        for entity in entities:
            doc_id: str = entity.metadata["doc_id"]
            doc_content = doc.content
            edges.append(
                GraphEdge(
                    uid=GraphHelper.generate_edge_id(index_name, entity.uid, RelationType.INCLUDE.value, doc_id),
                    source=doc_id,
                    target=entity.uid,
                    source_content=doc_content,
                    target_content=entity.content,
                    label=index_name,
                    edge_metadata={},
                    weight=1.0,
                    relation_type=RelationType.INCLUDE,
                    description={doc.uid: f"Entity {entity.content} is included in document {doc_content}."},
                )
            )

        return edges

    def _entities_bind_to_docs(self, entities: List[Document], doc: Document) -> List[Document]:
        """
        Bind entities to documents.
        Args:
            index_name: Name of the index.
            entities: List of Document objects representing entities.
            doc: Document object representing the main document.
        Returns:
            List of Document objects with updated metadata.
        """
        entity_ids = [entity.uid for entity in entities]
        doc.metadata["entities"] = entity_ids
        return doc

    async def _add_document(
        self, index_name: str, document: Document, ge: GraphExtractOperatorOutputModel, lang="en"
    ) -> Document:
        entities: List[Document] = []
        edges: List[GraphEdge] = []
        # docs_key = list(text_context_map.keys())
        entities_facts: Dict[str, List[str]] = {}
        document.uid = document.uid or compute_mdhash_id(index_name, document.content, Namespace.DOC.value)
        document.metadata = document.metadata or {}
        entities.extend(
            [
                Document(
                    uid=GraphHelper.generate_vertex_id(index_name, entity[0]),
                    content=entity[0],
                    summary=entity[1],
                    metadata={"doc_id": document.uid, "namespace": "entity"},
                )
                for entity in ge.entities
            ]
        )

        edges = self._to_edges(index_name, ge, document.uid)
        for edge in edges:
            if edge.source not in entities_facts:
                entities_facts[edge.source] = []
            entities_facts[edge.source].append(
                "#".join([edge.source, edge.relation_type, edge.target, edge.description[document.uid]])
            )
            if edge.target not in entities_facts:
                entities_facts[edge.target] = []
            entities_facts[edge.target].append(
                "#".join([edge.source, edge.relation_type, edge.target, edge.description[document.uid]])
            )
        # logger.debug(f"Entities extracted: {entities}")
        entities = await self.storage.add_documents(self._entities_index.format(index_name), entities)
        logger.debug(f"Entities added: {[entity.content for entity in entities]}")
        storage_tasks = []
        document = self._entities_bind_to_docs(entities, document)
        storage_tasks.append(self.storage.add_documents(self._doc_index.format(index_name), [document]))
        storage_tasks.append(
            self.storage.add_virtices(index_name, self._to_virtices(index_name, entities, document, entities_facts))
        )
        storage_result = await asyncio.gather(*storage_tasks, return_exceptions=False)
        for r in storage_result:
            if isinstance(r, Exception):
                raise RuntimeError(f"Add documents error:{str(r)}") from r
        edges.extend(self._build_entity_chunk_edge(index_name, entities, document))
        await self.storage.add_edges(index_name, edges)
        communities = await self.summary_communities(index_name, lang)
        if communities:
            await self._save_communities(self._communities_index.format(index_name), communities, document.uid)
        return document

    async def add_documents(self, index_name, documents: List[Document], lang="en"):
        """
        Add documents to the graph.
        Args:
            documents: List of Document objects to add.
            lang: Language of the documents.
        """
        for i, document in enumerate(documents):
            if not document.uid:
                document.uid = compute_mdhash_id(index_name, document.content, Namespace.DOC.value)
        text_context_map = await self.aload_chunk_context(index_name, documents)
        tasks = []
        for key, context in text_context_map.items():
            tasks.append(
                self._operators.extract_graph(
                    {"histories": context, "text": key, "index_name": self._context_history_index.format(index_name)},
                    lang,
                )
            )
        # openie_infos: List[OpenIEInfo] = []
        results: List[GraphExtractOperatorOutputModel] = await asyncio.gather(*tasks, return_exceptions=False)
        tasks = []
        for i, document in enumerate(documents):
            tasks.append(self._add_document(index_name, document, results[i], lang))
        documents = await asyncio.gather(*tasks, return_exceptions=False)
        for i, document in enumerate(documents):
            if isinstance(document, Exception):
                logger.error(f"Error in adding document {documents[i]}: {document}")
                raise RuntimeError(f"Error in adding document {documents[i]}: {document}") from document
            logger.info(f"Document {document.uid} added successfully.")
        return documents

    async def _save_communities(self, index: str, communities: List[GraphCommunity], doc_uid) -> List[Document]:
        docs = [
            Document(
                uid=compute_mdhash_id(index, c.summary, Namespace.COMMUNITY.value),
                content=c.summary,
                summary=c.summary,
                metadata={
                    "total": len(communities),
                    "vertices": list(set(vs.name for vs in c.graph.vertices)),
                    "doc_id": [doc_uid],
                },
            )
            for c in communities
        ]
        return await self.storage.add_documents(index, docs)

    async def summary_communities(self, label: str, lang="zh") -> Optional[GraphCommunity]:
        """Summarize single community."""
        async with self.lock:
            community_ids = await self.storage.discover_communities(label)
            logger.debug(f"Found {community_ids} communities to summarize in label '{label}'.")
            tasks = []
            communities: List[GraphCommunity] = []
            for cid in community_ids:
                community: GraphCommunity = await self.storage.get_community(label, cid)
                if not community:
                    logger.warning(f"Community with ID {cid} not found in label '{label}'.")
                    continue
                communities.append(community)
                graph = GraphHelper.format_community(community)
                tasks.append(self._operators.summarize_communities({"graph": graph}, lang))
            results: List[SummarizeOperatorOutputModel] = await asyncio.gather(*tasks, return_exceptions=False)
            for i, community in enumerate(communities):
                communities[i].summary = results[i].summary
            return communities

    async def aload_chunk_context(self, index_name: str, texts: List[Document]) -> Dict[str, str]:
        """Load chunk context."""
        text_context_map: Dict[str, str] = {}
        tasks = []
        for text in texts:
            # Load similar chunks
            tasks.append(
                self.storage.similar_search_with_scores(
                    self._context_history_index.format(index_name), text.content, self._topk
                )
            )
        histroies = []
        results = await asyncio.gather(*tasks)
        for text, chunks in zip(texts, results):
            # Filter chunks based on score threshold
            chunks = [(chunk, score) for chunk, score in chunks if score >= self._score_threshold]
            # Sort chunks by score
            chunks.sort(key=lambda x: x[1], reverse=True)
            history = [f"Section {i + 1}:\n{chunk[0].content}" for i, chunk in enumerate(chunks)]
            context = "\n".join(history) if history else ""
            text_context_map[text.content] = context
            histroies.append(
                Document(
                    uid=compute_mdhash_id(index_name, text.content, Namespace.CONTEXT.value),
                    content=text.content,
                    metadata={"relevant_cnt": len(history), "doc_id": [text.uid]},
                )
            )
        await self.storage.add_documents(self._context_history_index.format(index_name), histroies)
        return text_context_map

    async def _retrieve_query(self, unique_name: str, query: str, top_k: int = 5) -> GraphRAGRetrieveResultItem:
        """
        Retrieve documents from the graph storage system.
        Args:
            unique_name (str): Name of the index to search in.
            query (str): Query string for the search.
            top_k (int): Number of top results to return.
        Returns:
            List[Document]: List of retrieved documents.
        """
        lang = detect_language(query)
        if lang not in ["zh", "en"]:
            lang = "en"
        tasks = []

        tasks.append(self.storage.search_communities(self._communities_index.format(unique_name), query, top_k))
        tasks.append(self._operators.extract_keywords({"text": query}, lang))
        tasks.append(self._operators.detect_query_indent({"text": query}, lang))
        results: List[
            GraphCommunity | KeywordsOperatorOutputModel | QueryIndentDetectionOperator
        ] = await asyncio.gather(*tasks, return_exceptions=False)
        communities: List[GraphCommunity] = results[0]
        keywords: KeywordsOperatorOutputModel = results[1]
        query_indent: QueryIndentationModel = results[2]
        subgraph = ""
        docs: List[Document] = []
        entities: List[str] = [entity for entity in query_indent.entities]
        entities.extend(keywords.keywords)
        entities = list(set(entities))  # Remove duplicates
        s = set()
        communities = [c for c in communities if c.summary not in s and not s.add(c.summary)]  # type:ignore[func-returns-value]
        summaries = [f"Section {i + 1}:\n{community.summary}" for i, community in enumerate(communities)]
        context = "\n".join(summaries) if summaries else ""
        graph: GraphModel | None = None
        if query_indent.entities:
            logger.debug(f"Query indent detected: {query_indent.category} with {query_indent.entities}")
            graph = await self.storage.search_graph_by_indent(unique_name, query_indent)
            if graph and graph.vertices:
                docs = [
                    Document(content=doc.content, summary=doc.description[doc.uid], uid=doc.uid, metadata=doc.metadata)
                    for doc in graph.vertices
                    if doc.namespace == Namespace.DOC.value
                ]
                graph.vertices = [doc for doc in graph.vertices if doc.namespace == Namespace.ENTITY.value]
                # subgraph = GraphHelper.format_graph(graph)
        if not graph or not graph.vertices:
            graph = await self._search_subgraph(unique_name, keywords, query_indent, top_k)
            if graph and graph.vertices:
                docs = [
                    Document(content=doc.content, summary=doc.description[doc.uid], uid=doc.uid, metadata=doc.metadata)
                    for doc in graph.vertices
                    if doc.namespace == Namespace.DOC.value
                ]

                graph.vertices = [doc for doc in graph.vertices if doc.namespace == Namespace.ENTITY.value]
                # subgraph = GraphHelper.format_graph(graph)
        if not docs:
            if not graph or not graph.vertices:
                docs = await self.storage.get_docs_by_entities(unique_name, entities)
            else:
                docs = await self.storage.get_docs_by_entities(unique_name, [v.content for v in graph.vertices])

            logger.debug(f"Retrieved {len(docs)} documents by entities: {entities}")
        subgraph = ""
        if graph and graph.vertices:
            graph.vertices = duplicate_filter(graph.vertices)
            graph.edges = duplicate_filter(graph.edges)
            subgraph = GraphHelper.format_graph(graph) if graph and graph.vertices else ""
        return GraphRAGRetrieveResultItem(
            query=query, context=context, graph=graph, subgraph=subgraph, docs=duplicate_filter(docs)
        )

    # async def _match_entities_to_docs
    async def _search_subgraph(
        self, unique_name, keywords: KeywordsOperatorOutputModel, query_indent: QueryIndentationModel, top_k=5
    ) -> GraphModel | None:
        entities = [
            GraphHelper.generate_vertex_id(
                unique_name,
                entity,
            )
            for entity in query_indent.entities
        ] or []
        if keywords.keywords:
            similar_entities = await self._search_similar_entities(unique_name, keywords.keywords, top_k)
            entities.extend(
                [
                    GraphHelper.generate_vertex_id(unique_name, entity.content)
                    for entity in similar_entities
                    if entity.content
                ]
            )
        if not entities:
            return None
        graph = await self.storage.explore_trigraph(unique_name, list(set(entities)))
        return graph

    async def _search_similar_entities(
        self, index_name: str, keywords: List[str], top_k: int = 5, score_threshold: Optional[float] = 0.8
    ) -> List[Document]:
        similar_entities: List[Document] = []
        query_tasks = []
        for keyword in keywords:
            query_tasks.append(
                self.storage.similar_search_with_scores(self._entities_index.format(index_name), keyword, top_k)
            )
        results = await asyncio.gather(*query_tasks, return_exceptions=False)
        doc_id_to_score: Dict[str, float] = {}
        for _, result in enumerate(results):
            if score_threshold:
                result = [doc for doc in result if doc[1] >= score_threshold]
            if result:
                logger.debug(f"Found {result} similar entities for keyword: {keywords}")
                similar_entities.extend([doc[0] for doc in result])
                doc_id_to_score.update({doc[0].uid: doc[1] for doc in result})
        if not similar_entities:
            return []

        return similar_entities

    async def qa(
        self,
        unique_name: str,
        query: str,
        top_k: int = 5,
        prompt: Optional[str] = None,
        llm: Optional[BaseChat] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[QAChatResponse]:
        result = await self._retrieve_query(unique_name, query, top_k)
        lang = detect_language(query)
        if lang not in ["zh", "en"]:
            lang = "en"
        qa_prompt = self.qa_prompt_builder.build(
            question=query,
            context=result.context,
            knowledge_graph=result.subgraph,
            knowledge_graph_for_doc="\n".join([doc.content for doc in result.docs]),
        )
        from langchain.prompts import ChatPromptTemplate

        shuold_container_vars = ["question", "context", "knowledge_graph", "knowledge_graph_for_doc"]
        embbedding_retrieve_result = await self.embbedding_retrieve(unique_name, [query], filter=filter)
        embbedding_retrieve_result_items = embbedding_retrieve_result.get(query, [])
        if embbedding_retrieve_result_items:
            result.docs.extend(
                [
                    doc.document
                    for doc in embbedding_retrieve_result_items
                    if doc.document not in result.docs and doc.document.content
                ]
            )
        if prompt:
            cpt = ChatPromptTemplate.from_template(prompt)
            input_vars = cpt.input_variables
            missing = set(shuold_container_vars) - set(input_vars)
            if missing:
                raise ValueError(
                    f"Prompt is missing required variables: {missing}. Required variables are: {shuold_container_vars}"
                )
            qa_prompt = cpt.invoke(
                dict(
                    question=query,
                    context=result.context,
                    knowledge_graph=result.subgraph,
                    knowledge_graph_for_doc="\n".join([doc.content for doc in result.docs]),
                )
            ).to_string()

        logger.debug(f"Prompt for query '{query}': {qa_prompt}")
        llm = llm or self.llm

        async for answer in llm.astream(qa_prompt=ChatPromptTemplate.from_template(prompt), input={}):
            ans: ChatResponse = answer
            context = {
                "context": result.context,
                "knowledge_graph": result.graph.model_dump(),
                "knowledge_graph_for_doc": result.subgraph,
            }
            yield QAChatResponse(
                question=query,
                answer=ans.content,
                tokens=ans.tokens,
                context=json.dumps(context, ensure_ascii=False),
            )

    async def retrieve(
        self, unique_name: str, queries: List[str], retrieve_top_k: int = 10
    ) -> Dict[str, RetrieveResultItem]:
        """ """

        tasks = []
        for query in queries:
            tasks.append(self._retrieve_query(unique_name, query, retrieve_top_k))
        results: List[GraphRAGRetrieveResultItem] = await asyncio.gather(*tasks)
        retrieve_results: Dict[str, List[RetrieveResultItem]] = {}
        for i, query in enumerate(queries):
            retrieve_results[query] = [
                RetrieveResultItem(doc=doc, query=query, metadata=results[i].metadata) for doc in results[i].docs
            ]
        return retrieve_results

    async def embbedding_retrieve(
        self,
        unique_name: str,
        queries: List[str],
        filter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[RetrieveResultItem]]:
        """
        Retrieve documents using embedding-based search.
        Args:
            unique_name (str): Name of the index to use for retrieval.
            queries (List[str]): List of query strings.
        Returns:
            Dict[str, List[RetrieveResultItem]]: Dictionary with queries as keys and lists of Document as values.
        """
        embedding_retrieval_results: Dict[str, List[RetrieveResultItem]] = {}
        if len(queries) == 1:
            embedding_retrieval_result = await self.storage.similar_search_with_scores(
                (self._doc_index.format(unique_name), queries[0], 100, filter)
            )
            embedding_retrieval_results[queries[0]] = [
                RetrieveResultItem(document=doc, score=score, query=queries[0])
                for doc, score in embedding_retrieval_result
            ]
        else:
            tasks = [
                self.storage.similar_search_with_scores((self._doc_index.format(unique_name), query, 100, filter))
                for query in queries
            ]
            results = await asyncio.gather(*tasks, return_exceptions=False)
            for i, query in enumerate(queries):
                embedding_retrieval_results[query] = [
                    RetrieveResultItem(document=doc, score=score, query=query) for doc, score in results[i]
                ]
        return embedding_retrieval_results
