"""
GraphRAG implementation for RAG (Retrieval-Augmented Generation) using a graph database.

# This implementation is inspired by the DBGPT graphrag module.
# Original source code: https://github.com/eosphoros-ai/DB-GPT/blob/main/packages/dbgpt-ext/src/dbgpt_ext/storage/knowledge_graph/knowledge_graph.py
"""

import asyncio
import json
import traceback
from typing import Any, AsyncIterator, Dict, List, Optional

from loguru import logger
from raghub_core.chat.base_chat import BaseChat
from raghub_core.operators.graph.query_indent_det import QueryIndentDetectionOperator
from raghub_core.prompts.graph_qa import DefaultGraphRAGQAPromptBuilder, GraphRAGQAPromptBuilder
from raghub_core.rag.base_rag import BaseGraphRAGDAO, BaseRAG
from raghub_core.rag.graphrag.operators import GraphRAGOperators
from raghub_core.rerank.base_rerank import BaseRerank
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
    GraphRAG (Graph-based Retrieval-Augmented Generation) implementation.

    This class provides a comprehensive implementation of GraphRAG that combines
    traditional dense retrieval with graph-based reasoning for enhanced information
    retrieval and question answering capabilities.

    Key Features:
    - Graph-based entity and relationship extraction
    - Community detection and summarization
    - Multi-modal retrieval (graph + embedding)
    - Concurrent processing with batch optimization
    - Advanced query intent detection

    Reference: Inspired by DB-GPT GraphRAG module
    """

    def __init__(
        self,
        llm: BaseChat,
        reranker: BaseRerank,
        dao: BaseGraphRAGDAO,
        operators: GraphRAGOperators,
        qa_prompt_builder: GraphRAGQAPromptBuilder = DefaultGraphRAGQAPromptBuilder(),
        topk: int = 5,
        score_threshold: float = 0.5,
        max_chunks_once_load: int = 10,
        max_threads: int = 4,
    ):
        """
        Initialize GraphRAG implementation with required components.

        Args:
            llm: Language model for text generation and processing
            reranker: Document reranking component for result optimization
            dao: Data access object for graph storage operations
            operators: Graph processing operators for extraction and analysis
            qa_prompt_builder: Prompt builder for QA tasks
            topk: Number of top results to retrieve (default: 5)
            score_threshold: Minimum similarity score threshold (default: 0.5)
            max_chunks_once_load: Maximum chunks to load at once (default: 10)
            max_threads: Maximum concurrent threads for processing (default: 4)
        """
        # Core components
        self.llm = llm
        self.storage = dao
        self._reranker = reranker
        self._operators = operators
        self.qa_prompt_builder = qa_prompt_builder

        # Configuration parameters
        self._topk = topk
        self._score_threshold = score_threshold
        self._max_chunks_once_load = max_chunks_once_load
        self._max_threads = max_threads

        # Index naming templates for different data types
        self._context_history_index = "{}_context_history"  # Context history storage
        self._entities_index = "{}_entities"  # Entity storage
        self._doc_index = "{}_docs"  # Document storage
        self._communities_index = "{}_communities"  # Community storage

        # Concurrency control - lazy initialization to avoid event loop issues
        self._lock = None

    @property
    def lock(self):
        """
        Get the current event loop's Lock, creating one if it doesn't exist.

        This property provides thread-safe access to an asyncio Lock for coordinating
        concurrent operations. It creates a new Lock for each event loop to avoid
        cross-event-loop issues that could cause deadlocks or race conditions.

        Returns:
            asyncio.Lock: Lock instance for the current event loop, or None if no loop is running
        """
        # Create a new Lock for each event loop to avoid cross-loop issues
        try:
            asyncio.get_running_loop()
            return asyncio.Lock()
        except RuntimeError:
            # No running event loop, return None - Lock will be created when needed
            return None

    def init(self):
        """
        Initialize the GraphRAG implementation.

        This method is called to set up any required initial state or connections.
        Currently implemented as a no-op placeholder for future initialization logic.
        """
        pass

    async def create(self, index_name):
        """
        Create a new index for storing graph data.

        Args:
            index_name: Name of the index to create

        Note: Currently implemented as a no-op placeholder. Actual index creation
        logic should be implemented based on the storage backend requirements.
        """
        pass

    async def delete(self, unique_name: str, doc_ids: List[str] | str) -> None:
        """
        Delete documents from the graph storage.

        This method removes documents and their associated graph entities from storage.
        It handles both single document ID strings and lists of document IDs.

        Args:
            unique_name: Unique identifier for the storage namespace
            doc_ids: Document ID(s) to delete - can be a single string or list of strings
        """
        await self.storage.delete(
            unique_name,
            [doc_ids] if isinstance(doc_ids, str) else doc_ids,
        )

    def _to_virtices(
        self, index_name: str, entities: List[Document], doc: Document, entities_facts: Dict[str, List[str]]
    ) -> List[GraphVertex]:
        """
        Convert entities and documents to graph vertices.

        This method transforms extracted entities and the source document into graph vertex
        objects that can be stored in the graph database. Each entity becomes a vertex
        with associated facts and metadata, and the document itself becomes a vertex.

        Args:
            index_name: Name of the index for labeling vertices
            entities: List of entity documents extracted from the source document
            doc: Source document containing the entities
            entities_facts: Mapping of entity content to their associated facts

        Returns:
            List[GraphVertex]: List of graph vertices representing entities and document
        """
        vertices = []

        # Convert each entity to a graph vertex
        for entity in entities:
            metadata = entity.metadata
            # Associate facts with this entity
            metadata["facts"] = entities_facts.get(entity.content, [])

            vertices.append(
                GraphVertex(
                    uid=entity.uid,
                    name=entity.uid,
                    content=entity.content,
                    description={doc.uid: entity.summary},  # Link entity description to source document
                    metadata=entity.metadata,
                    namespace=Namespace.ENTITY.value,
                    embedding=entity.embedding,
                    label=index_name,
                    doc_id=[metadata.get("doc_id")] if "doc_id" in metadata else [],
                )
            )
            logger.debug(f"Converted entity to vertex: {entity.content} with description: {entity.summary}")
        # Add the source document as a vertex
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
                doc_id=[],  # Documents don't reference other documents in this context
            )
        )
        return vertices

    def _to_edges(self, index_name: str, result: GraphExtractOperatorOutputModel, doc_id: str) -> List[GraphEdge]:
        """
        Convert extracted graph relationships to graph edges.

        This method transforms relationship triples (subject, predicate, object) extracted
        from text into graph edge objects that represent connections between entities.

        Args:
            index_name: Name of the index for labeling edges
            result: Graph extraction result containing relationship triples
            doc_id: ID of the source document for tracking provenance

        Returns:
            List[GraphEdge]: List of graph edges representing entity relationships
        """
        edges: List[GraphEdge] = []

        # Convert each relationship triple to a graph edge
        edges.extend(
            [
                GraphEdge(
                    uid=GraphHelper.generate_edge_id(index_name, edge[0], edge[1], edge[2]),
                    source=GraphHelper.generate_vertex_id(index_name, edge[0]),  # Subject entity
                    target=GraphHelper.generate_vertex_id(index_name, edge[2]),  # Object entity
                    source_content=edge[0],
                    target_content=edge[2],
                    weight=1.0,  # Default weight
                    relation_type=RelationType.RELATION,
                    relation=edge[1],  # Predicate/relationship type
                    # Relationship description from source document
                    description={doc_id: edge[3]},
                    label=index_name,
                    edge_metadata={
                        "doc_id": doc_id,
                        "relation_type": edge[1],
                        "summary": [edge[3]],
                    },
                )
                for edge in result.triples  # Each edge is a tuple: (subject, predicate, object, description)
            ]
        )

        return edges

    def _build_entity_chunk_edge(self, index_name: str, entities: List[Document], doc: Document) -> List[GraphEdge]:
        """
        Build edges connecting entities to their source document chunks.

        This method creates "INCLUDE" relationships between documents and the entities
        they contain, establishing provenance and enabling document-to-entity lookups.

        Args:
            index_name: Name of the index for labeling edges
            entities: List of entity documents extracted from the source document
            doc: Source document containing the entities

        Returns:
            List[GraphEdge]: List of edges connecting document to its entities
        """
        edges: List[GraphEdge] = []
        for entity in entities:
            doc_id: str = entity.metadata["doc_id"]
            doc_content = doc.content
            edges.append(
                GraphEdge(
                    uid=GraphHelper.generate_edge_id(index_name, entity.uid, RelationType.INCLUDE.value, doc_id),
                    source=doc_id,  # Document as source
                    target=entity.uid,  # Entity as target
                    source_content=doc_content,
                    target_content=entity.content,
                    label=index_name,
                    edge_metadata={},
                    weight=1.0,
                    relation_type=RelationType.INCLUDE,  # INCLUDE relationship type
                    description={doc.uid: f"Entity {entity.content} is included in document {doc_content}."},
                )
            )

        return edges

    def _entities_bind_to_docs(self, entities: List[Document], doc: Document) -> Document:
        """
        Bind entities to their source document by updating document metadata.

        This method creates a bidirectional reference between documents and their
        extracted entities by adding entity IDs to the document's metadata.

        Args:
            entities: List of entity documents extracted from the source document
            doc: Source document to update with entity references

        Returns:
            Document: Updated document with entity IDs in metadata
        """
        entity_ids = [entity.uid for entity in entities]
        doc.metadata["entities"] = entity_ids
        return doc

    async def _add_document(
        self, index_name: str, document: Document, ge: GraphExtractOperatorOutputModel, lang="en"
    ) -> Document:
        """
        Add a single document to the graph storage with extracted entities and relationships.

        This method processes a document by:
        1. Extracting entities and creating entity documents
        2. Converting relationships to graph edges
        3. Building entity-fact associations
        4. Storing everything in the graph database
        5. Creating communities and summaries

        Args:
            index_name: Name of the index for storing graph data
            document: Source document to process and store
            ge: Graph extraction result containing entities and relationships
            lang: Language code for processing (default: "en")

        Returns:
            Document: Processed document with updated metadata

        Raises:
            RuntimeError: If storage operations fail
        """
        entities: List[Document] = []
        edges: List[GraphEdge] = []
        entities_facts: Dict[str, List[str]] = {}

        # Ensure document has a unique ID
        document.uid = document.uid or compute_mdhash_id(index_name, document.content, Namespace.DOC.value)
        document.metadata = document.metadata or {}

        # Create entity documents from extracted entities
        entities.extend(
            [
                Document(
                    uid=GraphHelper.generate_vertex_id(index_name, entity[0]),
                    content=entity[0],  # Entity name/content
                    summary=entity[1],  # Entity description
                    metadata={"doc_id": document.uid, "namespace": "entity"},
                )
                for entity in ge.entities  # Each entity is a tuple: (name, description)
            ]
        )

        # Convert relationship triples to graph edges
        edges = self._to_edges(index_name, ge, document.uid)

        # Build entity-fact associations for each entity involved in relationships
        for edge in edges:
            # Associate facts with source entity
            if edge.source not in entities_facts:
                entities_facts[edge.source] = []
            entities_facts[edge.source].append(
                "#".join([edge.source, edge.relation_type, edge.target, edge.description[document.uid]])
            )

            # Associate facts with target entity
            if edge.target not in entities_facts:
                entities_facts[edge.target] = []
            entities_facts[edge.target].append(
                "#".join([edge.source, edge.relation_type, edge.target, edge.description[document.uid]])
            )

        # Handle case where no entities were extracted
        if not entities:
            logger.warning(f"No entities extracted from document: {document.content}")
            await self.storage.add_documents(self._doc_index.format(index_name), [document])
            return document

        # Store entities in the entities index
        entities = await self.storage.add_documents(self._entities_index.format(index_name), entities)

        # Prepare concurrent storage operations
        storage_tasks = []
        document = self._entities_bind_to_docs(entities, document)

        # Add document and vertices concurrently
        storage_tasks.append(self.storage.add_documents(self._doc_index.format(index_name), [document]))
        storage_tasks.append(
            self.storage.add_virtices(index_name, self._to_virtices(index_name, entities, document, entities_facts))
        )

        # Execute storage operations and handle errors
        storage_result = await asyncio.gather(*storage_tasks, return_exceptions=False)
        for r in storage_result:
            if isinstance(r, BaseException):
                raise RuntimeError(f"Add documents error:{str(r)}") from r

        # Add entity-document inclusion edges
        edges.extend(self._build_entity_chunk_edge(index_name, entities, document))
        await self.storage.add_edges(index_name, edges)

        # Generate and store community summaries
        communities = await self.summary_communities(index_name, lang)
        if communities:
            await self._save_communities(self._communities_index.format(index_name), communities, document.uid)
        return document

    async def add_documents(self, index_name, documents: List[Document], lang="en", batch_size: int = 10):
        """
        Add multiple documents to the graph with optimized batch processing.

        This method processes large document collections efficiently by:
        1. Pre-processing to ensure all documents have unique IDs
        2. Batching documents to manage memory and resource usage
        3. Parallel processing within each batch
        4. Error isolation to prevent single failures from stopping entire process

        Args:
            index_name: Name of the index for storing graph data
            documents: List of Document objects to add to the graph
            lang: Language code for text processing (default: "en")
            batch_size: Number of documents to process in each batch (default: 10)

        Returns:
            List[Document]: Successfully processed documents
        """
        if not documents:
            logger.warning("No documents to add.")
            return []

        # Pre-processing: ensure all documents have unique IDs
        for i, document in enumerate(documents):
            if not document.uid:
                document.uid = compute_mdhash_id(index_name, document.content, Namespace.DOC.value)

        # Process documents in batches to manage memory and performance
        all_results = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1}: {len(batch)} documents")

            try:
                batch_results = await self._process_document_batch(index_name, batch, lang)
                all_results.extend(batch_results)
            except Exception as e:
                logger.error(f"Error processing batch {i // batch_size + 1}: {e}\n{traceback.format_exc()}")
                # Continue processing next batch rather than failing completely
                continue

        return all_results

    async def _process_document_batch(self, index_name: str, documents: List[Document], lang: str) -> List[Document]:
        """
        Process a batch of documents through graph extraction and storage.

        This method handles the core document processing pipeline:
        1. Load contextual information for each document
        2. Extract graph structures (entities and relationships) in parallel
        3. Filter out failed extractions and continue with valid results
        4. Add documents to storage with error isolation

        Args:
            index_name: Name of the index for storing graph data
            documents: Batch of documents to process
            lang: Language code for text processing

        Returns:
            List[Document]: Successfully processed documents from the batch
        """
        # Load contextual information for graph extraction
        text_context_map = await self.aload_chunk_context(index_name, documents)

        # Create parallel graph extraction tasks
        tasks = []
        for key, context in text_context_map.items():
            tasks.append(
                self._operators.extract_graph(
                    {"histories": context, "text": key, "index_name": self._context_history_index.format(index_name)},
                    lang,
                )
            )

        # Execute graph extractions with exception handling
        results: List[GraphExtractOperatorOutputModel] = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out extraction failures and prepare valid results
        valid_results = []
        valid_documents = []
        for i, (doc, result) in enumerate(zip(documents, results)):
            if isinstance(result, BaseException):
                logger.error(f"Error extracting graph from document {doc.uid}: {result}")
                continue
            valid_results.append(result)
            valid_documents.append(doc)

        if not valid_results:
            logger.warning("No valid graph extraction results in this batch.")
            return []

        # Process valid documents in parallel
        tasks = []
        for i, document in enumerate(valid_documents):
            tasks.append(self._add_document(index_name, document, valid_results[i], lang))

        processed_documents = await asyncio.gather(*tasks, return_exceptions=True)
        successful_documents = []

        # Collect successfully processed documents
        for i, document in enumerate(processed_documents):
            if isinstance(document, BaseException):
                logger.error(f"Error in adding document {valid_documents[i].uid}: {document}")
                continue
            logger.info(f"Document {document.uid} added successfully.")
            successful_documents.append(document)

        return successful_documents

    async def _save_communities(self, index: str, communities: List[GraphCommunity], doc_uid) -> List[Document]:
        """
        Save community summaries as documents in the storage.

        This method converts graph communities into document format for storage,
        preserving community metadata and associations with source documents.

        Args:
            index: Index name for storing community documents
            communities: List of graph communities to save
            doc_uid: UID of the source document that triggered community creation

        Returns:
            List[Document]: Saved community documents
        """
        docs = [
            Document(
                uid=compute_mdhash_id(index, c.summary, Namespace.COMMUNITY.value),
                content=c.summary,
                summary=c.summary,
                metadata={
                    "total": len(communities),  # Total number of communities
                    "vertices": list(set(vs.name for vs in c.graph.vertices)),  # Unique vertex names in community
                    "doc_id": [doc_uid],  # Source document reference
                },
            )
            for c in communities
        ]
        return await self.storage.add_documents(index, docs)

    async def summary_communities(self, label: str, lang="zh") -> List[GraphCommunity]:
        """
        Discover and summarize graph communities using concurrent processing.

        This method performs community detection on the graph and generates
        natural language summaries for each discovered community. It uses
        local locking to prevent cross-event-loop issues.

        Args:
            label: Graph label/index name for community discovery
            lang: Language code for summary generation (default: "zh")

        Returns:
            List[GraphCommunity]: Communities with generated summaries
        """
        # Use local lock to avoid cross-event-loop issues
        lock = asyncio.Lock()
        async with lock:
            # Discover communities in the graph
            community_ids = await self.storage.discover_communities(label)
            logger.debug(f"Found {len(community_ids)} communities to summarize in label '{label}'.")

            if not community_ids:
                logger.info(f"No communities found in label '{label}'.")
                return []

            # Load community data and prepare summarization tasks
            tasks = []
            communities: List[GraphCommunity] = []
            for cid in community_ids:
                community: GraphCommunity = await self.storage.get_community(label, cid)
                if not community:
                    logger.warning(f"Community with ID {cid} not found in label '{label}'.")
                    continue
                communities.append(community)
                # Format community graph for summarization
                graph = GraphHelper.format_community(community)
                tasks.append(self._operators.summarize_communities({"graph": graph}, lang))

            if not tasks:
                logger.warning(f"No valid communities found to summarize in label '{label}'.")
                return []

            # Execute summarization tasks in parallel
            results: List[SummarizeOperatorOutputModel] = await asyncio.gather(*tasks, return_exceptions=False)

            # Update communities with generated summaries
            for i, community in enumerate(communities):
                if i < len(results):
                    communities[i].summary = results[i].summary
            return communities

    async def aload_chunk_context(self, index_name: str, texts: List[Document]) -> Dict[str, str]:
        """
        Load contextual information for documents by finding similar existing chunks.

        This method helps provide relevant context for graph extraction by finding
        previously processed similar content. It enables better entity and relationship
        extraction by providing historical context.

        Args:
            index_name: Name of the index for searching similar content
            texts: List of documents to find context for

        Returns:
            Dict[str, str]: Mapping from document content to its contextual information
        """
        text_context_map: Dict[str, str] = {}
        tasks = []

        # Create parallel similarity search tasks for each document
        for text in texts:
            tasks.append(
                self.storage.similar_search_with_scores(
                    self._context_history_index.format(index_name), text.content, self._topk
                )
            )

        histories = []
        results = await asyncio.gather(*tasks)

        # Process similarity search results for each document
        for text, chunks in zip(texts, results):
            # Filter chunks based on similarity score threshold
            chunks = [(chunk, score) for chunk, score in chunks if score >= self._score_threshold]
            # Sort chunks by relevance score (highest first)
            chunks.sort(key=lambda x: x[1], reverse=True)

            # Format context sections from similar chunks
            history = [f"Section {i + 1}:\n{chunk[0].content}" for i, chunk in enumerate(chunks)]
            context = "\n".join(history) if history else ""
            text_context_map[text.content] = context

            # Store context metadata for future reference
            histories.append(
                Document(
                    uid=compute_mdhash_id(index_name, text.content, Namespace.CONTEXT.value),
                    content=text.content,
                    metadata={
                        "relevant_cnt": len(history),  # Number of relevant context sections found
                        "doc_id": [text.uid],  # Source document reference
                    },
                )
            )

        # Save context history for future similarity searches
        await self.storage.add_documents(self._context_history_index.format(index_name), histories)
        return text_context_map

    async def _retrieve_query(self, unique_name: str, query: str, top_k: int = 5) -> GraphRAGRetrieveResultItem:
        """
        Retrieve documents from the graph storage system using multi-modal search.

        This method implements the core GraphRAG retrieval pipeline:
        1. Detect query language and intent
        2. Search communities for high-level context
        3. Extract keywords and entities from the query
        4. Search graph structure for relevant subgraphs
        5. Retrieve documents based on graph traversal results

        Args:
            unique_name: Name of the index to search in
            query: Query string for document retrieval
            top_k: Number of top results to return (default: 5)

        Returns:
            GraphRAGRetrieveResultItem: Comprehensive retrieval result including context,
                                      graph structure, and relevant documents
        """
        # Detect query language and default to English if unsupported
        lang = detect_language(query)
        if lang not in ["zh", "en"]:
            lang = "en"

        # Set up parallel processing tasks for multi-modal retrieval
        tasks = []

        # Search for relevant communities (high-level context)
        tasks.append(self.storage.search_communities(self._communities_index.format(unique_name), query, top_k))
        # Extract keywords for entity matching
        tasks.append(self._operators.extract_keywords({"text": query}, lang))
        # Detect query intent and entities
        tasks.append(self._operators.detect_query_indent({"text": query}, lang))
        # Execute parallel retrieval tasks
        results: List[
            GraphCommunity | KeywordsOperatorOutputModel | QueryIndentDetectionOperator
        ] = await asyncio.gather(*tasks, return_exceptions=False)

        # Extract results from parallel operations
        communities: List[GraphCommunity] = results[0]
        keywords: KeywordsOperatorOutputModel = results[1]
        query_indent: QueryIndentationModel = results[2]

        # Initialize result variables
        subgraph = ""
        docs: List[Document] = []

        # Combine entities from query intent detection and keyword extraction
        entities: List[str] = [entity for entity in query_indent.entities]
        entities.extend(keywords.keywords)
        entities = list(set(entities))  # Remove duplicates

        # Community deduplication: use community ID instead of summary for uniqueness
        seen_cids = set()
        unique_communities = []
        for c in communities:
            if c.cid not in seen_cids:
                unique_communities.append(c)
                seen_cids.add(c.cid)
        communities = unique_communities

        # Build context from community summaries
        summaries = [f"Section {i + 1}:\n{community.summary}" for i, community in enumerate(communities)]
        context = "\n".join(summaries) if summaries else ""

        # Primary graph search strategy: query intent-based search
        graph: GraphModel | None = None
        if query_indent.entities:
            logger.debug(f"Query indent detected: {query_indent.category} with {query_indent.entities}")
            graph = await self.storage.search_graph_by_indent(unique_name, query_indent)
            if graph and graph.vertices:
                # Extract documents from graph vertices
                docs = [
                    Document(content=doc.content, summary="", uid=doc.uid, metadata=doc.metadata)
                    for doc in graph.vertices
                    if doc.namespace == Namespace.DOC.value
                ]
                # Keep only entity vertices for subgraph visualization
                graph.vertices = [doc for doc in graph.vertices if doc.namespace == Namespace.ENTITY.value]

        # Fallback graph search strategy: keyword and similarity-based search
        if not graph or not graph.vertices:
            graph = await self._search_subgraph(unique_name, keywords, query_indent, top_k)
            if graph and graph.vertices:
                # Extract documents from fallback graph search
                docs = [
                    Document(content=doc.content, summary=doc.description[doc.uid], uid=doc.uid, metadata=doc.metadata)
                    for doc in graph.vertices
                    if doc.namespace == Namespace.DOC.value
                ]
                # Keep only entity vertices for subgraph
                graph.vertices = [doc for doc in graph.vertices if doc.namespace == Namespace.ENTITY.value]

        # Final fallback: direct entity-to-document lookup
        if not docs:
            if not graph or not graph.vertices:
                # Use extracted entities for direct document lookup
                docs = await self.storage.get_docs_by_entities(unique_name, entities)
            else:
                # Use graph vertices for document lookup
                docs = await self.storage.get_docs_by_entities(unique_name, [v.content for v in graph.vertices])

            logger.debug(f"Retrieved {len(docs)} documents by entities: {entities}")

        # Prepare final subgraph representation
        subgraph = ""
        if graph and graph.vertices:
            # Remove duplicate vertices and edges for clean representation
            graph.vertices = duplicate_filter(graph.vertices)
            graph.edges = duplicate_filter(graph.edges)
            subgraph = GraphHelper.format_graph(graph) if graph and graph.vertices else ""

        # Return comprehensive retrieval result
        return GraphRAGRetrieveResultItem(
            query=query, context=context, graph=graph, subgraph=subgraph, docs=duplicate_filter(docs)
        )

    async def _search_subgraph(
        self, unique_name, keywords: KeywordsOperatorOutputModel, query_indent: QueryIndentationModel, top_k=5
    ) -> GraphModel | None:
        """
        Search for relevant subgraph using keywords and query intent.

        This method implements a fallback search strategy when direct intent-based
        search doesn't yield results. It combines entity matching and similarity
        search to find relevant graph structures.

        Args:
            unique_name: Index name for graph search
            keywords: Extracted keywords from the query
            query_indent: Query intent and entity detection results
            top_k: Number of top similar entities to consider (default: 5)

        Returns:
            GraphModel | None: Relevant subgraph or None if no matches found
        """
        # Start with entities detected from query intent
        entities = [GraphHelper.generate_vertex_id(unique_name, entity) for entity in query_indent.entities] or []

        # Expand search with similar entities based on keywords
        if keywords.keywords:
            similar_entities = await self._search_similar_entities(unique_name, keywords.keywords, top_k)
            entities.extend(
                [
                    GraphHelper.generate_vertex_id(unique_name, entity.content)
                    for entity in similar_entities
                    if entity.content
                ]
            )

        # Return None if no entities found for graph exploration
        if not entities:
            return None

        # Explore trigraph (3-hop neighborhood) around identified entities
        graph = await self.storage.explore_trigraph(unique_name, list(set(entities)))
        return graph

    async def _search_similar_entities(
        self, index_name: str, keywords: List[str], top_k: int = 5, score_threshold: Optional[float] = 0.7
    ) -> List[Document]:
        """
        Search for entities similar to the given keywords using embedding similarity.

        This method performs parallel similarity searches for each keyword and
        aggregates results with score-based filtering to find relevant entities.

        Args:
            index_name: Name of the entity index to search
            keywords: List of keywords to find similar entities for
            top_k: Number of top similar entities per keyword (default: 5)
            score_threshold: Minimum similarity score threshold (default: 0.8)

        Returns:
            List[Document]: List of similar entity documents
        """
        similar_entities: List[Document] = []
        query_tasks = []

        # Create parallel similarity search tasks for each keyword
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
        history_context: Optional[str] = "",
        top_k: int = 5,
        similarity_threshold=0.2,
        prompt: Optional[str] = None,
        llm: Optional[BaseChat] = None,
        reranker: Optional[BaseRerank] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[QAChatResponse]:
        from langchain.prompts import ChatPromptTemplate

        # Execute graph retrieval and embedding retrieval in parallel to improve performance
        graph_task = self._retrieve_query(unique_name, query, top_k)
        embedding_task = self.embedding_retrieve(unique_name, queries=[query], filter=filter)

        result, embedding_retrieve_result = await asyncio.gather(graph_task, embedding_task)

        # Merge retrieval results
        docs = embedding_retrieve_result.get(query, [])
        result.docs.extend([doc.document for doc in docs if doc.document.content])

        # Remove duplicates
        seen_uids = set()
        unique_docs = []
        for doc in result.docs:
            if doc.uid not in seen_uids:
                unique_docs.append(doc)
                seen_uids.add(doc.uid)
        result.docs = unique_docs

        # Reranking
        reranker = reranker or self._reranker
        if reranker:
            rerank_docs = await reranker.rerank(query, result.docs)
            logger.debug(f"Rerank docs: {rerank_docs}")
            uid_to_docs = {doc.uid: doc for doc in result.docs}
            result.docs = [uid_to_docs[uid] for uid, score in rerank_docs.items() if score >= similarity_threshold]
            logger.debug(f"Reranked documents: {[doc.content for doc in result.docs]}")

        # Build context
        lang = detect_language(query)
        if lang not in ["zh", "en"]:
            lang = "en"
        context = (
            result.context + "\n\n" + "History context: \n" + history_context + "\n\n"
            if history_context
            else result.context
        )

        # Build prompt template
        should_container_vars = ["question", "context", "knowledge_graph", "knowledge_graph_for_doc"]

        if prompt:
            cpt = ChatPromptTemplate.from_template(prompt)
            input_vars = cpt.input_variables
            missing = set(should_container_vars) - set(input_vars)
            if missing:
                raise ValueError(
                    f"Prompt is missing required variables: {missing}. Required variables are: {should_container_vars}"
                )
            qa_prompt = cpt.invoke(
                dict(
                    question=query,
                    context=context,
                    knowledge_graph=result.subgraph,
                    knowledge_graph_for_doc="\n".join([doc.content for doc in result.docs]),
                )
            ).to_string()
        else:
            qa_prompt = self.qa_prompt_builder.build(
                question=query,
                context=context,
                knowledge_graph=result.subgraph,
                knowledge_graph_for_doc="\n".join([doc.content for doc in result.docs]),
            )

        logger.debug(f"Prompt for query '{query}': {qa_prompt}")
        llm = llm or self.llm

        async for answer in llm.astream(prompt=ChatPromptTemplate.from_template(qa_prompt), input={}):
            ans: ChatResponse = answer
            context = {
                "context": result.context,
                "knowledge_graph": result.graph.model_dump() if result.graph else "",
                "knowledge_graph_for_doc": "\n".join([doc.content for doc in result.docs]),
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
                RetrieveResultItem(document=doc, score=1.0, query=query, metadata=results[i].metadata)
                for doc in results[i].docs
            ]
        return retrieve_results

    async def embedding_retrieve(
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
                self._doc_index.format(unique_name), queries[0], 100, filter
            )
            embedding_retrieval_results[queries[0]] = [
                RetrieveResultItem(document=doc, score=score, query=queries[0])
                for doc, score in embedding_retrieval_result
            ]
        else:
            tasks = [
                self.storage.similar_search_with_scores(self._doc_index.format(unique_name), query, 100, filter)
                for query in queries
            ]
            results = await asyncio.gather(*tasks, return_exceptions=False)
            for i, query in enumerate(queries):
                embedding_retrieval_results[query] = [
                    RetrieveResultItem(document=doc, score=score, query=query) for doc, score in results[i]
                ]
        return embedding_retrieval_results
