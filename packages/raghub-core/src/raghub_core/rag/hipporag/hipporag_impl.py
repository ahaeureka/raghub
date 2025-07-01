import asyncio
import copy
import difflib
import itertools
import json
import re
import traceback
from asyncio import Lock
from typing import Any, AsyncIterator, Dict, List, Optional, Set, Tuple

import numpy as np
from loguru import logger
from raghub_core.chat.base_chat import BaseChat
from raghub_core.embedding import BaseEmbedding
from raghub_core.operators.ner.ner_operator import NEROperator
from raghub_core.operators.ner.promopts import NERPrompt
from raghub_core.operators.openie.openie_operator import OpenIEOperator
from raghub_core.operators.rdf.prompts import REPrompt
from raghub_core.operators.rdf.rdf_operator import RDFOperator
from raghub_core.prompts.qa import DefaultQAPromptBuilder, QAPromptBuilder
from raghub_core.rag.base_rag import BaseRAG
from raghub_core.rag.hipporag.hipporag_storage import HipporagStorage
from raghub_core.rag.hipporag.prompts import DSPyRerankPrompt, get_query_instruction
from raghub_core.rerank.base_rerank import BaseRerank
from raghub_core.schemas.chat_response import ChatResponse, QAChatResponse
from raghub_core.schemas.document import Document
from raghub_core.schemas.graph_model import GraphEdge, GraphVertex, Namespace, RelationType
from raghub_core.schemas.hipporag_models import OpenIEInfo
from raghub_core.schemas.openie_mdoel import OpenIEModel
from raghub_core.schemas.rag_model import RetrieveResultItem
from raghub_core.storage.graph import GraphStorage
from raghub_core.storage.vector import VectorStorage
from raghub_core.utils.graph.graph_helper import GraphHelper
from raghub_core.utils.misc import compute_mdhash_id, detect_language
from tqdm import tqdm

from .rerank import DSPyFilter


class HippoRAGImpl(BaseRAG):
    """
    HippoRAG implementation for Multi-hop Knowledge Graph reasoning.

    This implementation combines dense passage retrieval with graph-based reasoning
    using personalized PageRank for enhanced retrieval accuracy.

    Optimizations include:
    - Batch processing for document operations
    - Concurrent embedding generation
    - Efficient graph traversal algorithms
    - Memory-optimized data structures
    """

    def __init__(
        self,
        llm: BaseChat,
        reranker: BaseRerank,
        embedder: BaseEmbedding,  # TODO: Fix typo in parameter name in future version
        embedding_store: VectorStorage,
        graph_store: GraphStorage,
        hipporag_store: HipporagStorage,
        dspy_file_path: str,
        embedding_prefix: str = "entity_embeddings",
        linking_top_k: int = 15,
        passage_node_weight: float = 0.05,
        graph_path: Optional[str] = None,
        synonymy_edge_topk: int = 2047,
        synonymy_edge_query_batch_size: int = 1000,
        synonymy_edge_key_batch_size: int = 1000,
        synonymy_edge_sim_threshold: float = 0.8,
        qa_prompt_builder: QAPromptBuilder = DefaultQAPromptBuilder(),
        # Performance tuning parameters
        max_concurrent_documents: int = 10,
        embedding_batch_size: int = 32,
        graph_batch_size: int = 100,
    ):
        # Initialize parent class first
        super().__init__()

        # Core components
        self._llm = llm
        self._reranker = reranker
        self._embedder = embedder
        self._embedd_store: VectorStorage = embedding_store
        self._graph_store: GraphStorage = graph_store
        self._db = hipporag_store

        # Configuration parameters
        self.linking_top_k = linking_top_k
        self.passage_node_weight = passage_node_weight
        self.graph_path = graph_path
        self.synonymy_edge_topk = synonymy_edge_topk
        self.synonymy_edge_query_batch_size = synonymy_edge_query_batch_size
        self.synonymy_edge_key_batch_size = synonymy_edge_key_batch_size
        self.synonymy_edge_sim_threshold = synonymy_edge_sim_threshold
        self._embedding_prefix = embedding_prefix

        # Performance optimization parameters
        self._max_concurrent_documents = max_concurrent_documents
        self._embedding_batch_size = embedding_batch_size
        self._graph_batch_size = graph_batch_size

        # Concurrency control
        self._lock = Lock()
        self._document_semaphore = asyncio.Semaphore(max_concurrent_documents)

        # Initialize operators
        self._initialize_operators(dspy_file_path)

        # Namespace prefixes
        self._entity_prefix = Namespace.ENTITY.value
        self._fact_prefix = Namespace.FACT.value
        self._doc_prefix = Namespace.DOC.value

        # State management
        self.already_init = False
        self._ready_to_retrieve = False
        self._qa_prompt_builder = qa_prompt_builder

    def _initialize_operators(self, dspy_file_path: str) -> None:
        """Initialize NLP operators for entity and relation extraction."""
        try:
            ner_prompt = NERPrompt()
            self._ner = NEROperator(ner_prompt, self._llm)
            rdf_prompt = REPrompt()
            self._rdf = RDFOperator(rdf_prompt, self._llm)
            self._openie = OpenIEOperator(self._ner, self._rdf)
            dsp_prompt = DSPyRerankPrompt(best_dspy_path=dspy_file_path)
            self.rerank_filter = DSPyFilter(dsp_prompt, self._llm)
        except Exception as e:
            logger.error(f"Failed to initialize operators: {e}")
            raise RuntimeError(f"Operator initialization failed: {e}") from e

    async def create(self, index_name: str):
        if not self.already_init:
            await self.init()
        await self._db.create_new_index(index_name)

    async def _add_document(self, index_name: str, doc: Document) -> Document:
        """
        Adds a single document to the vector store and graph store.

        Args:
            index_name: The name of the index where the document will be added.
            doc: The Document object to be added to the vector store and graph store.

        Returns:
            Document: The processed document with updated metadata.

        Raises:
            RuntimeError: If document processing fails.
        """
        try:
            # Input validation
            if not doc.content or not doc.content.strip():
                logger.warning(f"Empty document content for document {doc.uid}")
                return doc

            lang = detect_language(doc.content)
            chunk_triples: Dict[str, List[Tuple[str, str, str]]] = {}
            openie_infos: List[OpenIEInfo] = []
            entities_str = []
            doc_to_triple_entities: Dict[str, List[List[str]]] = {}
            facts_str = []

            # Generate UID if not provided
            if not doc.uid:
                doc.uid = compute_mdhash_id(index_name, doc.content, prefix=self._doc_prefix)

            # Initialize metadata with namespace
            doc.metadata = doc.metadata or {}
            doc.metadata["namespace"] = Namespace.PASSAGE.value
            doc.metadata["openie_idx"] = doc.uid

            # Extract entities and relations using OpenIE
            ie: OpenIEModel = await self._openie.extract(doc.content, lang=lang)

            if not ie.triples and not ie.ner:
                logger.warning(f"No entities or triples extracted from document {doc.uid}")
                # Still add the document even without extracted knowledge
                docs = await self._embedd_store.add_documents(index_name, [doc])
                return doc

            # Process extracted information
            openie_infos.append(
                OpenIEInfo(
                    idx=doc.uid,
                    passage=doc.content,
                    extracted_triples=[list(item) for item in ie.triples],
                    extracted_entities=list(set(ie.ner)),
                )
            )

            graph_nodes, triple_entities = self._extract_entity_nodes(ie.triples)
            doc_to_triple_entities[doc.uid] = triple_entities
            chunk_triples[doc.uid] = [(item[0], item[1], item[2]) for item in ie.triples if len(item) == 3]
            entities_str.extend([str(node) for node in graph_nodes])

            # Update document metadata with extracted entities and facts
            doc.metadata["entities"] = [
                compute_mdhash_id(index_name, str(node), self._entity_prefix) for node in graph_nodes
            ]
            doc.metadata["facts"] = [compute_mdhash_id(index_name, str(fact), self._fact_prefix) for fact in ie.triples]

            facts = self._flatten_facts([ie.triples])
            facts_str.extend([str(fact) for fact in facts])
            entities_str = list(set(entities_str))
            facts_str = list(set(facts_str))

            logger.debug(
                f"Extracted {len(entities_str)} entities and {len(facts_str)} facts "
                f"from document: {doc.content[:100]}..."
            )

            # Add document to embedding store
            docs = await self._embedd_store.add_documents(index_name, [doc])

            if not entities_str:
                logger.warning(f"No entities extracted from document: {doc.content[:100]}...")
                return doc

            # Add entity and fact embeddings
            embed_entities = await self.add_embeddings(
                index_name, entities_str, self._entity_prefix, metadata={"namespace": Namespace.ENTITY.value}
            )
            logger.info(f"Added {len(embed_entities)} entities to the embedding store.")

            await self.add_embeddings(
                index_name, facts_str, self._fact_prefix, metadata={"namespace": Namespace.FACT.value}
            )

            # Save OpenIE information
            await self._db.save_openie_info(index_name, openie_infos)

            # Create entity nodes
            entities_nodes = [
                Document(
                    uid=compute_mdhash_id(index_name, entity, self._entity_prefix),
                    content=entity,
                    metadata={
                        "namespace": Namespace.ENTITY.value,
                        "doc_id": [doc.uid],
                        "openie_idx": doc.metadata.get("openie_idx", ""),
                    },
                )
                for entity in entities_str
            ]

            await self._add_new_nodes(index_name, entities_nodes, docs)

            # Build graph edges
            ent_node_to_chunk_ids, node_to_node_stats = self._add_fact_edges(index_name, docs, chunk_triples)
            num_new_chunks = self._add_passage_edges(index_name, docs, doc_to_triple_entities, node_to_node_stats)
            logger.info(f"Added {num_new_chunks} new chunks.")

            # Add synonymy edges for entity linking
            entities: List[Document] = await self._embedd_store.select_on_metadata(
                index_name, {"namespace": Namespace.ENTITY.value}
            )
            if entities:
                self._add_synonymy_edges(entities, entities, node_to_node_stats)

            # Augment graph with new information
            await self._augment_graph(index_name, embed_entities, [doc], node_to_node_stats)
            self._save()

            # Update database caches
            await self._update_database_caches(index_name, ent_node_to_chunk_ids, node_to_node_stats, openie_infos)

            return doc

        except Exception as e:
            logger.error(f"Failed to add document {doc.uid}: {e}")
            raise RuntimeError(f"Document addition failed for {doc.uid}: {e}") from e

    async def _update_database_caches(
        self,
        index_name: str,
        ent_node_to_chunk_ids: Dict[str, List[str]],
        node_to_node_stats: Dict[Tuple[str, str], float],
        openie_infos: List[OpenIEInfo],
    ) -> None:
        """Update database caches with new graph information."""
        try:
            # Update entity-to-chunk mappings
            for key, chunks in ent_node_to_chunk_ids.items():
                exists_ent_chunks = await self._db.get_ent_node_to_chunk_ids(index_name, key) or []
                await self._db.set_ent_node_to_chunk_ids(
                    index_name, key, list(set(chunks).union(set(exists_ent_chunks)))
                )

            # Update node-to-node statistics
            for node_to_node, stats in node_to_node_stats.items():
                await self._db.set_node_to_node_stats(index_name, node_to_node[0], node_to_node[1], stats)

            # Update triples-to-documents mappings
            triples_to_docs = self.get_proc_triples_to_docs(openie_infos)
            await self._db.set_triples_to_docs(index_name, triples_to_docs)

        except Exception as e:
            logger.error(f"Failed to update database caches: {e}")
            # Don't raise here as the main document processing succeeded

    async def add_documents(self, index_name: str, texts: List[Document]) -> List[Document]:
        """
        Adds documents to the vector store and graph store with optimized batch processing.

        This method processes documents concurrently while respecting memory and resource limits.
        It uses semaphores to control concurrency and batches operations for better performance.

        Args:
            index_name: The name of the index where documents will be added.
            texts: List of Document objects to be added to the vector store and graph store.

        Returns:
            List[Document]: The processed documents with updated metadata.

        Raises:
            RuntimeError: If batch document processing fails.
        """
        if not texts:
            return []

        logger.info(f"Processing {len(texts)} documents with batch optimization")

        try:
            # Process documents in batches to manage memory and resources
            batch_size = min(self._max_concurrent_documents, len(texts))
            results = []

            # Use semaphore to limit concurrent document processing
            async def process_with_semaphore(doc: Document) -> Document:
                async with self._document_semaphore:
                    return await self._add_document(index_name, doc)

            # Process documents in batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                logger.debug(f"Processing batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}")

                # Create tasks for concurrent processing within the batch
                tasks = [process_with_semaphore(doc) for doc in batch]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                # Handle exceptions from batch processing
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Failed to process document {batch[j].uid}: {result}")
                        # Continue with other documents rather than failing the entire batch
                        continue
                    results.append(result)

                # Small delay between batches to prevent overwhelming the system
                if i + batch_size < len(texts):
                    await asyncio.sleep(0.1)

            logger.info(f"Successfully processed {len(results)} out of {len(texts)} documents")
            return results

        except Exception as e:
            logger.error(f"Batch document processing failed: {e}")
            raise RuntimeError(f"Failed to process documents in batch: {e}") from e

    async def _get_query_embeddings(self, queries: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Retrieves and computes embeddings for the given queries.

        This function checks if the embeddings for the queries already exist in the
        `query_to_embedding` dictionary. If not, it computes the embeddings using
        the `_embedder` and stores them in the dictionary.

        Args:
            queries : List[str] A list of query strings for which embeddings need to be computed.
        Returns:
            Dict[str, Dict[str, np.ndarray]]
                A dictionary containing the computed embeddings for the queries.
                The keys are the query strings, and the values are the corresponding
                embeddings.
        """
        all_query_strings = []
        query_to_embedding: Dict[str, Dict[str, np.ndarray]] = {
            "triple": {},
            "passage": {},
        }  # Temporary storage for new embeddings
        for query in queries:
            if query not in query_to_embedding["triple"] or query not in query_to_embedding["passage"]:
                all_query_strings.append(query)

        if len(all_query_strings) > 0:
            # get all query embeddings
            logger.info(f"Encoding {len(all_query_strings)} queries for query_to_fact.")
            query_embeddings_for_triple = await self._embedder.aencode_query(
                all_query_strings, instruction=get_query_instruction("query_to_fact")
            )
            for query, embedding in zip(all_query_strings, query_embeddings_for_triple):
                query_to_embedding["triple"][query] = embedding

            logger.info(f"Encoding {len(all_query_strings)} queries for query_to_passage.")
            query_embeddings_for_passage = await self._embedder.aencode_query(
                all_query_strings, instruction=get_query_instruction("query_to_passage")
            )
            for query, embedding in zip(all_query_strings, query_embeddings_for_passage):
                query_to_embedding["passage"][query] = embedding
            return query_to_embedding

        return query_to_embedding

    async def _get_fact_scores(
        self,
        index_name,
        query: str,
        query_to_embedding: Optional[Dict[str, Dict[str, np.ndarray]]] = {},
        top_k: int = 10,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieves and computes normalized similarity scores between the given query and pre-stored fact embeddings.

        Args:
            query : str
                The input query for which similarity scores should be computed.
            query_to_embedding : Optional[Dict[str, Dict[str, np.ndarray]]], optional
                A dictionary containing pre-computed embeddings for the queries.
                If not provided, the function will compute the embeddings for the query.
            top_k : int, optional
                The number of top similar facts to retrieve. Defaults to 10.
        Returns:
            List[Tuple[Document, float]]
                A list of tuples, where each tuple contains a Document object and its corresponding
        """
        query_embedding: np.ndarray = (
            query_to_embedding["triple"].get(query, np.array([]))
            if query_to_embedding and "triple" in query_to_embedding
            else np.array([])
        )
        if query_embedding is None:
            query_embedding = await self._embedder.aencode_query(
                [query], instruction=get_query_instruction("query_to_fact")
            )
        return await self._embedd_store.similarity_search_by_vector(
            index_name=index_name, embedding=query_embedding.tolist(), k=top_k, filter={"namespace": "fact"}
        )

    async def rerank_facts(
        self, query: str, query_fact_scores: List[Tuple[Document, float]], lang="en", link_top_k=5
    ) -> List[Tuple[Document, float]]:
        """
        Reranks the facts based on their relevance to the query using a DSPy model.

        Args:
            query : str
                The input query for which the facts need to be reranked.
            query_fact_scores : List[Tuple[Document, float]]
                A list of tuples containing Document objects and their corresponding scores.
            lang : str, optional
                The language of the query. Defaults to "en".
            link_top_k : int, optional
                The number of top facts to consider for reranking. Defaults to 5.
        Returns:
            List[Tuple[Document, float]]
                A list of tuples containing the reranked Document objects and their corresponding scores.
        """
        # load args

        docs: List[Tuple[Document, float]] = sorted(query_fact_scores, key=lambda x: x[1], reverse=True)

        fact_before_filter = {"fact": [list(eval(doc.content)) for doc, _ in docs]}
        input = {"question": query, "fact_before_filter": json.dumps(fact_before_filter, ensure_ascii=False)}

        output = await self.rerank_filter.execute(input, lang=lang)
        result_indices = []
        candidate_items = [eval(doc.content) for doc, _ in docs]
        for generated_fact in output.fact_after_filter:
            closest_matched_fact = difflib.get_close_matches(
                str(generated_fact), [str(i) for i in candidate_items], n=1, cutoff=0.0
            )[0]
            try:
                result_indices.append(candidate_items.index(eval(closest_matched_fact)))
            except Exception as e:
                logger.error(f"result_indices exception: {e}")
                logger.error(traceback.format_exc())

        sorted_candidate_items: List[Tuple[Document, float]] = [docs[i] for i in result_indices]
        top_k_facts = sorted_candidate_items[:link_top_k]
        # rerank_log = {"facts_before_rerank": candidate_facts, "facts_after_rerank": top_k_facts}

        return top_k_facts

    async def dense_passage_retrieval(
        self, index_name: str, query: str, query_to_embedding: Dict[str, Dict[str, np.ndarray]], top_k: int = 20
    ) -> List[Tuple[Document, float]]:
        """
        Performs dense passage retrieval using the provided query and its corresponding embeddings.
        Args:
            query : str
                The input query for which dense passage retrieval needs to be performed.
            query_to_embedding : Dict[str, Dict[str, np.ndarray]]
                A dictionary containing pre-computed embeddings for the queries.
            top_k : int, optional
                The number of top passages to retrieve. Defaults to 10.
        Returns:
            List[Tuple[Document, float]]
                A list of tuples, where each tuple contains a Document object and its corresponding score.
        """
        logger.debug(f"dense_passage_retrieval query: {query},top_k: {top_k}")
        query_embedding: np.ndarray = query_to_embedding["passage"].get(query, np.array([]))
        if query_embedding is None:
            query_embedding = self._embedder.encode(query, instruction=get_query_instruction("query_to_passage"))
        return await self._embedd_store.similarity_search_by_vector(
            index_name, query_embedding.tolist(), k=top_k, filter={"namespace": "passage"}
        )

    async def graph_search_with_fact_entities(
        self,
        index_name: str,
        query: str,
        query_to_embedding: Dict[str, Dict[str, np.ndarray]],
        link_top_k: int,
        top_k_facts: List[Tuple[Document, float]],
        passage_node_weight: float = 0.05,
        damping: float = 0.5,  # Damping factor for PPR algorithm
        top_k: int = 10,
    ) -> Dict[str, float]:
        """
        Performs optimized graph search using fact entities and passage nodes to compute personalized PageRank scores.

        This method includes several optimizations:
        - Batch processing of entity lookups
        - Efficient phrase deduplication
        - Optimized weight calculations
        - Memory-efficient data structures

        Args:
            index_name: The name of the index.
            query: The input query for which graph search needs to be performed.
            query_to_embedding: A dictionary containing pre-computed embeddings for the queries.
            link_top_k: The number of top facts to consider for graph search.
            top_k_facts: A list of tuples containing Document objects and their corresponding scores.
            passage_node_weight: The weight assigned to passage nodes. Defaults to 0.05.
            damping: Damping factor for the PageRank algorithm. Defaults to 0.5.
            top_k: The number of top nodes to retrieve. Defaults to 10.

        Returns:
            Dict[str, float]: A dictionary containing the personalized PageRank scores for the nodes.

        Raises:
            AssertionError: If no phrases are found in the graph for the given facts.
        """
        if not top_k_facts:
            logger.warning("No facts provided for graph search")
            return {}

        try:
            # Pre-allocate data structures for better memory efficiency
            linking_score_map: Dict[str, float] = {}
            phrase_scores: Dict[str, List[np.float64]] = {}
            phrase_weights: Dict[str, np.float64] = {}
            passage_weights: Dict[str, np.float64] = {}

            # Extract and deduplicate phrase keys efficiently
            phrase_keys_set: Set[str] = set()
            fact_data: List[Tuple[str, str, float]] = []  # (subject, object, score)

            for doc, fact_score in top_k_facts:
                try:
                    fact = eval(doc.content)
                    if len(fact) >= 3:
                        subject_phrase = fact[0].lower()
                        object_phrase = fact[2].lower()

                        # Cache fact data to avoid re-parsing
                        fact_data.append((subject_phrase, object_phrase, fact_score))

                        # Batch compute phrase keys
                        phrase_keys_set.add(
                            compute_mdhash_id(index_name, content=subject_phrase, prefix=self._entity_prefix)
                        )
                        phrase_keys_set.add(
                            compute_mdhash_id(index_name, content=object_phrase, prefix=self._entity_prefix)
                        )
                except (SyntaxError, ValueError) as e:
                    logger.warning(f"Failed to parse fact content: {doc.content}, error: {e}")
                    continue

            phrase_keys = list(phrase_keys_set)

            # Batch lookup graph nodes to reduce database calls
            nodes = self._graph_vertices_to_nodes(
                await self._graph_store.aselect_vertices(index_name, dict(name_in=phrase_keys))
            )
            node_id_to_phrase = {node["name"]: node for node in nodes}

            # Batch lookup entity-to-chunk mappings
            valid_phrase_keys = []

            for subject_phrase, object_phrase, fact_score in fact_data:
                subject_key = compute_mdhash_id(index_name, content=subject_phrase, prefix=self._entity_prefix)
                object_key = compute_mdhash_id(index_name, content=object_phrase, prefix=self._entity_prefix)

                for phrase, phrase_key in [(subject_phrase, subject_key), (object_phrase, object_key)]:
                    if phrase_key in node_id_to_phrase:
                        phrase_weights[phrase_key] = np.float64(fact_score)
                        valid_phrase_keys.append(phrase_key)

                    # Accumulate phrase scores for averaging
                    if phrase not in phrase_scores:
                        phrase_scores[phrase] = []
                    phrase_scores[phrase].append(np.float64(fact_score))

            # Batch fetch chunk mappings for valid phrases
            if valid_phrase_keys:
                chunk_tasks = [
                    self._db.get_ent_node_to_chunk_ids(index_name, phrase_key) for phrase_key in valid_phrase_keys
                ]
                chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)

                # Update weights based on chunk counts
                for phrase_key, chunks_result in zip(valid_phrase_keys, chunk_results):
                    if not isinstance(chunks_result, BaseException) and chunks_result:
                        chunks = chunks_result or set()
                        if len(chunks) > 0:
                            phrase_weights[phrase_key] /= len(chunks)

            # Calculate average fact scores efficiently using numpy
            for phrase, scores in phrase_scores.items():
                if scores:
                    linking_score_map[phrase] = float(np.mean(scores))

            # Apply top-k filtering if specified
            if link_top_k:
                phrase_weights, linking_score_map = await self.get_top_k_weights(
                    index_name, link_top_k, phrase_weights, linking_score_map
                )

            # Optimize passage retrieval and weight calculation
            passages = await self.dense_passage_retrieval(index_name, query, query_to_embedding)

            # Use numpy for efficient score normalization
            dpr_scores = np.array([doc[1] for doc in passages], dtype=float)

            # Batch process passage weights
            for idx, (doc, score) in enumerate(passages):
                passage_node_key = doc.uid
                normalized_score = dpr_scores[idx] * passage_node_weight
                passage_weights[passage_node_key] = normalized_score
                linking_score_map[doc.content] = normalized_score

            # Combine weights efficiently
            phrase_weights.update(passage_weights)
            node_weights: Dict[str, np.float64] = phrase_weights.copy()

            # Optimize linking_score_map truncation
            if len(linking_score_map) > 30:
                linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:30])

            # Validate weights before PPR
            total_weight = sum(node_weights.values())
            if total_weight <= 0.0:
                raise AssertionError(f"No phrases found in the graph for the given facts: {top_k_facts}")

            # Run optimized PPR algorithm
            ppr_sorted_doc_scores = await self.run_ppr(
                index_name, {k: float(v) for k, v in node_weights.items()}, damping=damping, top_k=top_k
            )

            logger.debug(f"Graph search completed. PPR scores: {len(ppr_sorted_doc_scores)} results")
            return ppr_sorted_doc_scores

        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            raise RuntimeError(f"Graph search with fact entities failed: {e}") from e

    async def get_top_k_weights(
        self,
        index_name: str,
        link_top_k: int,
        all_phrase_weights: Dict[str, np.float64],
        linking_score_map: Dict[str, float],
    ) -> Tuple[Dict[str, np.float64], Dict[str, float]]:
        """
        Filters the phrase weights and linking scores to retain only the top-k phrases.
        Args:
            link_top_k : int
                The number of top phrases to retain.
            all_phrase_weights : Dict[str, np.float64]
                A dictionary containing the weights of all phrases.
            linking_score_map : Dict[str, float]
                A dictionary containing the linking scores for each phrase.
        Returns:
            Tuple[Dict[str, np.float64], Dict[str, float]]
                A tuple containing two dictionaries:
                - The filtered phrase weights for the top-k phrases.
                - The linking scores for the top-k phrases.
        """
        # Step 1: 选择 top-k 短语
        linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:link_top_k])

        # Step 2: 生成 top-k 短语的 ID（如 "entity-<md5>"）
        top_k_phrases = set(linking_score_map.keys())
        top_k_phrase_ids = set(
            [compute_mdhash_id(index_name, content=phrase, prefix=self._entity_prefix) for phrase in top_k_phrases]
        )

        # Step 3: 查询图中实际存在的短语 ID
        top_k_nodes = self._graph_vertices_to_nodes(
            await self._graph_store.aselect_vertices(index_name, dict(name_in=list(top_k_phrase_ids)))
        )
        top_k_phrase_ids_in_graph = [node["name"] for node in top_k_nodes]

        # Step 4: 清除未选中短语的权重
        for phrase_id in all_phrase_weights:
            if phrase_id not in top_k_phrase_ids_in_graph:
                all_phrase_weights[phrase_id] = np.float64(0.0)
        # Step 5: 验证过滤后非零权重数量是否等于 link_top_k
        logger.debug(f"Filtered phrase weights: {all_phrase_weights}")
        logger.debug(f"Linking score map: {linking_score_map}")
        assert sum(1 for w in all_phrase_weights.values() if w != 0.0) == len(linking_score_map), (
            "Filtered phrase weights do not match the number of top-k phrases."
        )

        return all_phrase_weights, linking_score_map

    async def retrieve(
        self, index_name: str, queries: List[str], retrieve_top_k=10, lang="en", filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[RetrieveResultItem]]:
        """
        Retrieves documents based on the provided queries using a combination of dense
        passage retrieval and graph search.
        Args:
            queries : List[str]
                A list of query strings for which documents need to be retrieved.
            retrieve_top_k : int, optional
                The number of top documents to retrieve. Defaults to 10.
            lang : str, optional
                The language of the queries. Defaults to "en".
            filter : Optional[Dict[str, Any]], optional
        Returns:
            List[RetrieveResultItem]
                A list of RetrieveResultItem objects containing the retrieved documents and their corresponding scores.
        """
        query_to_embedding = await self._get_query_embeddings(queries)
        top_k_docs: List[RetrieveResultItem] = []
        query_to_docs: Dict[str, List[RetrieveResultItem]] = {}
        for q_idx, query in tqdm(enumerate(queries), desc="Retrieving", total=len(queries)):
            # rerank_start = time.time()
            query_fact_scores = await self._get_fact_scores(index_name, query, query_to_embedding)
            top_k_facts = await self.rerank_facts(query, query_fact_scores, lang=lang, link_top_k=self.linking_top_k)
            # rerank_end = time.time()

            # self.rerank_time += rerank_end - rerank_start

            if len(top_k_facts) == 0:
                logger.info("No facts found after reranking, return DPR results")
                sorted_doc_scores = await self.dense_passage_retrieval(
                    index_name, query, query_to_embedding, top_k=retrieve_top_k
                )
                top_k_docs.extend([RetrieveResultItem(document=d, score=s, query=query) for d, s in sorted_doc_scores])
            else:
                sorted_doc_key_scores = await self.graph_search_with_fact_entities(
                    index_name=index_name,
                    query=query,
                    query_to_embedding=query_to_embedding,
                    link_top_k=self.linking_top_k,
                    top_k_facts=top_k_facts,
                    passage_node_weight=self.passage_node_weight,
                    top_k=retrieve_top_k,
                )
                docs_ids = list(sorted_doc_key_scores.keys())
                docs = await self._embedd_store.get_by_ids(index_name, docs_ids)
                top_k_docs.extend(
                    [
                        RetrieveResultItem(document=doc, score=sorted_doc_key_scores[doc.uid], query=query, metadata={})
                        for doc in docs
                    ]
                )
            query_to_docs[query] = sorted(top_k_docs, key=lambda x: x.score, reverse=True)[:retrieve_top_k]
        return query_to_docs

    def get_proc_triples_to_docs(self, all_openie_info: List[OpenIEInfo]) -> Dict[str, Set[str]]:
        """
        Processes the extracted triples from OpenIE information and maps them to their corresponding document IDs.

        Args:
            all_openie_info : List[OpenIEInfo]
                A list of OpenIEInfo objects containing extracted triples and their corresponding document IDs.
        Returns:
            Dict[str, Set[str]]
                A dictionary mapping processed triples (as strings) to sets of document IDs.
        """
        proc_triples_to_docs: Dict[str, Set[str]] = {}

        for doc in all_openie_info:
            triples = self._flatten_facts([[tuple(t) for t in doc.extracted_triples]])
            for triple in triples:
                if len(triple) == 3:
                    # proc_triple = tuple(text_processing(list(triple)))
                    proc_triples_to_docs[str(triple)] = proc_triples_to_docs.get(str(triple), set()).union(
                        set([doc.idx])
                    )
        return proc_triples_to_docs

    async def delete(self, index_name, docs_to_delete: List[str] | str):
        """
        Deletes documents and their associated triples from the database, embedding store, and graph store.
        Args:
            docs_to_delete : List[str]
                A list of document IDs to be deleted from the database, embedding store, and graph store.
        Returns:
            None
        """
        triples_to_delete: List[str] = []
        entities_to_delete: List[str] = []
        docs_ids_to_triples = []
        if isinstance(docs_to_delete, str):
            docs_to_delete = [docs_to_delete]

        openie_infos = await self._db.get_openie_info(index_name, docs_to_delete)
        docs_to_delete = [doc.idx for doc in openie_infos]
        chunk_ids_triple_to_delete: Dict[str, List[Tuple[str, str, str]]] = {}
        for doc in openie_infos:
            chunk_ids_triple_to_delete[doc.idx] = [tuple(t) for t in doc.extracted_triples]
            for triple in doc.extracted_triples:
                triple_tuple = tuple(triple)
                docs_ids = set(await self._db.get_docs_from_triples(index_name, triple_tuple) or [])
                if not docs_ids:
                    continue
                docs_ids_to_triples.extend(list(docs_ids))
                # Only delete triples that are exclusively used by documents being deleted
                remaining_docs = docs_ids.difference(set(docs_to_delete))
                if not remaining_docs:
                    entities, _ = self._extract_entity_nodes([triple_tuple])
                    # Process entities for potential deletion
                    for entity in entities:
                        entity_id = compute_mdhash_id(index_name, content=entity, prefix=self._entity_prefix)
                        entity_docs = set(await self._db.get_ent_node_to_chunk_ids(index_name, entity_id) or [])
                        # Only delete entities that are exclusively used by documents being deleted
                        if entity_docs and not entity_docs.difference(set(docs_to_delete)):
                            entities_to_delete.append(entity_id)
                    # Mark triple for deletion
                    triples_to_delete.append(
                        compute_mdhash_id(index_name, content=str(triple_tuple), prefix=self._fact_prefix)
                    )
        embedding_ids_to_delete = triples_to_delete + entities_to_delete + list(chunk_ids_triple_to_delete.keys())
        logger.warning(f"Delete {len(embedding_ids_to_delete)} embeddings: {embedding_ids_to_delete}")
        if len(embedding_ids_to_delete) > 0:
            await self._embedd_store.delete(index_name, embedding_ids_to_delete)
        graph_vertex_ids_to_delete = entities_to_delete + docs_to_delete
        logger.warning(f"Delete {len(graph_vertex_ids_to_delete)} graph vertices: {graph_vertex_ids_to_delete}")
        if len(graph_vertex_ids_to_delete) > 0:
            await self._graph_store.adelete_vertices(index_name, graph_vertex_ids_to_delete)
        if len(docs_to_delete) > 0:
            await self._db.delete_openie_info(index_name, docs_to_delete)
        if len(entities_to_delete) > 0:
            await self._db.delete_ent_node_to_chunk_ids(index_name, entities_to_delete)
        if len(docs_ids_to_triples) > 0:
            await self._db.delete_nodes_cache(index_name, graph_vertex_ids_to_delete)
            await self._db.delete_triples_to_docs(index_name, triples_to_delete)

    async def run_ppr(
        self, index_name: str, node_weights: Dict[str, float], damping: float = 0.5, top_k: int = 10
    ) -> Dict[str, float]:
        """
        Computes personalized PageRank scores for the given nodes in the graph.
        Args:
            node_weights : Dict[str, float]
                A dictionary mapping node IDs to their respective weights.
            damping : float, optional
                Damping factor for the PageRank algorithm. Defaults to 0.5.
            top_k : int, optional
                The number of top nodes to retrieve. Defaults to 10.
        Returns:
            Dict[str, float]
                A dictionary containing the personalized PageRank scores for the nodes.
        """

        if damping is None:
            damping = 0.5  # for potential compatibilityp

        pagerank_scores = await self._graph_store.apersonalized_pagerank(
            label=index_name,
            vertices_with_weight=node_weights,
            damping=damping,
            top_k=top_k * 2,  # Ensure top_k is passed correctly here
        )
        # filter doc- prefix
        logger.debug(f"Pagerank scores: {pagerank_scores}")
        filtered_dict = {k: v for k, v in pagerank_scores.items() if k.startswith(self._doc_prefix)}
        return filtered_dict

    async def _augment_graph(
        self,
        index_name: str,
        entities: List[Document],
        passages: List[Document],
        node_to_node_stats: Dict[Tuple[str, str], float],
    ):
        """
        Provides utility functions to augment a graph by adding new nodes and edges.
        It ensures that the graph structure is extended to include additional components,
        and logs the completion status along with printing the updated graph information.
        """

        await self._add_new_nodes(index_name, entities, passages)
        await self._graph_store.aupsert_edges(index_name, self._edge_to_graph_edge(index_name, node_to_node_stats))

        logger.info("Graph construction completed!")

    def _save(self):
        logger.info("Saving graph completed!")

    async def init(self):
        """
        Initializes the graph and embedding stores.
        This method is responsible for setting up the necessary components
        required for the graph and embedding functionalities.
        It initializes the embedder, database, embedding store, and graph store.
        This method should be called before using the graph and embedding functionalities.
        It ensures that all necessary components are properly initialized and ready for use.
        """
        self._embedder.init()
        logger.debug(f"Initializing graph and embedding stores:{self._db}...")
        await self._db.init()
        await self._embedd_store.init()
        await self._graph_store.init()

    def _flatten_facts(self, chunk_triples: List[List[Tuple[str, str, str]]]) -> List[Tuple[str, ...]]:
        """
        Flattens a list of lists of triples into a single list of unique triples.
        This method ensures that each triple is unique and represented as a tuple.
        Args:
            chunk_triples (List[List[Tuple[str, str, str]]]): A list of lists of triples.
        Returns:
            List[Tuple[str, ...]]: A list of unique triples represented as tuples.
        """
        seen = set()
        graph_triples = []

        for triples in chunk_triples:
            for t in triples:
                t_tuple = tuple(t)  # 确保三元组为元组类型
                if t_tuple not in seen:
                    seen.add(t_tuple)
                    graph_triples.append(t_tuple)

        return graph_triples

    def _extract_entity_nodes(self, triples: List[Tuple[str, str, str]]) -> Tuple[List[str], List[List[str]]]:
        """
        Extracts unique entity nodes from a list of triples.

        Args:
            triples (List[Tuple[str, str, str]]): A list of lists of triples.
        Returns:
            Tuple[List[str], List[List[str]]]: A tuple containing a list of unique graph nodes and
            a list of lists of unique entities from each chunk's triples.
        """
        triple_entities = []  # a list of lists of unique entities from each chunk's triples
        for triple in triples:
            triple_entities_set = set()
            if len(triple) == 3:
                triple_entities_set.update([triple[0], triple[2]])
            else:
                logger.warning(f"During graph construction, invalid triple is found: {triple}")
            triple_entities.append(list(triple_entities_set))
        graph_nodes = [str(ent) for ent in np.unique([ent for ents in triple_entities for ent in ents])]
        return graph_nodes, triple_entities

    async def add_embeddings(
        self, index_name: str, entity_nodes: List[str], prefix: str, metadata: Optional[Dict[str, str]] = None
    ) -> List[Document]:
        """
        Adds embeddings for the given entity nodes to the embedding store with batch optimization.

        This method computes the hash IDs for the entity nodes, checks if they already exist in the
        embedding store, and if not, encodes the nodes in batches and adds them to the store.

        Args:
            index_name: The name of the index.
            entity_nodes: A list of entity nodes to be added to the embedding store.
            prefix: The prefix used for computing hash IDs for the entity nodes.
            metadata: Additional metadata to be associated with the documents. Defaults to None.

        Returns:
            List[Document]: A list of Document objects representing the added embeddings.

        Raises:
            RuntimeError: If embedding processing fails.
        """
        if not entity_nodes:
            return []

        try:
            # Prepare node mappings with batch optimization
            nodes_dict = {}
            for node in entity_nodes:
                hash_id = compute_mdhash_id(index_name, str(node), f"{prefix}-")
                nodes_dict[hash_id] = {"content": str(node)}

            all_hash_ids = list(nodes_dict.keys())
            logger.debug(f"Processing {len(entity_nodes)} embeddings with metadata: {metadata}")

            if not all_hash_ids:
                return []

            # Batch check for existing embeddings
            existing = await self._embedd_store.get_by_ids(index_name, all_hash_ids)
            existing_ids = {doc.uid for doc in existing}
            missing_ids = [hash_id for hash_id in all_hash_ids if hash_id not in existing_ids]

            if not missing_ids:
                logger.debug("All embeddings already exist, skipping encoding")
                return existing

            # Prepare documents for batch encoding
            texts_to_encode = [nodes_dict[hash_id]["content"] for hash_id in missing_ids]
            docs = [
                Document(uid=hash_id, content=text, metadata=metadata)
                for hash_id, text in zip(missing_ids, texts_to_encode)
            ]

            logger.debug(f"Adding {len(docs)} new embeddings to the store")

            # Batch process embeddings in chunks to manage memory
            batch_size = self._embedding_batch_size
            processed_docs = []

            for i in range(0, len(docs), batch_size):
                batch_docs = docs[i : i + batch_size]
                await self._embedd_store.add_documents(index_name, batch_docs)
                processed_docs.extend(batch_docs)

                # Small delay between batches for resource management
                if i + batch_size < len(docs):
                    await asyncio.sleep(0.05)

            # Retrieve all documents (new + existing) with their embeddings
            new_docs = await self._embedd_store.get_by_ids(index_name, missing_ids)
            all_docs = new_docs + existing

            # Update document embeddings efficiently
            docs_map = {doc.uid: doc for doc in all_docs}
            for doc in docs:
                if doc.uid in docs_map:
                    doc.embedding = docs_map[doc.uid].embedding

            return processed_docs

        except Exception as e:
            logger.error(f"Failed to add embeddings: {e}")
            raise RuntimeError(f"Embedding processing failed: {e}") from e

    def _add_fact_edges(
        self, index_name: str, new_docs: List[Document], chunk_triples: Dict[str, List[Tuple[str, str, str]]]
    ) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], float]]:
        """
        Adds edges between entity nodes and chunk nodes in the graph.

        This method is responsible for iterating through a list of chunk identifiers
        and their corresponding triple entities. It calculates and adds new edges
        between the entity nodes (defined by the computed unique hash IDs of triple entities)
        and the chunk nodes (defined by the chunk identifiers). The method also
        updates the node-to-node statistics map and keeps track of the chunk IDs
        associated with each entity node.

        Args:
            new_docs : List[Document]
                A list of Document objects representing the new documents to be added.
            chunk_triples : Dict[str, List[Tuple[str, str, str]]]
                A dictionary mapping chunk identifiers to lists of triples (subject, predicate, object).
        Returns:
            Tuple[Dict[str, List[str]], Dict[Tuple[str, str], float]]
        """
        node_to_node_stats: Dict[Tuple[str, str], float] = {}
        ent_node_to_chunk_ids: Dict[str, List[str]] = {}
        for doc in tqdm(new_docs):
            entities_in_chunk = set()
            triples = chunk_triples[doc.uid]
            chunk_key = doc.uid
            # if chunk_key not in current_graph_nodes:
            for triple in triples:
                node_key = compute_mdhash_id(index_name, content=triple[0], prefix=(self._entity_prefix))
                node_2_key = compute_mdhash_id(index_name, content=triple[2], prefix=(self._entity_prefix))
                node_to_node_stats[(node_key, node_2_key)] = node_to_node_stats.get((node_key, node_2_key), 0.0) + 1
                node_to_node_stats[(node_2_key, node_key)] = node_to_node_stats.get((node_2_key, node_key), 0.0) + 1
                entities_in_chunk.add(node_key)
                entities_in_chunk.add(node_2_key)
            for node in entities_in_chunk:
                ent_node_to_chunk_ids[node] = list(set(ent_node_to_chunk_ids.get(node, set())).union(set([chunk_key])))

        return ent_node_to_chunk_ids, node_to_node_stats

    def _add_passage_edges(
        self,
        index_name: str,
        new_docs: List[Document],
        chunk_triples: Dict[str, List[List[str]]],
        node_to_node_stats: Dict[Tuple[str, str], float],
    ) -> int:
        """
        Adds edges connecting passage nodes to phrase nodes in the graph.

        This method is responsible for iterating through a list of chunk identifiers
        and their corresponding triple entities. It calculates and adds new edges
        between the passage nodes (defined by the chunk identifiers) and the phrase
        nodes (defined by the computed unique hash IDs of triple entities). The method
        also updates the node-to-node statistics map and keeps count of newly added
        passage nodes.

        Parameters:
            chunk_ids : List[str]
                A list of identifiers representing passage nodes in the graph.
            chunk_triple_entities : List[List[str]]
                A list of lists where each sublist contains entities (strings) associated
                with the corresponding chunk in the chunk_ids list.

        Returns:
            int
                The number of new passage nodes added to the graph.
        """

        num_new_chunks = 0
        for doc in new_docs:
            chunk = chunk_triples[doc.uid]
            for chunk_ents in chunk:
                for chunk_ent in chunk_ents:
                    node_key = compute_mdhash_id(index_name, chunk_ent, prefix=self._entity_prefix)
                    node_to_node_stats[(doc.uid, node_key)] = 1.0
            num_new_chunks += 1
        return num_new_chunks

    def _add_synonymy_edges(
        self, query_docs: List[Document], target_docs: List[Document], node_to_node_stats: Dict[Tuple[str, str], float]
    ):
        """
        Adds synonymy edges between query and target documents based on their embeddings.
        This method retrieves the k-nearest neighbors for each query document and
        establishes synonymy edges based on the similarity scores.
        It also updates the node-to-node statistics map with the similarity scores.
        Args:
            query_docs : List[Document]
                A list of query documents for which synonymy edges need to be added.
            target_docs : List[Document]
                A list of target documents to establish synonymy edges with.
            node_to_node_stats : Dict[Tuple[str, str], float]
                A dictionary mapping pairs of node keys to their corresponding similarity scores.
        Returns:
            None
        """
        logger.debug(
            f"Adding synonymy edges for {len(query_docs)} query documents and {len(target_docs)} target documents."
        )
        query_node_key2knn_node_keys = self._embedd_store.knn(
            query_docs=query_docs,
            target_docs=target_docs,
            k=self.synonymy_edge_topk,
            query_batch_size=self.synonymy_edge_query_batch_size,
            key_batch_size=self.synonymy_edge_key_batch_size,
        )
        entity_id_to_row = {
            doc.uid: {"content": doc.content}  # 假设Document类有id和text属性
            for doc in itertools.chain(query_docs, target_docs)
        }
        num_synonym_triple = 0
        synonym_candidates = []  # [(node key, [(synonym node key, corresponding score), ...]), ...]

        for node_key in tqdm(query_node_key2knn_node_keys.keys(), total=len(query_node_key2knn_node_keys)):
            synonyms = []

            entity = entity_id_to_row[node_key]["content"]

            if len(re.sub("[^A-Za-z0-9]", "", entity)) > 2:
                nns = query_node_key2knn_node_keys[node_key]

                num_nns = 0
                for nn, score in zip(nns[0], nns[1]):
                    if score < self.synonymy_edge_sim_threshold or num_nns > 100:
                        break

                    nn_phrase = entity_id_to_row[nn]["content"]

                    if nn != node_key and nn_phrase != "":
                        sim_edge = (node_key, nn)
                        synonyms.append((nn, score))
                        num_synonym_triple += 1
                        node_to_node_stats[sim_edge] = score  # Need to seriously discuss on this
                        num_nns += 1
            synonym_candidates.append((node_key, synonyms))

    async def _add_new_nodes(
        self, index_name: str, entities: List[Document], passages: List[Document]
    ) -> Tuple[List[Document], List[Document]]:
        """
        Adds new nodes to the graph for the given entities and passages.
        This method checks for existing nodes in the graph and adds only the new ones.
        It also updates the node-to-rows mapping for the new nodes.
        Args:
            entities : List[Document]
                A list of entity documents to be added as nodes in the graph.
            passages : List[Document]
                A list of passage documents to be added as nodes in the graph.
        Returns:
            Tuple[List[Document], List[Document]]
                A tuple containing two lists: the first list contains the new entity documents,
                and the second list contains the new passage documents.
        """
        docs = copy.deepcopy(entities)
        docs.extend(passages)

        node_to_rows_map = {doc.uid: doc.model_dump() for doc in docs}
        node_ids = list(node_to_rows_map.keys())
        existing_nodes: List[Dict[str, Any]] = self._graph_vertices_to_nodes(
            await self._graph_store.aselect_vertices(index_name, dict(name_in=node_ids))
        )
        existing_ids = [node["name"] for node in existing_nodes]
        new_nodes: List[Dict[str, Any]] = []
        new_node_ids = set(node_ids) - set(existing_ids)
        new_pasages: List[Document] = []
        new_entities: List[Document] = []

        for node_id in new_node_ids:
            if node_id.startswith(self._doc_prefix):
                new_pasages.append(Document.model_validate(node_to_rows_map[node_id]))
            else:
                new_entities.append(Document.model_validate(node_to_rows_map[node_id]))
        for node_id in new_node_ids:
            node = node_to_rows_map[node_id]
            node["name"] = node_id
            new_nodes.append(node)
        if len(new_nodes) > 0:
            # logger.debug(f"Adding {new_nodes} new nodes to the graph.")
            # new_vertices = self._vertices_nodes_to_graph_vertices(index_name, new_nodes)
            await self._graph_store.aupsert_virtices(
                index_name, self._vertices_nodes_to_graph_vertices(index_name, new_nodes)
            )
        return new_entities, new_pasages

    def _edge_to_graph_edge(self, index_name: str, edge_nodes: Dict[Tuple[str, str], float]) -> List[GraphEdge]:
        """
        Converts an edge represented as a tuple of node keys and a weight into a dictionary format.
        Args:
            edge : Dict[Tuple[str, str], float]
                A dictionary where keys are tuples representing edges (source, target) and values are weights.
        Returns:
            Dict[str, Any]
                A dictionary representation of the edge with its properties.
        """
        edges = []
        for edge, weight in edge_nodes.items():
            edges.append(
                GraphEdge(
                    source=edge[0],
                    target=edge[1],
                    relation="",
                    uid=GraphHelper.generate_edge_id(index_name, edge[0], "", edge[1]),
                    weight=weight,
                    relation_type=RelationType.RELATION,
                    label=index_name,
                    metadata={},
                )
            )
        return edges

    def _vertices_nodes_to_graph_vertices(self, index_name: str, vertices: List[Dict[str, Any]]) -> List[GraphVertex]:
        """
        Converts a list of vertices (dictionaries) to a list of GraphVertex objects.
        Args:
            vertices : List[Dict[str, Any]]
                A list of dictionaries representing the vertices in the graph.
        Returns:
            List[GraphVertex]
                A list of GraphVertex objects created from the input vertices.
        """
        graph_vertices = []
        for vertex in vertices:
            v = GraphVertex(
                label=index_name,
                uid=vertex["uid"],
                name=vertex["name"],
                namespace=vertex.get("metadata", {}).get("namespace", ""),
                content=str(vertex["content"]) if not isinstance(vertex["content"], str) else vertex["content"],
                metadata=vertex.get("metadata", {}),
                embedding=vertex.get("embedding", None),
                doc_id=vertex.get("metadata", {}).get("doc_id", []),
            )
            graph_vertices.append(v)
        return graph_vertices

    def _graph_vertices_to_nodes(self, vertices: List[GraphVertex]) -> List[Dict[str, Any]]:
        """
        Converts a list of GraphVertex objects to a list of dictionaries representing the nodes in the graph.
        Args:
            vertices : List[GraphVertex]
                A list of GraphVertex objects to be converted.
        Returns:
            List[Dict[str, Any]]
                A list of dictionaries representing the nodes in the graph.
        """
        nodes: List[Dict[str, Any]] = []
        if not vertices:
            return nodes
        for vertex in vertices:
            node = {
                "uid": vertex.uid,
                "name": vertex.name,
                "content": str(vertex.content) if not isinstance(vertex.content, str) else vertex.content,
                "embedding": vertex.embedding,
                "metadata": vertex.metadata,
                "description": vertex.description or {},
            }
            if vertex.namespace:
                node["metadata"]["namespace"] = vertex.namespace
            nodes.append(node)
        return nodes

    async def embedding_retrieve(
        self,
        unique_name: str,
        queries: List[str],
        filter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[RetrieveResultItem]]:
        """
        Retrieves documents based on the provided queries using the embedding store.

        Args:
            unique_name : str
                The unique name for the index to be used for retrieval.
            queries : List[str]
                A list of query strings for which documents need to be retrieved.
            filter : Optional[Dict[str, Any]], optional
                Additional filter criteria for the retrieval. Defaults to None.

        Returns:
            Dict[str, List[RetrieveResultItem]]
                A dictionary mapping each query to a list of retrieved Document objects.
        """
        if filter:
            filter["namespace"] = Namespace.PASSAGE.value
        tasks = [
            self._embedd_store.asimilar_search_with_scores(
                index_name=unique_name,
                query=query,
                k=100,
                filter=filter,
            )
            for query in queries
        ]
        results: List[List[Tuple[Document, float]]] = await asyncio.gather(*tasks)
        query_to_docs: Dict[str, List[Document]] = {}
        for index, r in enumerate(results):
            query_to_docs[queries[index]] = [
                RetrieveResultItem(document=doc, score=score, query=queries[index], metadata={})
                for doc, score in r
                if doc.metadata.get("namespace") == Namespace.PASSAGE.value
            ]
        return query_to_docs

    async def qa(
        self,
        unique_name: str,
        query: str,
        history_context: Optional[str] = "",
        top_k: int = 5,
        similarity_threshold=0.7,
        prompt: Optional[str] = None,
        llm: Optional[BaseChat] = None,
        reranker: Optional[BaseRerank] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[QAChatResponse]:
        """
        Performs a question-answering operation on the specified index using the provided query.

        Args:
            unique_name : str
                The unique name for the index to be used for question answering.
            query : str
                The query string for which the answer needs to be retrieved.
            context : Optional[str], optional,The context to be used for the question answering. Defaults to "".
            top_k : int, optional
                The number of top documents to retrieve. Defaults to 5.
            prompt : Optional[str], optional,
            llm : Optional[BaseChat], optional
                The language model to be used for generating the answer. If not provided, the default LLM will be used.

        Returns:
            AsyncIterator[QAChatResponse]
                An asynchronous iterator yielding QAChatResponse objects containing the answers and related information.
        """
        lang = detect_language(query)
        reranker = reranker or self._reranker
        res = await self.hybrid_retrieve(unique_name, [query], reranker, top_k, similarity_threshold, filter=filter)
        for _, items in res.items():
            if not items:
                yield QAChatResponse(query=query, answer="", sources=[], metadata={})
                continue
            qa_prompt = self._qa_prompt_builder.build(items, query, lang, history_context)

            from langchain.prompts import ChatPromptTemplate

            should_contains_vars = ["question", "context"]
            if prompt is not None:
                cpt = ChatPromptTemplate.from_template(prompt)
                input_variables = cpt.input_variables
                missing = set(should_contains_vars) - set(input_variables)
                if missing:
                    raise ValueError(f"Prompt is missing required variables: {missing}")
                qa_prompt = cpt.invoke(
                    {
                        "question": query,
                        "context": "\n".join([item.document.content for item in items])
                        + "\n\n"
                        + "History context: \n"
                        + (history_context or ""),
                    }
                ).to_string()
            llm = llm or self._llm
            async for resp in self._llm.astream(ChatPromptTemplate.from_template(qa_prompt), {}):
                ans: ChatResponse = resp
                logger.debug(f"Answer: {ans.content}, Tokens: {ans.tokens}")
                yield QAChatResponse(
                    question=query,
                    answer=ans.content,
                    tokens=ans.tokens,
                    context=json.dumps([item.document.model_dump() for item in items], ensure_ascii=False),
                )

    async def _ensure_resource_cleanup(self) -> None:
        """
        Ensures proper cleanup of resources and connections.

        This method should be called during shutdown or error recovery
        to prevent resource leaks and ensure system stability.
        """
        try:
            # Close any open connections or resources
            if hasattr(self._embedd_store, "close"):
                await self._embedd_store.close()
            if hasattr(self._graph_store, "close"):
                await self._graph_store.close()
            if hasattr(self._db, "close"):
                await self._db.close()
            logger.info("Resource cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error during resource cleanup: {e}")

    async def _validate_system_health(self, index_name: str) -> bool:
        """
        Validates the health of the system components.

        Args:
            index_name: The index name to validate.

        Returns:
            bool: True if all components are healthy, False otherwise.
        """
        try:
            # Check database connectivity
            await self._db.init()

            # Check embedding store
            await self._embedd_store.init()

            # Check graph store
            await self._graph_store.init()

            # Validate index exists
            # This is a basic health check - could be expanded
            logger.debug(f"System health check passed for index: {index_name}")
            return True

        except Exception as e:
            logger.error(f"System health check failed: {e}")
            return False

    @property
    def lock(self) -> Lock:
        """
        Returns the async lock for thread-safe operations.

        This property provides access to the internal lock for operations
        that require thread safety or coordination between concurrent tasks.

        Returns:
            Lock: The asyncio Lock instance for synchronization.
        """
        return self._lock

    async def _batch_process_entities(
        self, index_name: str, entities_data: List[Tuple[str, str]], batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process entities in batches for better performance and memory management.

        Args:
            index_name: The index name for the entities.
            entities_data: List of tuples containing (entity_text, entity_type).
            batch_size: Optional batch size override. If None, uses default.

        Returns:
            Dict[str, Any]: Results of batch processing including success counts and errors.
        """
        batch_size = batch_size or self._graph_batch_size
        total_entities = len(entities_data)
        processed_count = 0
        error_count = 0
        results: Dict[str, List[Any]] = {"processed": [], "errors": []}

        try:
            for i in range(0, total_entities, batch_size):
                batch = entities_data[i : i + batch_size]

                try:
                    # Process batch of entities
                    batch_results = []
                    for entity_text, entity_type in batch:
                        entity_id = compute_mdhash_id(index_name, entity_text, self._entity_prefix)
                        batch_results.append({"id": entity_id, "text": entity_text, "type": entity_type})

                    results["processed"].extend(batch_results)
                    processed_count += len(batch)

                    # Small delay between batches for resource management
                    if i + batch_size < total_entities:
                        await asyncio.sleep(0.02)

                except Exception as e:
                    logger.error(f"Error processing entity batch {i // batch_size + 1}: {e}")
                    results["errors"].append(f"Batch {i // batch_size + 1}: {e}")
                    error_count += len(batch)

            logger.info(f"Entity batch processing completed: {processed_count} processed, {error_count} errors")
            return results

        except Exception as e:
            logger.error(f"Critical error in batch entity processing: {e}")
            raise RuntimeError(f"Batch entity processing failed: {e}") from e

    # def _optimize_memory_usage(self, data_structures: Dict[str, Any]) -> Dict[str, Any]:
    #     """
    #     Optimize memory usage of large data structures.

    #     This method applies various optimization techniques to reduce memory footprint:
    #     - Converting lists to more memory-efficient structures where appropriate
    #     - Deduplicating data
    #     - Compressing sparse data structures

    #     Args:
    #         data_structures: Dictionary of data structures to optimize.

    #     Returns:
    #         Dict[str, Any]: Optimized data structures.
    #     """
    #     optimized = {}

    #     for key, data in data_structures.items():
    #         try:
    #             if isinstance(data, list):
    #                 # Deduplicate lists while preserving order
    #                 if data and all(isinstance(item, (str, int, float)) for item in data):
    #                     seen = set()
    #                     optimized[key] = [x for x in data if x not in seen and not seen.add(x)]
    #                 else:
    #                     optimized[key] = data

    #             elif isinstance(data, dict):
    #                 # Remove empty values from dictionaries
    #                 optimized[key] = {k: v for k, v in data.items() if v is not None and v != ""}

    #             else:
    #                 optimized[key] = data

    #         except Exception as e:
    #             logger.warning(f"Failed to optimize data structure '{key}': {e}")
    #             optimized[key] = data

    #     return optimized
