import copy
import difflib
import itertools
import json
import re
import traceback
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from deeprag_core.chat.base_chat import BaseChat
from deeprag_core.embedding import BaseEmbedding
from deeprag_core.operators.ner.ner_operator import NEROperator
from deeprag_core.operators.ner.promopts import NERPrompt
from deeprag_core.operators.openie.openie_operator import OpenIEOperator
from deeprag_core.operators.rdf.prompts import REPrompt
from deeprag_core.operators.rdf.rdf_operator import RDFOperator
from deeprag_core.rag.base_rag import BaseRAG
from deeprag_core.rag.hipporag.hipporag_storage import HipporagStorage
from deeprag_core.rag.hipporag.prompts import DSPyRerankPrompt, get_query_instruction
from deeprag_core.schemas.document import Document
from deeprag_core.schemas.hipporag_models import OpenIEInfo
from deeprag_core.schemas.openie_mdoel import OpenIEModel
from deeprag_core.schemas.rag_model import RetrieveResultItem
from deeprag_core.storage.graph import GraphStorage
from deeprag_core.storage.vector import VectorStorage
from deeprag_core.utils.misc import compute_mdhash_id
from loguru import logger
from tqdm import tqdm

from .rerank import DSPyFilter


class HippoRAGImpl(BaseRAG):
    def __init__(
        self,
        llm: BaseChat,
        embbeder: BaseEmbedding,
        embedding_store: VectorStorage,
        graph_store: GraphStorage,
        hipporag_store: HipporagStorage,
        dspy_file_path: str,
        embedding_key_prefix: str = "entity_embeddings",
        linking_top_k: int = 15,
        passage_node_weight: float = 0.05,
        graph_path: Optional[str] = None,
        synonymy_edge_topk: int = 2047,
        synonymy_edge_query_batch_size=1000,
        synonymy_edge_key_batch_size=1000,
        synonymy_edge_sim_threshold=0.8,
    ):
        self.linking_top_k = linking_top_k
        self.passage_node_weight = passage_node_weight
        self.graph_path = graph_path
        self.synonymy_edge_topk = synonymy_edge_topk
        self.synonymy_edge_query_batch_size = synonymy_edge_query_batch_size
        self.synonymy_edge_key_batch_size = synonymy_edge_key_batch_size
        self.synonymy_edge_sim_threshold = synonymy_edge_sim_threshold
        ner_prompt = NERPrompt()
        self._ner = NEROperator(ner_prompt, llm)
        rdf_prompt = REPrompt()
        self._rdf = RDFOperator(rdf_prompt, llm)
        self._openie = OpenIEOperator(self._ner, self._rdf)
        dsp_prompt = DSPyRerankPrompt(best_dspy_path=dspy_file_path)
        super().__init__()
        self._embedding_prefix = embedding_key_prefix

        self._embedder = embbeder

        self._embedd_store: VectorStorage = embedding_store
        self._graph_store: GraphStorage = graph_store
        self.rerank_filter = DSPyFilter(dsp_prompt, self._llm)
        # self.config = config
        self._db = hipporag_store

        self._ready_to_retrieve = False
        self._query_to_embedding: Dict = {"triple": {}, "passage": {}}

        self._entity_prefix = "entity"
        self._fact_prefix = "fact"
        self._doc_prefix = "doc"

    def create(self, name: str, description: str, idx: str):
        pass

    def add_documents(self, texts: List[Document], lang="en") -> List[Document]:
        """
        Adds documents to the vector store and graph store.

        Args:
            texts : List[Document]
                A list of Document objects to be added to the vector store and graph store.
            lang : str
                The language of the documents. Defaults to "en".
        """
        chunk_triples: Dict[str, List[Tuple[str, str, str]]] = {}  # a dictionary of triples for each chunk
        openie_infos: List[OpenIEInfo] = []
        entities_str = []
        facts_str = []
        for doc in texts:
            if not doc.uid:
                doc.uid = compute_mdhash_id(doc.content, prefix=self._doc_prefix)
            doc.metadata = doc.metadata or {}
            doc.metadata["namespace"] = "passage"
            doc.metadata["openie_idx"] = doc.uid
            ie: OpenIEModel = self._openie.extract(doc.content, lang=lang)
            openie_infos.append(
                OpenIEInfo(
                    idx=doc.uid,
                    passage=doc.content,
                    extracted_triples=[list(item) for item in ie.triples],
                    extracted_entities=list(set(ie.ner)),
                )
            )
            graph_nodes, _ = self._extract_entity_nodes(ie.triples)
            chunk_triples[doc.uid] = [(item[0], item[1], item[2]) for item in ie.triples if len(item) == 3]
            entities_str.extend([str(node) for node in graph_nodes])
            doc.metadata["entities"] = [compute_mdhash_id(str(node), self._entity_prefix) for node in graph_nodes]
            doc.metadata["facts"] = [compute_mdhash_id(str(fact)) for fact in ie.triples]
            facts = self._flatten_facts([ie.triples])
            facts_str.extend([str(fact) for fact in facts])

        entities_str = list(set(entities_str))
        facts_str = list(set(facts_str))
        embed_entities = self.add_embeddings(entities_str, self._entity_prefix, metadata={"namespace": "entity"})

        self.add_embeddings(facts_str, self._fact_prefix, metadata={"namespace": "fact"})

        self._db.save_openie_info(openie_infos)
        docs = self._embedd_store.add_documents_filter_exists(texts)
        _, new_docs = self._add_new_nodes(embed_entities, docs)

        ent_node_to_chunk_ids, node_to_node_stats = self._add_fact_edges(new_docs, chunk_triples)
        num_new_chunks = self._add_passage_edges(new_docs, chunk_triples, node_to_node_stats)
        logger.info(f"Added {num_new_chunks} new chunks.")
        if num_new_chunks > 0:
            entities: List[Document] = self._embedd_store.select_on_metadata({"namespace": "entity"})
            self._add_synonymy_edges(entities, entities, node_to_node_stats)
            self._augment_graph(embed_entities, texts, node_to_node_stats)
            self._save()
        for key, chunks in ent_node_to_chunk_ids.items():
            exists_ent_chunks = self._db.get_ent_node_to_chunk_ids(key) or []
            self._db.set_ent_node_to_chunk_ids(key, list(set(chunks).union(set(exists_ent_chunks))))
        for node_to_node, stats in node_to_node_stats.items():
            self._db.set_node_to_node_stats(node_to_node[0], node_to_node[1], stats)
        triples_to_docs = self.get_proc_triples_to_docs(openie_infos)
        self._db.set_triples_to_docs(triples_to_docs)
        return texts

    def _get_query_embeddings(self, queries: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
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
            query_embeddings_for_triple = self._embedder.encode(
                all_query_strings, instruction=get_query_instruction("query_to_fact")
            )
            for query, embedding in zip(all_query_strings, query_embeddings_for_triple):
                query_to_embedding["triple"][query] = embedding

            logger.info(f"Encoding {len(all_query_strings)} queries for query_to_passage.")
            query_embeddings_for_passage = self._embedder.encode(
                all_query_strings, instruction=get_query_instruction("query_to_passage")
            )
            for query, embedding in zip(all_query_strings, query_embeddings_for_passage):
                query_to_embedding["passage"][query] = embedding
            return query_to_embedding

        return query_to_embedding

    def _get_fact_scores(
        self, query: str, query_to_embedding: Optional[Dict[str, Dict[str, np.ndarray]]] = {}, top_k: int = 10
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
            query_embedding = self._embedder.encode(query, instruction=get_query_instruction("query_to_fact"))
        return self._embedd_store.similarity_search_by_vector(
            query_embedding.tolist(), k=top_k, filter={"namespace": "fact"}
        )

    def rerank_facts(
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
        logger.debug(f"rerank_facts fact_before_filter: {fact_before_filter}")
        input = {"question": query, "fact_before_filter": json.dumps(fact_before_filter, ensure_ascii=False)}

        output = self.rerank_filter.execute(input, lang=lang)
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
        logger.debug(f"rerank_facts Top {link_top_k} facts: {sorted_candidate_items}")
        # rerank_log = {"facts_before_rerank": candidate_facts, "facts_after_rerank": top_k_facts}

        return top_k_facts

    def dense_passage_retrieval(
        self, query: str, query_to_embedding: Dict[str, Dict[str, np.ndarray]], top_k: int = 20
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
        return self._embedd_store.similarity_search_by_vector(
            query_embedding.tolist(), k=top_k, filter={"namespace": "passage"}
        )

    def graph_search_with_fact_entities(
        self,
        query: str,
        query_to_embedding: Dict[str, Dict[str, np.ndarray]],
        link_top_k: int,
        top_k_facts: List[Tuple[Document, float]],
        passage_node_weight: float = 0.05,
        damping: float = 0.5,  # Damping factor for PPR algorithm
        top_k: int = 10,
    ) -> Dict[str, float]:
        """
        Performs graph search using the fact entities and passage nodes to compute personalized PageRank scores.
        Args:
            query : str
                The input query for which graph search needs to be performed.
            query_to_embedding : Dict[str, Dict[str, np.ndarray]]
                A dictionary containing pre-computed embeddings for the queries.
            link_top_k : int
                The number of top facts to consider for graph search.
            top_k_facts : List[Tuple[Document, float]]
                A list of tuples containing Document objects and their corresponding scores.
            passage_node_weight : float, optional
                The weight assigned to passage nodes. Defaults to 0.05.
            damping : float, optional
                Damping factor for the PageRank algorithm. Defaults to 0.5.
            top_k : int, optional
                The number of top nodes to retrieve. Defaults to 10.
        Returns:
            Dict[str, float]
                A dictionary containing the personalized PageRank scores for the nodes.
        """
        linking_score_map: Dict[
            str, float
        ] = {}  # from phrase to the average scores of the facts that contain the phrase
        phrase_scores: Dict[
            str, List[np.float64]
        ] = {}  # store all fact scores for each phrase regardless of whether they exist in the knowledge graph or not
        phrase_weights: Dict[str, np.float64] = {}
        passage_weights: Dict[str, np.float64] = {}
        phrase_keys: List[str] = []
        for doc, fact_score in top_k_facts:
            f = eval(doc.content)
            subject_phrase = f[0].lower()
            # predicate_phrase = f[1].lower()
            object_phrase = f[2].lower()
            phrase_keys.append(compute_mdhash_id(content=subject_phrase, prefix=self._entity_prefix))
            phrase_keys.append(compute_mdhash_id(content=object_phrase, prefix=self._entity_prefix))
        phrase_keys = list(set(phrase_keys))
        nodes = self._graph_store.select_vertices(dict(name_in=phrase_keys))
        node_id_to_phrase = {node["name"]: node for node in nodes}

        for doc, fact_score in top_k_facts:
            f = eval(doc.content)
            subject_phrase = f[0].lower()
            object_phrase = f[2].lower()

            for phrase in [subject_phrase, object_phrase]:
                phrase_key = compute_mdhash_id(content=phrase, prefix=self._entity_prefix)

                if phrase_key in node_id_to_phrase:
                    phrase_weights[phrase_key] = np.float64(fact_score)
                    chunks = self._db.get_ent_node_to_chunk_ids(phrase_key) or set()
                    if len(chunks) > 0:
                        phrase_weights[phrase_key] /= len(chunks)

                if phrase not in phrase_scores:
                    phrase_scores[phrase] = []
                phrase_scores[phrase].append(np.float64(fact_score))

        # calculate average fact score for each phrase
        for phrase, scores in phrase_scores.items():
            linking_score_map[phrase] = float(np.mean(scores))

        if link_top_k:
            phrase_weights, linking_score_map = self.get_top_k_weights(link_top_k, phrase_weights, linking_score_map)
        logger.debug(f"linking_score_map: {linking_score_map}, phrase_weights: {phrase_weights}")
        passages = self.dense_passage_retrieval(query, query_to_embedding)
        logger.debug(f"dense_passage_retrieval passages: {len(passages)}:{passages}")
        sorted_passages = sorted(passages, key=lambda x: x[1], reverse=True)
        dpr_doc_scores = [doc[1] for doc in sorted_passages]
        dpr_sorted_docs = [(doc[0].uid, doc[1]) for doc in sorted_passages]
        dpr_sorted_doc_scores = np.array(dpr_doc_scores, dtype=float)
        normalized_dpr_sorted_scores = dpr_sorted_doc_scores
        logger.debug(f"normalized_dpr_sorted_scores:{dpr_sorted_doc_scores}: {normalized_dpr_sorted_scores}")
        for index, doc_score in enumerate(dpr_sorted_docs):
            passage_dpr_score = normalized_dpr_sorted_scores[index]
            passage_node_key = doc_score[0]
            passage_weights[passage_node_key] = passage_dpr_score * passage_node_weight
            passage_node_text = self._embedd_store.get(passage_node_key).content
            linking_score_map[passage_node_text] = passage_dpr_score * passage_node_weight

        phrase_weights.update(passage_weights)

        node_weights: Dict[str, np.float64] = copy.deepcopy(phrase_weights)

        # Recording top 30 facts in linking_score_map
        if len(linking_score_map) > 30:
            linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:30])

        assert sum(node_weights.values()) > 0.0, f"No phrases found in the graph for the given facts: {top_k_facts}"

        # Running PPR algorithm based on the passage and phrase weights previously assigned
        logger.debug(f"graph_search_with_fact_entities node_weights: {node_weights}: {len(node_weights)}")
        ppr_sorted_doc_scores = self.run_ppr(
            {k: v.astype(float) for k, v in node_weights.items()}, damping=damping, top_k=top_k
        )

        return ppr_sorted_doc_scores

    def get_top_k_weights(
        self, link_top_k: int, all_phrase_weights: Dict[str, np.float64], linking_score_map: Dict[str, float]
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
            [compute_mdhash_id(content=phrase, prefix=self._entity_prefix) for phrase in top_k_phrases]
        )

        # Step 3: 查询图中实际存在的短语 ID
        top_k_nodes = self._graph_store.select_vertices(dict(name_in=list(top_k_phrase_ids)))
        top_k_phrase_ids_in_graph = [node["name"] for node in top_k_nodes]

        # Step 4: 清除未选中短语的权重
        for phrase_id in all_phrase_weights:
            if phrase_id not in top_k_phrase_ids_in_graph:
                all_phrase_weights[phrase_id] = np.float64(0.0)

        # Step 5: 验证过滤后非零权重数量是否等于 link_top_k
        assert sum(1 for w in all_phrase_weights.values() if w != 0.0) == len(linking_score_map), (
            "过滤后非零权重数量与 link_top_k 不一致"
        )

        return all_phrase_weights, linking_score_map

    def retrieve(self, queries: List[str], retrieve_top_k=10, lang="en") -> List[RetrieveResultItem]:
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
        Returns:
            List[RetrieveResultItem]
                A list of RetrieveResultItem objects containing the retrieved documents and their corresponding scores.
        """
        query_to_embedding = self._get_query_embeddings(queries)
        top_k_docs: List[RetrieveResultItem] = []
        for q_idx, query in tqdm(enumerate(queries), desc="Retrieving", total=len(queries)):
            # rerank_start = time.time()
            query_fact_scores = self._get_fact_scores(query, query_to_embedding)
            top_k_facts = self.rerank_facts(query, query_fact_scores, lang=lang, link_top_k=self.linking_top_k)
            # rerank_end = time.time()

            # self.rerank_time += rerank_end - rerank_start

            if len(top_k_facts) == 0:
                logger.info("No facts found after reranking, return DPR results")
                sorted_doc_scores = self.dense_passage_retrieval(query, query_to_embedding, top_k=retrieve_top_k)
                top_k_docs.extend([RetrieveResultItem(document=d, score=s, query=query) for d, s in sorted_doc_scores])
            else:
                sorted_doc_key_scores = self.graph_search_with_fact_entities(
                    query=query,
                    query_to_embedding=query_to_embedding,
                    link_top_k=self.linking_top_k,
                    top_k_facts=top_k_facts,
                    passage_node_weight=self.passage_node_weight,
                    top_k=retrieve_top_k,
                )
                docs_ids = list(sorted_doc_key_scores.keys())
                docs = self._embedd_store.get_by_ids(docs_ids)
                top_k_docs.extend(
                    [
                        RetrieveResultItem(document=doc, score=sorted_doc_key_scores[doc.uid], query=query, metadata={})
                        for doc in docs
                    ]
                )
        return sorted(top_k_docs, key=lambda x: x.score, reverse=True)

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
                        set([doc["idx"]])
                    )
        return proc_triples_to_docs

    def delete(self, docs_to_delete: List[str]):
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
        openie_infos = self._db.get_openie_info(docs_to_delete)
        docs_to_delete = [doc.idx for doc in openie_infos]
        chunk_ids_triple_to_delete: Dict[str, List[Tuple[str, str, str]]] = {}
        for doc in openie_infos:
            chunk_ids_triple_to_delete[doc.idx] = [tuple(t) for t in doc.extracted_triples]
            for triple in doc.extracted_triples:
                triple_tuple = tuple(triple)
                docs_ids = set(self._db.get_docs_from_triples(triple_tuple) or [])
                if not docs_ids:
                    continue
                docs_ids_to_triples.extend(list(docs_ids))
                # Only delete triples that are exclusively used by documents being deleted
                remaining_docs = docs_ids.difference(set(docs_to_delete))
                if not remaining_docs:
                    entities, _ = self._extract_entity_nodes([triple_tuple])
                    # Process entities for potential deletion
                    for entity in entities:
                        entity_id = compute_mdhash_id(content=entity, prefix=self._entity_prefix)
                        entity_docs = set(self._db.get_ent_node_to_chunk_ids(entity_id) or [])
                        # Only delete entities that are exclusively used by documents being deleted
                        if entity_docs and not entity_docs.difference(set(docs_to_delete)):
                            entities_to_delete.append(entity_id)
                    # Mark triple for deletion
                    triples_to_delete.append(compute_mdhash_id(content=str(triple_tuple), prefix=self._fact_prefix))
        embedding_ids_to_delete = triples_to_delete + entities_to_delete + list(chunk_ids_triple_to_delete.keys())
        self._embedd_store.delete(embedding_ids_to_delete)
        graph_vertex_ids_to_delete = entities_to_delete + entities_to_delete
        self._graph_store.delete_vertices(graph_vertex_ids_to_delete)
        self._db.delete_openie_info(docs_to_delete)
        self._db.delete_ent_node_to_chunk_ids(entities_to_delete)
        self._db.delete_node_to_node_stats(triples_to_delete)
        self._db.delete_triples_to_docs(triples_to_delete)

    def run_ppr(self, node_weights: Dict[str, float], damping: float = 0.5, top_k: int = 10) -> Dict[str, float]:
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

        pagerank_scores = self._graph_store.personalized_pagerank(
            vertices_with_weight=node_weights,
            damping=damping,
            top_k=top_k,  # Ensure top_k is passed correctly here
        )
        # filter doc- prefix
        filtered_dict = {k: v for k, v in pagerank_scores.items() if k.startswith(self._doc_prefix)}
        return filtered_dict

    def _augment_graph(
        self, entities: List[Document], passages: List[Document], node_to_node_stats: Dict[Tuple[str, str], float]
    ):
        """
        Provides utility functions to augment a graph by adding new nodes and edges.
        It ensures that the graph structure is extended to include additional components,
        and logs the completion status along with printing the updated graph information.
        """

        self._add_new_nodes(entities, passages)
        self._graph_store.add_new_edges(node_to_node_stats)

        logger.info("Graph construction completed!")

    def _save(self):
        from deeprag_core.storage.igraph_store import IGraphStore

        if isinstance(self._graph_store, IGraphStore):
            self._graph_store.save(self.graph_path)
        logger.info("Saving graph completed!")

    def init(self):
        """
        Initializes the graph and embedding stores.
        This method is responsible for setting up the necessary components
        required for the graph and embedding functionalities.
        It initializes the embedder, database, embedding store, and graph store.
        This method should be called before using the graph and embedding functionalities.
        It ensures that all necessary components are properly initialized and ready for use.
        """
        self._embedder.init()
        self._db.init()
        self._embedd_store.init()
        self._graph_store.init()

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

    def add_embeddings(
        self, entity_nodes: List[str], prefix: str, metadata: Optional[Dict[str, str]] = None
    ) -> List[Document]:
        """
        Adds embeddings for the given entity nodes to the embedding store.

        This method computes the hash IDs for the entity nodes, checks if they already exist in the
        embedding store, and if not, encodes the nodes and adds them to the store.

        Args:
            entity_nodes : List[str]
                A list of entity nodes to be added to the embedding store.
            prefix : str
                The prefix used for computing hash IDs for the entity nodes.
            metadata : Optional[Dict[str, str]], optional
                Additional metadata to be associated with the documents. Defaults to None.
        Returns:
            List[Document]
                A list of Document objects representing the added embeddings.
        """
        # graph_nodes, triple_entities = self.extract_graph_nodes(triples)
        nodes_dict = {}
        for node in entity_nodes:
            hash_id = compute_mdhash_id(str(node), f"{prefix}-")
            nodes_dict[hash_id] = {"content": str(node)}
        all_hash_ids = list(nodes_dict.keys())
        if not all_hash_ids:
            return []
        existing = self._embedd_store.get_by_ids(all_hash_ids)
        existing_ids = {doc.uid for doc in existing}
        missing_ids = [hash_id for hash_id in all_hash_ids if hash_id not in existing_ids]
        texts_to_encode = [nodes_dict[hash_id]["content"] for hash_id in missing_ids]
        docs = [
            Document(uid=hash_id, content=text, metadata=metadata)
            for hash_id, text in zip(missing_ids, texts_to_encode)
        ]
        if len(docs) == 0:
            return []
        logger.debug(f"Adding {len(docs)}:{docs} new embeddings to the store.")
        self._embedd_store.add_documents_filter_exists(docs)
        return docs

    def _get_not_existing_nodes(self, chunk_key: List[str]):
        """
        Retrieves nodes that do not exist in the graph for a given chunk key.
        """

    def _add_fact_edges(
        self, new_docs: List[Document], chunk_triples: Dict[str, List[Tuple[str, str, str]]]
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
                node_key = compute_mdhash_id(content=triple[0], prefix=(self._entity_prefix))
                node_2_key = compute_mdhash_id(content=triple[2], prefix=(self._entity_prefix))
                node_to_node_stats[(node_key, node_2_key)] = node_to_node_stats.get((node_key, node_2_key), 0.0) + 1
                node_to_node_stats[(node_2_key, node_key)] = node_to_node_stats.get((node_2_key, node_key), 0.0) + 1
                entities_in_chunk.add(node_key)
                entities_in_chunk.add(node_2_key)
            for node in entities_in_chunk:
                ent_node_to_chunk_ids[node] = list(set(ent_node_to_chunk_ids.get(node, set())).union(set([chunk_key])))

        return ent_node_to_chunk_ids, node_to_node_stats

    def _add_passage_edges(
        self,
        new_docs: List[Document],
        chunk_triples: Dict[str, List[Tuple[str, str, str]]],
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
                    node_key = compute_mdhash_id(chunk_ent, prefix=self._entity_prefix)
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

    def _add_new_nodes(
        self, entities: List[Document], passages: List[Document]
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
        existing_nodes: List[Dict[str, Any]] = self._graph_store.select_vertices(dict(name_in=node_ids))
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
            self._graph_store.add_vertices(new_nodes)
            logger.info(f"Added {len(new_nodes)} new nodes to the graph.")
        return new_entities, new_pasages
