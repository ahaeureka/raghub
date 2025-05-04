import copy
import difflib
import itertools
import json
import re
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from deeprag_app.app_schemas.hipporag_models import OpenIEInfo
from deeprag_app.apps.hipporag.prompts import DSPyRerankPrompt, get_query_instruction
from deeprag_app.apps.hipporag.storage import HippoRAGStorage
from deeprag_app.config.config_models import APPConfig
from deeprag_core.chat.base_chat import BaseChat
from deeprag_core.embedding import BaseEmbedding
from deeprag_core.operators.ner.ner_operator import NEROperator
from deeprag_core.operators.ner.promopts import NERPrompt
from deeprag_core.operators.openie.openie_operator import OpenIEOperator
from deeprag_core.operators.rdf.prompts import REPrompt
from deeprag_core.operators.rdf.rdf_operator import RDFOperator
from deeprag_core.rag.base_rag import BaseRAG
from deeprag_core.schemas.document import Document
from deeprag_core.schemas.openie_mdoel import OpenIEModel
from deeprag_core.schemas.rag_model import RetrieveResultItem
from deeprag_core.storage.graph import GraphStorage
from deeprag_core.storage.vector import VectorStorage
from deeprag_core.utils.class_meta import ClassFactory
from deeprag_core.utils.misc import compute_mdhash_id
from loguru import logger
from tqdm import tqdm

from .rerank import DSPyFilter


class HippoRAGImpl(BaseRAG):
    def __init__(self, config: APPConfig):
        self._llm = ClassFactory.get_instance(
            config.rag.llm.provider,
            BaseChat,
            model_name=config.rag.llm.model,
            api_key=config.rag.llm.api_key,
            base_url=config.rag.llm.base_url,
            temperature=config.rag.llm.temperature,
            timeout=config.rag.llm.timeout,
        )  # Assuming BaseChat is imported from the correct module
        ner_prompt = NERPrompt()
        self._ner = NEROperator(ner_prompt, self._llm)
        rdf_prompt = REPrompt()  # Assuming RDFPrompt is defined somewhere
        self._rdf = RDFOperator(rdf_prompt, self._llm)
        self._openie = OpenIEOperator(self._ner, self._rdf)
        dsp_prompt = DSPyRerankPrompt(best_dspy_path=config.hipporag.dspy_file_path)
        super().__init__()
        self._embedding_prefix = config.rag.embbeding.embedding_key_prefix

        self._embedder = ClassFactory.get_instance(
            "Embbedder",
            BaseEmbedding,
            model=config.rag.embbeding.model,
            provider=config.rag.embbeding.provider,
            batch_size=config.rag.embbeding.batch_size,
            base_url=config.rag.embbeding.base_url,
            api_key=config.rag.embbeding.api_key,
            n_dims=config.rag.embbeding.n_dims,
        )

        self._embedd_store: VectorStorage = ClassFactory.get_instance(
            config.vector_storage.provider, VectorStorage, embedder=self._embedder, **config.vector_storage.model_dump()
        )
        self._graph_store: GraphStorage = ClassFactory.get_instance(
            config.graph.provider, GraphStorage, **config.graph.model_dump()
        )
        config.hipporag.dspy_file_path
        self.rerank_filter = DSPyFilter(dsp_prompt, self._llm)
        self.config = config
        self._db = HippoRAGStorage(config.database, config.cache)

        self._ready_to_retrieve = False
        self._query_to_embedding: Dict = {"triple": {}, "passage": {}}

        self._entity_prefix = "entity"
        self._fact_prefix = "fact"
        self._doc_prefix = "doc"

    def add_documents(self, texts: List[Document], lang="en") -> List[Document]:
        chunk_triples: Dict[str, List[Tuple[str, str, str]]] = {}  # a dictionary of triples for each chunk
        openie_infos: List[OpenIEInfo] = []
        entities_str = []
        facts_str = []
        for doc in texts:
            if not doc.uid:
                doc.uid = compute_mdhash_id(doc.content, prefix=self._doc_prefix)
            doc.metadata = doc.metadata or {}
            doc.metadata["namespace"] = "passage"
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
            chunk_triples[doc.uid] = [tuple(item) for item in ie.triples if len(item) == 3]

            entities_str.extend([str(node) for node in graph_nodes])
            facts = self._flatten_facts([ie.triples])
            facts_str.extend([str(fact) for fact in facts])

        entities_str = list(set(entities_str))
        facts_str = list(set(facts_str))
        logger.debug(f"Entities: {entities_str}")
        embed_entities = self.add_embeddings(entities_str, self._entity_prefix, metadata={"namespace": "entity"})
        logger.debug(f"Facts: {facts_str}")

        self.add_embeddings(facts_str, self._fact_prefix, metadata={"namespace": "fact"})

        self._db.save_openie_info(openie_infos)
        vcount = self._graph_store.vertices_count()
        logger.info(f"Current graph vertex count: {vcount}")
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
            self._db.set_ent_node_to_chunk_ids(key, chunks)
        for node_to_node, stats in node_to_node_stats.items():
            self._db.set_node_to_node_stats(node_to_node[0], node_to_node[1], stats)
        return texts

    def _get_query_embeddings(self, queries: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Retrieves embeddings for given queries and updates the internal query-to-embedding mapping.
        The method determines whether each query
        is already present in the `self.query_to_embedding` dictionary under the keys 'triple' and 'passage'.
        If a query is not present in
        either, it is encoded into embeddings using the embedding model and stored.

        Args:
            queries List[str] | List[QuerySolution]: A list of query strings or QuerySolution objects.
            Each query is checked for
            its presence in the query-to-embedding mappings.
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
            logger.debug(f"_get_query_embeddings query_to_embedding: {query_to_embedding}")
            return query_to_embedding

        return query_to_embedding

    def _get_fact_scores(
        self, query: str, query_to_embedding: Optional[Dict[str, Dict[str, np.ndarray]]] = {}, top_k: int = 10
    ) -> List[Tuple[Document, float]]:
        """
        Retrieves and computes normalized similarity scores between the given query and pre-stored fact embeddings.

        Parameters:
        query : str
            The input query text for which similarity scores with fact embeddings
            need to be computed.

        Returns:
        numpy.ndarray
            A normalized array of similarity scores between the query and fact
            embeddings. The shape of the array is determined by the number of
            facts.

        Raises:
        KeyError
            If no embedding is found for the provided query in the stored query
            embeddings dictionary.
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

        Args:

        Returns:
            top_k_fact_indicies:
            top_k_facts:
            rerank_log (dict): {'facts_before_rerank': candidate_facts, 'facts_after_rerank': top_k_facts}
                - candidate_facts (list): list of link_top_k facts (each fact is a relation triple in tuple data type).
                - top_k_facts:


        """
        # load args

        docs: List[Tuple[Document, float]] = sorted(query_fact_scores, key=lambda x: x[1], reverse=True)
        logger.debug(f"rerank_facts Top {link_top_k} facts: {docs}")
        # candidate_facts = [
        #     eval(doc.content) for doc, _ in query_fact_scores
        # ]  # list of link_top_k facts (each fact is a relation triple in tuple data type)
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
        self, query: str, query_to_embedding: Dict[str, Dict[str, np.ndarray]], top_k: int = 10
    ) -> List[Tuple[Document, float]]:
        """
        Conduct dense passage retrieval to find relevant documents for a query.

        This function processes a given query using a pre-trained embedding model
        to generate query embeddings. The similarity scores between the query
        embedding and passage embeddings are computed using dot product, followed
        by score normalization. Finally, the function ranks the documents based
        on their similarity scores and returns the ranked document identifiers
        and their scores.

        Parameters
        ----------
        query : str
            The input query for which relevant passages should be retrieved.

        Returns
        -------
        tuple : Tuple[np.ndarray, np.ndarray]
            A tuple containing two elements:
            - A list of sorted document identifiers based on their relevance scores.
            - A numpy array of the normalized similarity scores for the corresponding
              documents.
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
        Computes document scores based on fact-based similarity and relevance using personalized
        PageRank (PPR) and dense retrieval models. This function combines the signal from the relevant
        facts identified with passage similarity and graph-based search for enhanced result ranking.

        Parameters:
            query (str): The input query string for which similarity and relevance computations
                need to be performed.
            link_top_k (int): The number of top phrases to include from the linking score map for
                downstream processing.
            query_fact_scores (np.ndarray): An array of scores representing fact-query similarity
                for each of the provided facts.
            top_k_facts (List[Tuple]): A list of top-ranked facts, where each fact is represented
                as a tuple of its subject, predicate, and object.
            top_k_fact_indices (List[str]): Corresponding indices or identifiers for the top-ranked
                facts in the query_fact_scores array.
            passage_node_weight (float): Default weight to scale passage scores in the graph.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two arrays:
                - The first array corresponds to document IDs sorted based on their scores.
                - The second array consists of the PPR scores associated with the sorted document IDs.
        """
        # Assigning phrase weights based on selected facts from previous steps.
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
        nodes = self._graph_store.select_vertices(name_in=phrase_keys)
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
        passages = self.dense_passage_retrieval(query, query_to_embedding, self._graph_store.vertices_count())
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
        ppr_sorted_doc_scores = self.run_ppr(node_weights, damping=damping, top_k=top_k)

        return ppr_sorted_doc_scores

    def get_top_k_weights(
        self, link_top_k: int, all_phrase_weights: Dict[str, np.float64], linking_score_map: Dict[str, float]
    ) -> Tuple[Dict[str, np.float64], Dict[str, float]]:
        """
        Filters all_phrase_weights to retain only the weights for the top-ranked phrases in linking_score_map.
        Non-selected phrases are reset to 0.0.

        Args:
            link_top_k (int): Number of top-ranked phrases to retain.
            all_phrase_weights (Dict[str, np.float64]): A dictionary mapping phrase IDs to their weights.
            linking_score_map (Dict[str, float]): A mapping of phrase content to its linking score.

        Returns:
            Tuple[Dict[str, np.float64], Dict[str, float]]: Filtered weights and linking scores.
        """
        # Step 1: 选择 top-k 短语
        linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:link_top_k])

        # Step 2: 生成 top-k 短语的 ID（如 "entity-<md5>"）
        top_k_phrases = set(linking_score_map.keys())
        top_k_phrase_ids = set(
            [compute_mdhash_id(content=phrase, prefix=self._entity_prefix) for phrase in top_k_phrases]
        )

        # Step 3: 查询图中实际存在的短语 ID
        top_k_nodes = self._graph_store.select_vertices(name_in=list(top_k_phrase_ids))
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

    def retrieve(self, queries: List[str], retrieve_top_k=10, lang="en", link_top_k=5) -> List[RetrieveResultItem]:
        """
        Performs retrieval using the HippoRAG 2 framework, which consists of several steps:
        - Fact Retrieval
        - Recognition Memory for improved fact selection
        - Dense passage scoring
        - Personalized PageRank based re-ranking

        Parameters:
            queries: List[str]
                A list of query strings for which documents are to be retrieved.
            num_to_retrieve: int, optional
                The maximum number of documents to retrieve for each query. If not specified, defaults to
                the `retrieval_top_k` value defined in the global configuration.
            gold_docs: List[List[str]], optional
                A list of lists containing gold-standard documents corresponding to each query. Required
                if retrieval performance evaluation is enabled (`do_eval_retrieval` in global configuration).

        Returns:
            List[QuerySolution] or (List[QuerySolution], Dict)
                If retrieval performance evaluation is not enabled, returns a list of QuerySolution objects,
                each containing
                the retrieved documents and their scores for the corresponding query.
                If evaluation is enabled, also returns
                a dictionary containing the evaluation metrics computed over the retrieved results.

        Notes
        -----
        - Long queries with no relevant facts after reranking will default to results from dense passage retrieval.
        """
        logger.info("Starting retrieval process...")

        # if not self._ready_to_retrieve:
        #     self.prepare_retrieval_objects()

        query_to_embedding = self._get_query_embeddings(queries)
        top_k_docs: List[RetrieveResultItem] = []
        for q_idx, query in tqdm(enumerate(queries), desc="Retrieving", total=len(queries)):
            # rerank_start = time.time()
            query_fact_scores = self._get_fact_scores(query, query_to_embedding)
            logger.debug(f"Query: {query}, Fact scores: {query_fact_scores}")
            top_k_facts = self.rerank_facts(query, query_fact_scores, lang=lang, link_top_k=link_top_k)
            logger.debug(f"retrieve Top {self.config.hipporag.linking_top_k} facts: {top_k_facts}")
            # rerank_end = time.time()

            # self.rerank_time += rerank_end - rerank_start

            if len(top_k_facts) == 0:
                logger.info("No facts found after reranking, return DPR results")
                sorted_doc_scores = self.dense_passage_retrieval(
                    query, query_to_embedding, top_k=self._graph_store.vertices_count()
                )
                top_k_docs.extend([RetrieveResultItem(document=d, score=s, query=query) for d, s in sorted_doc_scores])
            else:
                sorted_doc_key_scores = self.graph_search_with_fact_entities(
                    query=query,
                    query_to_embedding=query_to_embedding,
                    link_top_k=self.config.hipporag.linking_top_k,
                    top_k_facts=top_k_facts,
                    passage_node_weight=self.config.hipporag.passage_node_weight,
                    top_k=retrieve_top_k,
                )
                docs_ids = list(sorted_doc_key_scores.keys())
                logger.debug(f"docs_ids: {docs_ids}")
                docs = self._embedd_store.get_by_ids(docs_ids)
                top_k_docs.extend(
                    [
                        RetrieveResultItem(document=doc, score=sorted_doc_key_scores[doc.uid], query=query, metadata={})
                        for doc in docs
                    ]
                )
        return top_k_docs

    def run_ppr(self, node_weights: Dict[str, np.float64], damping: float = 0.5, top_k: int = 10) -> Dict[str, float]:
        """
        Runs Personalized PageRank (PPR) on a graph and computes relevance scores for
        nodes corresponding to document passages. The method utilizes a damping
        factor for teleportation during rank computation and can take a reset
        probability array to influence the starting state of the computation.

        Parameters:
            reset_prob (np.ndarray): A 1-dimensional array specifying the reset
                probability distribution for each node. The array must have a size
                equal to the number of nodes in the graph. NaNs or negative values
                within the array are replaced with zeros.
            damping (float): A scalar specifying the damping factor for the
                computation. Defaults to 0.5 if not provided or set to `None`.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays. The
                first array represents the sorted node IDs of document passages based
                on their relevance scores in descending order. The second array
                contains the corresponding relevance scores of each document passage
                in the same order.
        """

        if damping is None:
            damping = 0.5  # for potential compatibilityp
        # reset_prob = np.array(list(node_weights.values()))
        reset_prob = np.zeros(self._graph_store.vertices_count())
        # logger.debug(f"Reset probability: {np_reset_prob} and vertices size:{np_reset_prob.shape[0]}")
        vs = self._graph_store.select_vertices(name_in=list(node_weights.keys()))

        node_name_to_vertex_idx = {v["name"]: v["index"] for v in vs}
        for node_name, node_weight in node_weights.items():
            if node_name in node_name_to_vertex_idx:
                vertex_idx = node_name_to_vertex_idx[node_name]
                reset_prob[vertex_idx] = node_weight
        # reset_prob = np.array(reset_prob)  # Ensure reset_prob is a NumPy array
        np_reset_prob = np.where(np.isnan(reset_prob) | (reset_prob < 0.0), 0, reset_prob)

        logger.debug(f"node_name_to_vertex_idx: {node_name_to_vertex_idx}:{reset_prob.tolist()}")
        pagerank_scores = self._graph_store.personalized_pagerank(
            vertices=range(self._graph_store.vertices_count()),
            damping=damping,
            directed=False,
            weights="weight",
            reset=np_reset_prob,
            arpack_options=None,
            implementation="prpack",
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
        self._add_new_edges(node_to_node_stats)

        logger.info("Graph construction completed!")

    def _save(self):
        from deeprag_core.storage.igraph_store import IGraphStore

        if isinstance(self._graph_store, IGraphStore):
            self._graph_store.save(self.config.graph.graph_path)
        logger.info("Saving graph completed!")

    def init(self):
        self._embedder.init()
        self._db.init()
        self._embedd_store.init()
        self._graph_store.init()

    def _flatten_facts(self, chunk_triples: List[List[Tuple[str, str, str]]]) -> List[Tuple[str, ...]]:
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
        Stores embeddings for entity nodes in the graph.
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
        node_to_node_stats: Dict[Tuple[str, str], float] = {}
        ent_node_to_chunk_ids: Dict[str, List[str]] = {}
        for doc in tqdm(new_docs):
            logger.debug(f"add_fact_edges Processing doc: {doc.uid}:{chunk_triples}")
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
        print(f"After add_passage_edges node_to_node_stats: {node_to_node_stats}")
        return num_new_chunks

    def _add_synonymy_edges(
        self, query_docs: List[Document], target_docs: List[Document], node_to_node_stats: Dict[Tuple[str, str], float]
    ):
        query_node_key2knn_node_keys = self._embedd_store.knn(
            query_docs=query_docs,
            target_docs=target_docs,
            k=self.config.hipporag.synonymy_edge_topk,
            query_batch_size=self.config.hipporag.synonymy_edge_query_batch_size,
            key_batch_size=self.config.hipporag.synonymy_edge_key_batch_size,
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
                    if score < self.config.hipporag.synonymy_edge_sim_threshold or num_nns > 100:
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
        Adds new nodes to the graph from entity and passage embedding stores based on their attributes.

        This method identifies and adds new nodes to the graph by comparing existing nodes
        in the graph and nodes retrieved from the entity embedding store and the passage
        embedding store. The method checks attributes and ensures no duplicates are added.
        New nodes are prepared and added in bulk to optimize graph updates.
        """

        docs = copy.deepcopy(entities)
        docs.extend(passages)

        node_to_rows_map = {doc.uid: doc.model_dump() for doc in docs}
        node_ids = list(node_to_rows_map.keys())
        existing_nodes: List[Dict[str, Any]] = self._graph_store.select_vertices(name_in=node_ids)
        existing_ids = [node["name"] for node in existing_nodes]
        new_nodes: Dict[str, List[Any]] = {}
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
            for k, v in node.items():
                if k not in new_nodes:
                    new_nodes[k] = []
                new_nodes[k].append(v)
        if len(new_nodes) > 0:
            self._graph_store.add_vertices(n=len(next(iter(new_nodes.values()))), attributes=new_nodes)
        logger.info(f"Added {len(new_nodes)} new nodes to the graph.")
        return new_entities, new_pasages

    def _add_new_edges(self, node_to_node_stats: Dict[Tuple[str, str], float]):
        """
        Processes edges from `node_to_node_stats` to add them into a graph object while
        managing adjacency lists, validating edges, and logging invalid edge cases.
        """

        # 初始化邻接表（若后续需要使用）
        # self.graph_adj_list: DefaultDict[str, Dict[str, float]] = defaultdict(dict)
        # self.graph_inverse_adj_list: DefaultDict[str, Dict[str, float]] = defaultdict(dict)

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

        source_nodes = self._graph_store.select_vertices(name_in=edge_source_node_keys)
        target_nodes = self._graph_store.select_vertices(name_in=edge_target_node_keys)
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
        logger.info(f"Adding {valid_edges} edges to the graph.")
        self._graph_store.add_edges(valid_edges, valid_weights)
