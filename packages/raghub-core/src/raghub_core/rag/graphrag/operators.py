from abc import ABC, abstractmethod
from typing import Any, Dict

from loguru import logger
from raghub_core.chat.base_chat import BaseChat
from raghub_core.operators.graph.graph_extractor_operator import GraphExtractorOperator
from raghub_core.operators.graph.prompts import GraphExtractorPrompt
from raghub_core.operators.graph.query_indent_det import QueryIndentDetectionOperator
from raghub_core.operators.graph.query_indent_det_prompt import QueryIndentDetectionPrompt
from raghub_core.operators.keywords.keywords_operator import KeywordsOperator
from raghub_core.operators.keywords.prompts import KeywordsPrompts
from raghub_core.operators.summarize.community_summarizer_prompts import CommunitySummarizerPrompts
from raghub_core.operators.summarize.summary_operator import SummaryOperator
from raghub_core.schemas.graph_extract_model import GraphExtractOperatorOutputModel
from raghub_core.schemas.graph_model import QueryIndentationModel
from raghub_core.schemas.keywords_model import KeywordsOperatorOutputModel
from raghub_core.schemas.summarize_model import SummarizeOperatorOutputModel
from raghub_core.storage.vector import VectorStorage


class GraphRAGOperators(ABC):
    @abstractmethod
    async def extract_keywords(self, input: Dict[str, Any], lang="zh") -> KeywordsOperatorOutputModel:
        """
        Extract keywords from the input text.

        :param input: Input text for keyword extraction.
        :param lang: Language of the input text.
        :return: KeywordsOperatorOutputModel containing extracted keywords.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    async def summarize_communities(self, input: Dict[str, Any], lang="zh") -> SummarizeOperatorOutputModel:
        """
        Summarize communities from the input text.

        :param input: Input text for community summarization.
        :param lang: Language of the input text.
        :return: SummarizeOperatorOutputModel containing summarized communities.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    async def detect_query_indent(self, input: Dict[str, Any], lang="zh") -> QueryIndentationModel:
        """
        Detect query indentation in the input text.

        :param input: Input text for query indentation detection.
        :param lang: Language of the input text.
        :return: QueryIndentationModel containing detected query indentation.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    async def extract_graph(self, input: Dict[str, Any], lang="zh") -> GraphExtractOperatorOutputModel:
        """
        Extract graph entities and relationships from the input text.

        :param input: Input text for graph extraction.
        :param lang: Language of the input text.
        :return: GraphExtractOperatorOutputModel containing extracted entities and relationships.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")


class DefaultGraphRAGOperators(GraphRAGOperators):
    """
    Default implementation of GraphRAGOperators.
    This class can be extended to implement specific graph RAG operations.
    """

    def __init__(self, llm: BaseChat, embedding_store: VectorStorage):
        _summary_prompt = CommunitySummarizerPrompts()
        _keywords_prompt = KeywordsPrompts()
        self._communities_summarizer = SummaryOperator(_summary_prompt, llm)
        self._keywords_extractor = KeywordsOperator(_keywords_prompt, llm)
        _query_indent_prompt = QueryIndentDetectionPrompt()
        self._query_indent_extractor = QueryIndentDetectionOperator(_query_indent_prompt, llm)
        _graph_extractor_prompt = GraphExtractorPrompt()
        self._graph_extractor = GraphExtractorOperator(_graph_extractor_prompt, llm, embedding_store)

    async def extract_keywords(self, input: Dict[str, Any], lang="zh") -> KeywordsOperatorOutputModel:
        return await self._keywords_extractor.execute(input, lang=lang)

    async def summarize_communities(self, input: Dict[str, Any], lang="zh") -> SummarizeOperatorOutputModel:
        return await self._communities_summarizer.execute(input, lang=lang)

    async def detect_query_indent(self, input: Dict[str, Any], lang="zh") -> QueryIndentationModel:
        return await self._query_indent_extractor.execute(input, lang=lang)

    async def extract_graph(self, input: Dict[str, Any], lang="zh") -> GraphExtractOperatorOutputModel:
        try:
            return await self._graph_extractor.execute(input, lang=lang)
        except Exception as e:
            import traceback

            logger.error(f"Error extracting graph: {str(e)}\n{traceback.format_exc()}")
            # Handle exceptions and return an empty model or raise as needed
            raise e
