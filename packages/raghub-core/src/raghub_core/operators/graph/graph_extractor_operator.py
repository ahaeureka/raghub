import re
from typing import Dict, List, Optional

from langchain_core.messages import AIMessage
from raghub_core.chat.base_chat import BaseChat
from raghub_core.operators.base_operator import BaseOperator
from raghub_core.prompts.base_prompt import BasePrompt
from raghub_core.schemas.chat_response import ChatResponse
from raghub_core.schemas.document import Document
from raghub_core.schemas.graph_extract_model import GraphExtractOperatorOutputModel
from raghub_core.storage.vector import VectorStorage
from raghub_core.utils.misc import compute_mdhash_id


class GraphExtractorOperator(BaseOperator[GraphExtractOperatorOutputModel]):
    name = "GraphExtractorOperator"
    description = "Operator for extracting entities and triples from text"
    output_cls = GraphExtractOperatorOutputModel

    def __init__(
        self,
        prompt: BasePrompt,
        chat: BaseChat,
        embedding_store: VectorStorage,
        top_k: Optional[int] = 5,
        score_threshold: Optional[float] = 0.7,
    ):
        super().__init__(prompt, chat)
        self._top_k = top_k
        self._score_threshold = score_threshold
        self._chunk_history = embedding_store

    def output_parser(self, output: AIMessage) -> ChatResponse:
        # logger.debug(f"GraphExtractorOperator output: {output}")
        text = output.content
        if isinstance(text, list):
            text = "\n".join(text)
        entities: List[List[str]] = []
        triples: List[List[str]] = []
        current_section = None
        for line in text.split("\n"):
            line = line.strip()
            if line in ["Entities:", "Relationships:"]:
                current_section = line[:-1]
            elif line and current_section:
                if current_section == "Entities":
                    match = re.match(r"\((.*?)#(.*?)\)", line)
                    if match:
                        name, summary = [part.strip() for part in match.groups()]
                        entities.append([name, summary])
                elif current_section == "Relationships":
                    match = re.match(r"\((.*?)#(.*?)#(.*?)#(.*?)\)", line)
                    if match:
                        source, name, target, summary = [part.strip() for part in match.groups()]
                        triples.append([source, name, target, summary])
        out = GraphExtractOperatorOutputModel(
            entities=entities,
            triples=triples,
            tokens=output.usage_metadata["total_tokens"],
            name=self.name,
        )
        return ChatResponse(tokens=output.usage_metadata["total_tokens"], content=out.model_dump())

    async def aload_chunk_context(self, index_name: str, texts: List[str]) -> Dict[str, str]:
        """Load chunk context."""
        text_context_map: Dict[str, str] = {}
        documents: List[Document] = []
        for text in texts:
            # Load similar chunks
            chunks = await self._chunk_history.asimilar_search_with_scores(index_name, text, self._top_k)
            # Filter chunks based on score threshold
            chunks = [(chunk, score) for chunk, score in chunks if score >= self._score_threshold]
            # Sort chunks by score
            chunks.sort(key=lambda x: x[1], reverse=True)

            history = [f"Section {i + 1}:\n{chunk[0].content}" for i, chunk in enumerate(chunks)]
            context = "\n".join(history) if history else ""
            text_context_map[text] = context
            documents.append(
                Document(content=text, metadata={"relevant_cnt": len(history)}, uid=compute_mdhash_id(text, "context"))
            )
        await self._chunk_history.add_documents_filter_exists(index_name, documents)
        return text_context_map

    async def pre_process(self, input):
        text = input["text"]
        index_name = input["index_name"]
        if not text:
            raise ValueError("text is required")
        if not index_name:
            raise ValueError("index_name is required")
        # Load chunk context
        text_context_map = await self.aload_chunk_context(index_name, [text])
        if text_context_map:
            input["histories"] = text_context_map[text]
        return input
