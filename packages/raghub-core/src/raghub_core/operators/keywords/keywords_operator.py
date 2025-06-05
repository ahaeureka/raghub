from langchain_core.messages import AIMessage
from raghub_core.operators.base_operator import BaseOperator
from raghub_core.schemas.chat_response import ChatResponse
from raghub_core.schemas.keywords_model import KeywordsOperatorOutputModel


class KeywordsOperator(BaseOperator[KeywordsOperatorOutputModel]):
    """Operator for keywords text and extracting keywords."""

    name = "KeywordsOperator"
    description = "Operator for summarizing text and extracting keywords"
    output_cls = KeywordsOperatorOutputModel

    def output_parser(self, output: AIMessage) -> ChatResponse:
        """Parse the output from the model into a structured format."""
        text = output.content
        if isinstance(text, list):
            text = "\n".join(text)
        lines = text.replace(":", "\n").split("\n")
        keywords = set()
        for line in lines:
            for part in line.split(";"):
                for s in part.strip().split(","):
                    keyword = s.strip()
                    if keyword:
                        keywords.add(keyword)
        return ChatResponse(
            tokens=output.usage_metadata["total_tokens"],
            content=KeywordsOperatorOutputModel(name=self.name, keywords=keywords).model_dump(),
        )
