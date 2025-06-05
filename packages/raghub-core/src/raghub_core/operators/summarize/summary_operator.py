from langchain_core.messages import AIMessage
from raghub_core.operators.base_operator import BaseOperator
from raghub_core.schemas.chat_response import ChatResponse
from raghub_core.schemas.summarize_model import SummarizeOperatorOutputModel


class SummaryOperator(BaseOperator[SummarizeOperatorOutputModel]):
    """Operator for summarizing text and extracting keywords."""

    name = "SummaryOperator"
    description = "Operator for summarizing text and extracting keywords"
    output_cls = SummarizeOperatorOutputModel

    def output_parser(self, output: AIMessage) -> ChatResponse:
        """Parse the output from the model into a structured format."""
        text = output.content
        if isinstance(text, list):
            text = "\n".join(text)
        summary_text = ""
        keywords = []
        lines = text.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("Summary:"):
                summary_text = line[len("Summary:") :].strip()
            elif line.startswith("-"):
                keywords.append(line[1:].strip())
        return ChatResponse(
            tokens=output.usage_metadata["total_tokens"],
            content=SummarizeOperatorOutputModel(
                summary=summary_text,
                keywords=list(set(keywords)),
                name=self.name,
            ).model_dump(),
        )
