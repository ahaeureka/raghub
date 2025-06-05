from typing import List

from pydantic import Field
from raghub_core.schemas.operator_model import OperatorOutputModel
from raghub_core.schemas.prompt import PromptModel


class SummarizeOperatorOutputModel(OperatorOutputModel):
    """Model for the output of the SummarizeOperator."""

    summary: str = Field(
        ...,
        description="The summary of the input text.",
    )
    keywords: List[str] = Field(
        ...,
        description="List of keywords extracted from the input text.",
    )


class SummarizePromptModel(PromptModel):
    """Model for the prompt used in the summarization task."""

    language: str = Field(
        ...,
        description="The language of the text.",
    )
    system_message: str = Field(
        ...,
        description="The system message defining the summarization task requirements.",
    )
    input_example: str = Field(
        ...,
        description="An example output showing the expected summary format.",
    )
    output_example: str = Field(
        ...,
        description="An example output showing the expected summary format.",
    )
    user_message: str = Field(
        ...,
        description="The input text to be summarized.",
    )
