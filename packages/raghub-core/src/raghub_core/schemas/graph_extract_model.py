from typing import List

from pydantic import Field
from raghub_core.schemas.operator_model import OperatorOutputModel
from raghub_core.schemas.prompt import PromptModel


class GraphExtroactPromptModel(PromptModel):
    language: str = Field(..., description="The language of the text.")
    passage: str = Field(..., description="The input passage/text to extract relations from.")
    system_message: str = Field(
        ..., description="The system message defining the relation extraction task requirements."
    )
    context: str = Field(..., description="The conversation history or context for the extraction task.")
    example_output: str = Field(
        ..., description="An example output showing the extracted relations in RDF triple format."
    )


class GraphExtractOperatorOutputModel(OperatorOutputModel):
    entities: List[List[str]] = Field(..., description="List of entities extracted from the text.")
    triples: List[List[str]] = Field(..., description="List of triples extracted from the text.")
