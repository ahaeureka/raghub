from typing import List

from deeprag_core.schemas.operator_model import OperatorOutputModel
from deeprag_core.schemas.prompt import PromptModel
from pydantic import Field


class NERPromptModel(PromptModel):
    language: str = Field(..., description="The language of the text.")
    paragraph: str = Field(..., description="The paragraph of text to extract named entities from.")
    system_message: str = Field(..., description="The system message to use for the prompt.")
    example_output: str = Field(..., description="An example output of the named entities.")


class NEROperatorOutputModel(OperatorOutputModel):
    named_entities: List[str] = Field(..., description="List of named entities extracted from the text.")
