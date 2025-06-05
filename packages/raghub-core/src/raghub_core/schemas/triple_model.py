from typing import List

from openai import BaseModel
from pydantic import Field
from raghub_core.schemas.operator_model import OperatorOutputModel
from raghub_core.schemas.prompt import PromptModel


class Triple(BaseModel):
    subject: str = Field(..., description="The subject of the triple.")
    predicate: List[str] = Field(..., description="The predicate of the triple.")
    object: List[str] = Field(..., description="The object of the triple.")


class TriplesOperatorOutputModel(OperatorOutputModel):
    triples: List[Triple] = Field(
        ...,
        description="List of keywords extracted from the input text.",
    )


class TriplePromptModel(PromptModel):
    """Model for the prompt used in the summarization task."""

    pass
