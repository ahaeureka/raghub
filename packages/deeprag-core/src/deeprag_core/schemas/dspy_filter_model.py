from typing import List, Optional

from deeprag_core.schemas.operator_model import OperatorOutputModel
from deeprag_core.schemas.prompt import PromptModel
from pydantic import Field


class DSPyFilterPromptModel(PromptModel):
    question: Optional[str] = Field(None, description="The question to be filtered.")
    fact_before_filter: Optional[str] = Field(None, description="The fact before filtering.")
    fact_after_filter: Optional[str] = Field(None, description="The fact after filtering.")


class DSPyFilterOutputModel(OperatorOutputModel):
    fact_after_filter: List[List[str]] = Field(..., description="The fact after filtering.")
    completed: Optional[str] = Field(..., description="Marker indicating the completion of the filtering process.")
