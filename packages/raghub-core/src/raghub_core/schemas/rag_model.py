from typing import Any, Dict, List, Optional

from openai import BaseModel
from pydantic import Field
from raghub_core.schemas.document import Document


class RetrieveResultItem(BaseModel):
    document: Document = Field(..., description="The retrieved document.")
    score: float = Field(..., description="Score associated with the retrieved document.")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for the retrieved item.")
    query: str = Field(..., description="The original query used for retrieval.")


class RetrieveResult:
    documents: List[Document] = Field(..., description="List of retrieved documents.")
    scores: List[float] = Field(..., description="Scores associated with each retrieved document.")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for the retrieval result.")
    query: List[str] = Field(..., description="The original query used for retrieval.")
