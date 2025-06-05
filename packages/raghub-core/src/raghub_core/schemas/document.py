from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class Document(BaseModel):
    content: str = Field(..., description="The content of the document.")
    metadata: Optional[Union[Dict[str, Any], BaseModel]] = Field(
        None, description="Metadata associated with the document."
    )
    uid: str = Field(None, description="Unique identifier for the document.")
    embedding: Optional[List[float]] = Field(None, description="Embedding vector of the document.")
    summary: Optional[str] = Field(None, description="Summary of the document.")
