from typing import List, Tuple

from pydantic import BaseModel, Field


class OpenIEModel(BaseModel):
    ner: List[str] = Field(..., description="List of named entities extracted from the text.")
    triples: List[Tuple[str, str, str]] = Field(..., description="List of triples extracted from the text.")
