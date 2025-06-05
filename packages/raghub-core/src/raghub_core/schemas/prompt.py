from typing import Optional

from pydantic import BaseModel, Field


class PromptModel(BaseModel):
    language: str = Field(..., description="The language of the text.")
    system_message: str = Field(..., description="The system message to use for the prompt.")
    user_message: Optional[str] = Field(None, description="The user message to use for the prompt.")
    output_example: Optional[str] = Field(None, description="An example output for the prompt.")
