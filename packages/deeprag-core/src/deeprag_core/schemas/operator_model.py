from deeprag_core.schemas.chat_response import ChatResponse
from pydantic import Field


class OperatorOutputModel(ChatResponse):
    name: str = Field(..., description="The name of the operator.")
