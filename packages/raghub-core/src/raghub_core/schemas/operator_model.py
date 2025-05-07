from pydantic import Field
from raghub_core.schemas.chat_response import ChatResponse


class OperatorOutputModel(ChatResponse):
    name: str = Field(..., description="The name of the operator.")
