from typing import Any, Optional

from openai import BaseModel


class ChatResponse(BaseModel):
    tokens: Optional[int] = None
    error: Optional[str] = None
    content: Optional[Any] = None


class QAChatResponse(ChatResponse):
    question: Optional[str] = None
    answer: Optional[str] = None
    context: Optional[str] = None
