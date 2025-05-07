from abc import abstractmethod
from typing import Callable, Dict, Optional, TypeVar

from langchain.prompts import (
    ChatPromptTemplate,
)
from langchain_core.messages import AIMessage
from raghub_core.schemas.chat_response import ChatResponse
from raghub_core.utils.class_meta import RegsiterMeta

TChatResponse = TypeVar("TChatResponse", bound=ChatResponse)


class BaseChat(metaclass=RegsiterMeta):
    name = ""

    @abstractmethod
    def chat(
        self,
        prompt: ChatPromptTemplate,
        input: Dict[str, str],
        output_parser: Optional[Callable[[AIMessage], ChatResponse]] = None,
    ) -> ChatResponse:
        raise NotImplementedError("This `chat` method should be overridden by subclasses.")

    @abstractmethod
    async def achat(
        self,
        prompt: ChatPromptTemplate,
        input: Dict[str, str],
        output_parser: Optional[Callable[[AIMessage], ChatResponse]] = None,
    ) -> ChatResponse:
        raise NotImplementedError("This `achat` method should be overridden by subclasses.")

    @abstractmethod
    def preprocess_input(self, input: Dict[str, str]) -> Dict[str, str]:
        """
        Preprocess the input data before sending it to the LLM.
        """
        raise NotImplementedError("This `preprocess_input` method should be overridden by subclasses.")
