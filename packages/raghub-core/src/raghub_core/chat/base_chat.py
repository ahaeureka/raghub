from abc import abstractmethod
from typing import AsyncIterator, Callable, Dict, Optional, TypeVar

from langchain.prompts import (
    ChatPromptTemplate,
)
from langchain_core.messages import AIMessage
from loguru import logger
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

    @abstractmethod
    async def astream(
        self,
        prompt: ChatPromptTemplate,
        input: Dict[str, str],
        output_parser: Optional[Callable[[AIMessage], ChatResponse]] = None,
    ) -> AsyncIterator[ChatResponse]:
        """
        Asynchronously stream the response from the LLM.
        This method should be overridden by subclasses if streaming is supported.
        """
        raise NotImplementedError("This `astream` method should be overridden by subclasses.")

    def default_output_parser(self, output: AIMessage) -> ChatResponse:
        """
        Default output parser for the chat response.
        This method can be overridden by subclasses if a different parsing logic is needed.
        """
        logger.debug(f"Default output parser called with output: {output}")
        return ChatResponse(
            tokens=output.usage_metadata["total_tokens"] if output.usage_metadata else 0,
            error=None,
            content=output.content,
        )
