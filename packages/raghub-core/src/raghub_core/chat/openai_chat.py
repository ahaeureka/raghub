from typing import Any, AsyncIterator, Callable, Dict, Optional, TypeVar

import tenacity
from langchain.prompts import (
    ChatPromptTemplate,
)
from langchain_core.caches import BaseCache
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_openai import ChatOpenAI
from loguru import logger
from pydantic import SecretStr
from raghub_core.chat.base_chat import BaseChat
from raghub_core.schemas.chat_response import ChatResponse

TChatResponse = TypeVar("TChatResponse", bound=ChatResponse)


class OpenAIProxyChat(BaseChat):
    name = "openai-proxy"

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.3,
        timeout: int = 30,
        response_cache: Optional[BaseCache] = None,
    ):
        self._model_name = model_name
        self._api_key = api_key
        self._base_url = base_url
        self._temperature = temperature
        self._timeout = timeout
        self._response_cache = response_cache

    def _build_chain(
        self, prompt: ChatPromptTemplate, output_parser: Optional[Callable[[AIMessage], TChatResponse]] = None
    ) -> Runnable[Any, TChatResponse]:
        from langchain.callbacks import StreamingStdOutCallbackHandler

        llm = ChatOpenAI(
            api_key=SecretStr(self._api_key) if self._api_key is not None else None,
            base_url=self._base_url,
            model=self._model_name,
            temperature=self._temperature,
            timeout=self._timeout,
            verbose=True,  # 启用详细日志
            callbacks=[StreamingStdOutCallbackHandler()],  # 实时输出调试信息
            # cache=self._response_cache,  # 添加缓存支持
        )
        # 构建链，将 prompt 和 llm 串联起来
        chain: Runnable = prompt | llm
        # 如果提供了 output_parser，添加到链中
        if output_parser is not None:
            chain = chain | RunnableLambda(output_parser)
        else:
            # 默认输出解析器
            chain = chain | RunnableLambda(self.default_output_parser)
        return chain

    @tenacity.retry(
        wait=tenacity.wait_random_exponential(multiplier=2, max=60),
        stop=tenacity.stop_after_attempt(5),
        reraise=True,
    )
    def chat(
        self,
        prompt: ChatPromptTemplate,
        input: Dict[str, str],
        output_parser: Optional[Callable[[AIMessage], ChatResponse]] = None,
    ) -> ChatResponse:
        chain = self._build_chain(prompt, output_parser)
        # 执行链并返回结果
        input = self.preprocess_input(input)
        logger.debug(f"Executing chain with input: {input}")
        return chain.invoke(input)
        # llm.invoke(prompt, output_parser=output_parser)

    @tenacity.retry(
        wait=tenacity.wait_random_exponential(multiplier=1, max=60),
        stop=tenacity.stop_after_attempt(5),
        reraise=True,
    )
    async def achat(
        self,
        prompt: ChatPromptTemplate,
        input: Dict[str, str],
        output_parser: Optional[Callable[[AIMessage], ChatResponse]] = None,
    ) -> ChatResponse:
        chain = self._build_chain(prompt, output_parser)
        # 执行链并返回结果
        input = self.preprocess_input(input)
        return await chain.ainvoke(input)

    async def astream(
        self,
        prompt: ChatPromptTemplate,
        input: Dict[str, str],
        output_parser: Optional[Callable[[AIMessage], ChatResponse]] = None,
    ) -> AsyncIterator[ChatResponse]:
        chain = self._build_chain(prompt, output_parser)
        # 执行链并返回结果
        input = self.preprocess_input(input)
        async for ans in chain.astream(input):
            yield ans

    def preprocess_input(self, input: Dict[str, str]) -> Dict[str, str]:
        """
        Preprocess the input data before sending it to the LLM.
        """
        return input
