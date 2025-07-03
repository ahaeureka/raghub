import asyncio
import json
import traceback
from abc import ABC
from typing import Any, Dict, Generic, Optional, TypeVar

from langchain_core.messages import AIMessage
from langchain_core.output_parsers.json import JsonOutputParser
from loguru import logger
from raghub_core.chat.base_chat import BaseChat
from raghub_core.prompts.base_prompt import BasePrompt
from raghub_core.schemas.chat_response import ChatResponse
from raghub_core.schemas.operator_model import OperatorOutputModel

TOperatorOutputModel = TypeVar("TOperatorOutputModel", bound=OperatorOutputModel)


class BaseOperator(ABC, Generic[TOperatorOutputModel]):
    name = "BaseOperator"
    description = "Base class for all operators"
    output_cls: Optional[TOperatorOutputModel] = None

    def __init__(self, prompt: BasePrompt, chat: BaseChat):
        self._prompt = prompt
        self._chat = chat

    async def execute(self, input: Dict[str, Any], lang: str = "zh") -> TOperatorOutputModel:
        _prompt = self._prompt.get(lang)
        input = await self.pre_process(input)  # Pre-process the input
        rsp = await self._chat.achat(_prompt, input, self.output_parser)
        d = rsp.model_dump()
        d["name"] = self.name
        try:
            d = self.post_process(d)  # Apply post-processing to the output
            if not self.output_cls:
                raise ValueError("output_cls must be set")
            return self.output_cls.model_validate(d)
        except Exception as e:
            logger.error(f"Error in operator {self.name}: {e} with {d}")
            logger.error(traceback.format_exc())
            raise

    def output_parser(self, output: AIMessage) -> ChatResponse:
        parser = JsonOutputParser()
        try:
            content = (
                str(output.content) if not isinstance(output.content, (dict, list)) else json.dumps(output.content)
            )
            items = parser.parse(content)
            return ChatResponse(tokens=output.usage_metadata["total_tokens"], content=items)
        except Exception as e:
            logger.error(f"Error in parsing message content: {e}")
            logger.error(traceback.format_exc())
            return ChatResponse(tokens=output.usage_metadata["total_tokens"], content={})

    def post_process(self, output: Dict[str, Any]) -> Dict[str, Any]:
        # Implement any post-processing logic here
        return output["content"] if "content" in output else output

    async def pre_process(self, input: Dict[str, Any]) -> Dict[str, Any]:
        # Implement any pre-processing logic here
        await asyncio.sleep(0.01)
        return input
