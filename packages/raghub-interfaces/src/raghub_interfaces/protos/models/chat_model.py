# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   chat.py
@Time    :
@Desc    :
"""

from typing import Any, Dict, List, Optional, Type

from google.protobuf import message as _message
from google.protobuf import message_factory
from protobuf_pydantic_gen.ext import PydanticModel, model2protobuf, pool, protobuf2model
from pydantic import BaseModel, ConfigDict
from pydantic import Field as _Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    role: str = _Field(description="Role of the message, e.g., 'user', 'assistant', 'system'")
    content: str = _Field(description="Content of the message")
    name: Optional[str] = _Field(description="Name of the message sender, if applicable", default="")

    def to_protobuf(self) -> _message.Message:
        _proto = pool.FindMessageTypeByName("raghub_interfaces.ChatMessage")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls: Type[PydanticModel], src: _message.Message) -> PydanticModel:
        return protobuf2model(cls, src)


class RetrievalSetting(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    top_k: int = _Field(description="Maximum number of retrieved results", default=5)
    score_threshold: float = _Field(
        description="The score limit of relevance of the result to the query, scope: 0~1", default=0.7
    )

    def to_protobuf(self) -> _message.Message:
        _proto = pool.FindMessageTypeByName("raghub_interfaces.RetrievalSetting")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls: Type[PydanticModel], src: _message.Message) -> PydanticModel:
        return protobuf2model(cls, src)


class CreateChatCompletionRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model: str = _Field(description="The model to use for chat completion")
    messages: List[ChatMessage] = _Field(description="List of messages in the chat conversation")
    max_tokens: Optional[int] = _Field(description="Maximum number of tokens to generate in the response", default=1000)
    temperature: Optional[float] = _Field(description="Sampling temperature for response generation", default=0.7)
    metadata: Optional[Dict[str, Any]] = _Field(description="Additional options for the chat completion request")
    user: Optional[str] = _Field(description="Identifier for the user making the request", default="")
    stream: Optional[bool] = _Field(description="Whether to stream the response back", default=False)
    response_format: Optional[str] = _Field(description="Format of the response, e.g., 'json', 'text'", default="json")
    knowledge_id: Optional[str] = _Field(description="Knowledge base ID to use for the chat completion", default="")
    retrieval_setting: Optional[RetrievalSetting] = _Field(
        description="Knowledge base retrieval settings", default=RetrievalSetting()
    )

    def to_protobuf(self) -> _message.Message:
        _proto = pool.FindMessageTypeByName("raghub_interfaces.CreateChatCompletionRequest")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls: Type[PydanticModel], src: _message.Message) -> PydanticModel:
        return protobuf2model(cls, src)


class ChatCompletionChoice(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    index: str = _Field(description="Index of the choice in the response")
    message: ChatMessage = _Field(description="The message generated as part of the choice")
    finish_reason: Optional[float] = _Field(
        description="Reason for finishing the generation, e.g., 'stop', 'length'", default=0.0
    )

    def to_protobuf(self) -> _message.Message:
        _proto = pool.FindMessageTypeByName("raghub_interfaces.ChatCompletionChoice")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls: Type[PydanticModel], src: _message.Message) -> PydanticModel:
        return protobuf2model(cls, src)


class ChatCompletionUsage(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    prompt_tokens: int = _Field(description="Number of tokens in the prompt")
    completion_tokens: int = _Field(description="Number of tokens in the completion")
    total_tokens: int = _Field(description="Total number of tokens used in the request")

    def to_protobuf(self) -> _message.Message:
        _proto = pool.FindMessageTypeByName("raghub_interfaces.ChatCompletionUsage")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls: Type[PydanticModel], src: _message.Message) -> PydanticModel:
        return protobuf2model(cls, src)


class ChatCompletionReference(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    knowledge_id: str = _Field(description="Knowledge base ID used for the chat completion")
    chunk_id: str = _Field(description="ID of the chunk used in the chat completion")
    content: str = _Field(description="Content of the chunk used in the chat completion")
    type: Optional[str] = _Field(description="Type of the reference, e.g., 'text', 'image'", default="text")
    metadata: Optional[Dict[str, Any]] = _Field(description="Additional metadata about the reference")
    source: Optional[str] = _Field(
        description="Source of the reference, e.g., 'knowledge_base', 'external'", default="knowledge_base"
    )

    def to_protobuf(self) -> _message.Message:
        _proto = pool.FindMessageTypeByName("raghub_interfaces.ChatCompletionReference")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls: Type[PydanticModel], src: _message.Message) -> PydanticModel:
        return protobuf2model(cls, src)


class CreateChatCompletionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: str = _Field(description="Unique identifier for the chat completion request")
    object: str = _Field(description="Type of object returned, e.g., 'chat.completion'")
    created: int = _Field(description="Timestamp of when the chat completion was created")
    model: str = _Field(description="Model used for generating the chat completion")
    choices: List[ChatCompletionChoice] = _Field(description="List of choices generated in the chat completion")
    metadata: Optional[Dict[str, Any]] = _Field(description="Additional metadata about the chat completion response")
    usage: ChatCompletionUsage = _Field(description="Usage statistics for the chat completion request")
    references: Optional[List[ChatCompletionReference]] = _Field(
        description="References to knowledge base chunks used in the chat completion", default=None
    )

    def to_protobuf(self) -> _message.Message:
        _proto = pool.FindMessageTypeByName("raghub_interfaces.CreateChatCompletionResponse")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls: Type[PydanticModel], src: _message.Message) -> PydanticModel:
        return protobuf2model(cls, src)
