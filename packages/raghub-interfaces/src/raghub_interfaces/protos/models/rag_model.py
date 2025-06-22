# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   rag.py
@Time    :
@Desc    :
"""

from typing import Any, Dict, List, Optional, Type

from google.protobuf import message as _message
from google.protobuf import message_factory
from protobuf_pydantic_gen.ext import PydanticModel, model2protobuf, pool, protobuf2model
from pydantic import BaseModel, ConfigDict
from pydantic import Field as _Field

from .chat_model import RetrievalSetting


class MetadataConditionItem(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    name: List[str] = _Field(description="Names of the metadata to filter")
    comparison_operator: str = _Field(description="Comparison operator")
    value: Optional[str] = _Field(
        description="Comparison value, can be omitted when the operator is empty, not empty, null, or not null",
        default="",
    )

    def to_protobuf(self) -> _message.Message:
        _proto = pool.FindMessageTypeByName("raghub_interfaces.MetadataConditionItem")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls: Type[PydanticModel], src: _message.Message) -> PydanticModel:
        return protobuf2model(cls, src)


class MetadataCondition(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    logical_operator: Optional[str] = _Field(
        description="Logical operator, values can be and or or, default is and", default="and"
    )
    conditions: List[MetadataConditionItem] = _Field(description="List of conditions")

    def to_protobuf(self) -> _message.Message:
        _proto = pool.FindMessageTypeByName("raghub_interfaces.MetadataCondition")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls: Type[PydanticModel], src: _message.Message) -> PydanticModel:
        return protobuf2model(cls, src)


class RetrievalRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    knowledge_id: str = _Field(description="knowledge’s unique ID")
    query: str = _Field(description="User’s query")
    retrieval_setting: RetrievalSetting = _Field(
        description="Knowledge’s retrieval parameters", default=RetrievalSetting()
    )
    metadata_condition: Optional[MetadataCondition] = _Field(description="Original array filtering", default=None)

    def to_protobuf(self) -> _message.Message:
        _proto = pool.FindMessageTypeByName("raghub_interfaces.RetrievalRequest")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls: Type[PydanticModel], src: _message.Message) -> PydanticModel:
        return protobuf2model(cls, src)


class RetrievalResponseRecord(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    content: str = _Field(description="Contains a chunk of text from a data source in the knowledge base.")
    score: float = _Field(description="The score of relevance of the result to the query, scope: 0~1")
    title: str = _Field(description="Document title")
    metadata: Optional[Dict[str, str]] = _Field(
        description="Contains metadata attributes and their values for the document in the data source."
    )

    def to_protobuf(self) -> _message.Message:
        _proto = pool.FindMessageTypeByName("raghub_interfaces.RetrievalResponseRecord")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls: Type[PydanticModel], src: _message.Message) -> PydanticModel:
        return protobuf2model(cls, src)


class RetrievalResponseError(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    error_code: int = _Field(description="Error code", default=1001)
    error_msg: str = _Field(description="The description of API exception")

    def to_protobuf(self) -> _message.Message:
        _proto = pool.FindMessageTypeByName("raghub_interfaces.RetrievalResponseError")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls: Type[PydanticModel], src: _message.Message) -> PydanticModel:
        return protobuf2model(cls, src)


class RetrievalResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    records: List[RetrievalResponseRecord] = _Field(description="A list of records from querying the knowledge base.")
    error: Optional[RetrievalResponseError] = _Field(description="Error information")
    request_id: Optional[str] = _Field(description="Request ID for tracking", default="")

    def to_protobuf(self) -> _message.Message:
        _proto = pool.FindMessageTypeByName("raghub_interfaces.RetrievalResponse")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls: Type[PydanticModel], src: _message.Message) -> PydanticModel:
        return protobuf2model(cls, src)


class CreateIndexRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    unique_name: str = _Field(description="Name of the index")

    def to_protobuf(self) -> _message.Message:
        _proto = pool.FindMessageTypeByName("raghub_interfaces.CreateIndexRequest")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls: Type[PydanticModel], src: _message.Message) -> PydanticModel:
        return protobuf2model(cls, src)


class CreateIndexResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    unique_name: str = _Field(description="Name of the index")

    def to_protobuf(self) -> _message.Message:
        _proto = pool.FindMessageTypeByName("raghub_interfaces.CreateIndexResponse")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls: Type[PydanticModel], src: _message.Message) -> PydanticModel:
        return protobuf2model(cls, src)


class ChatRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    knowledge_id: str = _Field(description="knowledge’s unique ID")
    question: str = _Field(description="User’s question")
    retrieval_setting: RetrievalSetting = _Field(
        description="Knowledge’s retrieval parameters", default=RetrievalSetting()
    )
    metadata_condition: Optional[MetadataCondition] = _Field(description="Original array filtering")

    def to_protobuf(self) -> _message.Message:
        _proto = pool.FindMessageTypeByName("raghub_interfaces.ChatRequest")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls: Type[PydanticModel], src: _message.Message) -> PydanticModel:
        return protobuf2model(cls, src)


class ChatResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    answer: str = _Field(description="The answer to the question based on the retrieved knowledge")
    records: List[RetrievalResponseRecord] = _Field(description="A list of records used to generate the answer")
    error: Optional[RetrievalResponseError] = _Field(description="Error information")

    def to_protobuf(self) -> _message.Message:
        _proto = pool.FindMessageTypeByName("raghub_interfaces.ChatResponse")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls: Type[PydanticModel], src: _message.Message) -> PydanticModel:
        return protobuf2model(cls, src)


class RAGDocument(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    content: str = _Field(description="Content of the document")
    title: str = _Field(description="Title of the document")
    metadata: Optional[Dict[str, Any]] = _Field(description="Metadata attributes and their values for the document")
    type: Optional[str] = _Field(description="Type of the document, e.g., 'text', 'image'", default="text")
    source: Optional[str] = _Field(
        description="Source of the document, e.g., 'knowledge_base', 'external'", default="knowledge_base"
    )

    def to_protobuf(self) -> _message.Message:
        _proto = pool.FindMessageTypeByName("raghub_interfaces.RAGDocument")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls: Type[PydanticModel], src: _message.Message) -> PydanticModel:
        return protobuf2model(cls, src)


class AddDocumentsRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    knowledge_id: str = _Field(description="knowledge’s unique ID")
    documents: List[RAGDocument] = _Field(description="List of documents to be added")

    def to_protobuf(self) -> _message.Message:
        _proto = pool.FindMessageTypeByName("raghub_interfaces.AddDocumentsRequest")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls: Type[PydanticModel], src: _message.Message) -> PydanticModel:
        return protobuf2model(cls, src)


class AddDocumentsResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    documents: List[RAGDocument] = _Field(description="List of documents that were added")
    error: Optional[RetrievalResponseError] = _Field(description="Error information")
    request_id: Optional[str] = _Field(description="Request ID for tracking", default="")

    def to_protobuf(self) -> _message.Message:
        _proto = pool.FindMessageTypeByName("raghub_interfaces.AddDocumentsResponse")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls: Type[PydanticModel], src: _message.Message) -> PydanticModel:
        return protobuf2model(cls, src)
