# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   rag_model.py
@Time    :   2025-07-02 09:16:48
@Desc    :   Generated Pydantic models from protobuf definitions
"""

from .chat_model import RetrievalSetting
from google.protobuf import message as _message, message_factory
from protobuf_pydantic_gen.ext import model2protobuf, pool, protobuf2model
from pydantic import BaseModel, ConfigDict, Field as _Field
from typing import Any, Dict, List, Optional, Type


class MetadataConditionItem(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    name: List[str] = _Field(description="Names of the metadata to filter", default="")
    comparison_operator: str = _Field(description="Comparison operator", default="")
    value: Optional[str] = _Field(
        description="Comparison value, can be omitted when the operator is empty, not empty, null, or not null",
        default="",
    )

    def to_protobuf(self) -> _message.Message:
        """Convert Pydantic model to protobuf message"""
        _proto = pool.FindMessageTypeByName("raghub_interfaces.MetadataConditionItem")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls, src: _message.Message) -> "MetadataConditionItem":
        """Convert protobuf message to Pydantic model"""
        return protobuf2model(cls, src)


class MetadataCondition(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    logical_operator: Optional[str] = _Field(
        description="Logical operator, values can be and or or, default is and", default="and"
    )
    conditions: List[MetadataConditionItem] = _Field(description="List of conditions", default=None)

    def to_protobuf(self) -> _message.Message:
        """Convert Pydantic model to protobuf message"""
        _proto = pool.FindMessageTypeByName("raghub_interfaces.MetadataCondition")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls, src: _message.Message) -> "MetadataCondition":
        """Convert protobuf message to Pydantic model"""
        return protobuf2model(cls, src)


class RetrievalRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    knowledge_id: str = _Field(description="knowledge’s unique ID", default="")
    query: str = _Field(description="User’s query", default="")
    retrieval_setting: RetrievalSetting = _Field(
        description="Knowledge’s retrieval parameters", default=RetrievalSetting()
    )
    metadata_condition: Optional[MetadataCondition] = _Field(description="Original array filtering", default=None)

    def to_protobuf(self) -> _message.Message:
        """Convert Pydantic model to protobuf message"""
        _proto = pool.FindMessageTypeByName("raghub_interfaces.RetrievalRequest")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls, src: _message.Message) -> "RetrievalRequest":
        """Convert protobuf message to Pydantic model"""
        return protobuf2model(cls, src)


class RetrievalResponseRecord(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    content: str = _Field(description="Contains a chunk of text from a data source in the knowledge base.", default="")
    score: float = _Field(description="The score of relevance of the result to the query, scope: 0~1", default=0.0)
    title: str = _Field(description="Document title", default="")
    metadata: Optional[Dict[str, Any]] = _Field(
        description="Contains metadata attributes and their values for the document in the data source.", default=None
    )

    def to_protobuf(self) -> _message.Message:
        """Convert Pydantic model to protobuf message"""
        _proto = pool.FindMessageTypeByName("raghub_interfaces.RetrievalResponseRecord")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls, src: _message.Message) -> "RetrievalResponseRecord":
        """Convert protobuf message to Pydantic model"""
        return protobuf2model(cls, src)


class RetrievalResponseError(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    error_code: int = _Field(description="Error code", default=1001)
    error_msg: str = _Field(description="The description of API exception", default="")

    def to_protobuf(self) -> _message.Message:
        """Convert Pydantic model to protobuf message"""
        _proto = pool.FindMessageTypeByName("raghub_interfaces.RetrievalResponseError")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls, src: _message.Message) -> "RetrievalResponseError":
        """Convert protobuf message to Pydantic model"""
        return protobuf2model(cls, src)


class RetrievalResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    records: List[RetrievalResponseRecord] = _Field(
        description="A list of records from querying the knowledge base.", default=None
    )
    error: Optional[RetrievalResponseError] = _Field(description="Error information", default=None)
    request_id: Optional[str] = _Field(description="Request ID for tracking", default="")

    def to_protobuf(self) -> _message.Message:
        """Convert Pydantic model to protobuf message"""
        _proto = pool.FindMessageTypeByName("raghub_interfaces.RetrievalResponse")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls, src: _message.Message) -> "RetrievalResponse":
        """Convert protobuf message to Pydantic model"""
        return protobuf2model(cls, src)


class CreateIndexRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    unique_name: str = _Field(description="Name of the index", default="")

    def to_protobuf(self) -> _message.Message:
        """Convert Pydantic model to protobuf message"""
        _proto = pool.FindMessageTypeByName("raghub_interfaces.CreateIndexRequest")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls, src: _message.Message) -> "CreateIndexRequest":
        """Convert protobuf message to Pydantic model"""
        return protobuf2model(cls, src)


class CreateIndexResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    unique_name: str = _Field(description="Name of the index", default="")

    def to_protobuf(self) -> _message.Message:
        """Convert Pydantic model to protobuf message"""
        _proto = pool.FindMessageTypeByName("raghub_interfaces.CreateIndexResponse")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls, src: _message.Message) -> "CreateIndexResponse":
        """Convert protobuf message to Pydantic model"""
        return protobuf2model(cls, src)


class ChatRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    knowledge_id: str = _Field(description="knowledge’s unique ID", default="")
    question: str = _Field(description="User’s question", default="")
    retrieval_setting: RetrievalSetting = _Field(
        description="Knowledge’s retrieval parameters", default=RetrievalSetting()
    )
    metadata_condition: Optional[MetadataCondition] = _Field(description="Original array filtering", default=None)

    def to_protobuf(self) -> _message.Message:
        """Convert Pydantic model to protobuf message"""
        _proto = pool.FindMessageTypeByName("raghub_interfaces.ChatRequest")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls, src: _message.Message) -> "ChatRequest":
        """Convert protobuf message to Pydantic model"""
        return protobuf2model(cls, src)


class ChatResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    answer: str = _Field(description="The answer to the question based on the retrieved knowledge", default="")
    records: List[RetrievalResponseRecord] = _Field(
        description="A list of records used to generate the answer", default=None
    )
    error: Optional[RetrievalResponseError] = _Field(description="Error information", default=None)

    def to_protobuf(self) -> _message.Message:
        """Convert Pydantic model to protobuf message"""
        _proto = pool.FindMessageTypeByName("raghub_interfaces.ChatResponse")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls, src: _message.Message) -> "ChatResponse":
        """Convert protobuf message to Pydantic model"""
        return protobuf2model(cls, src)


class RAGDocument(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    content: str = _Field(description="Content of the document", default="")
    title: str = _Field(description="Title of the document", default="")
    metadata: Optional[Dict[str, Any]] = _Field(
        description="Metadata attributes and their values for the document", default=None
    )
    type: Optional[str] = _Field(description="Type of the document, e.g., 'text', 'image'", default="text")
    source: Optional[str] = _Field(
        description="Source of the document, e.g., 'knowledge_base', 'external'", default="knowledge_base"
    )

    def to_protobuf(self) -> _message.Message:
        """Convert Pydantic model to protobuf message"""
        _proto = pool.FindMessageTypeByName("raghub_interfaces.RAGDocument")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls, src: _message.Message) -> "RAGDocument":
        """Convert protobuf message to Pydantic model"""
        return protobuf2model(cls, src)


class AddDocumentsRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    knowledge_id: str = _Field(description="knowledge’s unique ID", default="")
    documents: List[RAGDocument] = _Field(description="List of documents to be added", default=None)

    def to_protobuf(self) -> _message.Message:
        """Convert Pydantic model to protobuf message"""
        _proto = pool.FindMessageTypeByName("raghub_interfaces.AddDocumentsRequest")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls, src: _message.Message) -> "AddDocumentsRequest":
        """Convert protobuf message to Pydantic model"""
        return protobuf2model(cls, src)


class AddDocumentsResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    documents: List[RAGDocument] = _Field(description="List of documents that were added", default=None)
    error: Optional[RetrievalResponseError] = _Field(description="Error information", default=None)
    request_id: Optional[str] = _Field(description="Request ID for tracking", default="")

    def to_protobuf(self) -> _message.Message:
        """Convert Pydantic model to protobuf message"""
        _proto = pool.FindMessageTypeByName("raghub_interfaces.AddDocumentsResponse")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls, src: _message.Message) -> "AddDocumentsResponse":
        """Convert protobuf message to Pydantic model"""
        return protobuf2model(cls, src)


class DeleteDocumentsRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    knowledge_id: str = _Field(description="knowledge’s unique ID", default="")
    document_ids: List[str] = _Field(description="List of document IDs to be deleted", default="")

    def to_protobuf(self) -> _message.Message:
        """Convert Pydantic model to protobuf message"""
        _proto = pool.FindMessageTypeByName("raghub_interfaces.DeleteDocumentsRequest")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls, src: _message.Message) -> "DeleteDocumentsRequest":
        """Convert protobuf message to Pydantic model"""
        return protobuf2model(cls, src)


class DeleteDocumentsResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    deleted_ids: List[str] = _Field(description="List of document IDs that were successfully deleted", default="")
    error: Optional[RetrievalResponseError] = _Field(description="Error information", default=None)
    request_id: Optional[str] = _Field(description="Request ID for tracking", default="")

    def to_protobuf(self) -> _message.Message:
        """Convert Pydantic model to protobuf message"""
        _proto = pool.FindMessageTypeByName("raghub_interfaces.DeleteDocumentsResponse")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls, src: _message.Message) -> "DeleteDocumentsResponse":
        """Convert protobuf message to Pydantic model"""
        return protobuf2model(cls, src)
