from google.api import annotations_pb2 as _annotations_pb2
from google.protobuf import any_pb2 as _any_pb2
from google.protobuf import empty_pb2 as _empty_pb2
from protobuf_pydantic_gen import pydantic_pb2 as _pydantic_pb2
import chat_pb2 as _chat_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class MetadataConditionItem(_message.Message):
    __slots__ = ("name", "comparison_operator", "value")
    NAME_FIELD_NUMBER: _ClassVar[int]
    COMPARISON_OPERATOR_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    name: _containers.RepeatedScalarFieldContainer[str]
    comparison_operator: str
    value: str
    def __init__(self, name: _Optional[_Iterable[str]] = ..., comparison_operator: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...

class MetadataCondition(_message.Message):
    __slots__ = ("logical_operator", "conditions")
    LOGICAL_OPERATOR_FIELD_NUMBER: _ClassVar[int]
    CONDITIONS_FIELD_NUMBER: _ClassVar[int]
    logical_operator: str
    conditions: _containers.RepeatedCompositeFieldContainer[MetadataConditionItem]
    def __init__(self, logical_operator: _Optional[str] = ..., conditions: _Optional[_Iterable[_Union[MetadataConditionItem, _Mapping]]] = ...) -> None: ...

class RetrievalRequest(_message.Message):
    __slots__ = ("knowledge_id", "query", "retrieval_setting", "metadata_condition")
    KNOWLEDGE_ID_FIELD_NUMBER: _ClassVar[int]
    QUERY_FIELD_NUMBER: _ClassVar[int]
    RETRIEVAL_SETTING_FIELD_NUMBER: _ClassVar[int]
    METADATA_CONDITION_FIELD_NUMBER: _ClassVar[int]
    knowledge_id: str
    query: str
    retrieval_setting: _chat_pb2.RetrievalSetting
    metadata_condition: MetadataCondition
    def __init__(self, knowledge_id: _Optional[str] = ..., query: _Optional[str] = ..., retrieval_setting: _Optional[_Union[_chat_pb2.RetrievalSetting, _Mapping]] = ..., metadata_condition: _Optional[_Union[MetadataCondition, _Mapping]] = ...) -> None: ...

class RetrievalResponseRecord(_message.Message):
    __slots__ = ("content", "score", "title", "metadata")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: _any_pb2.Any
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[_any_pb2.Any, _Mapping]] = ...) -> None: ...
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    TITLE_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    content: str
    score: float
    title: str
    metadata: _containers.MessageMap[str, _any_pb2.Any]
    def __init__(self, content: _Optional[str] = ..., score: _Optional[float] = ..., title: _Optional[str] = ..., metadata: _Optional[_Mapping[str, _any_pb2.Any]] = ...) -> None: ...

class RetrievalResponseError(_message.Message):
    __slots__ = ("error_code", "error_msg")
    ERROR_CODE_FIELD_NUMBER: _ClassVar[int]
    ERROR_MSG_FIELD_NUMBER: _ClassVar[int]
    error_code: int
    error_msg: str
    def __init__(self, error_code: _Optional[int] = ..., error_msg: _Optional[str] = ...) -> None: ...

class RetrievalResponse(_message.Message):
    __slots__ = ("records", "error", "request_id")
    RECORDS_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    records: _containers.RepeatedCompositeFieldContainer[RetrievalResponseRecord]
    error: RetrievalResponseError
    request_id: str
    def __init__(self, records: _Optional[_Iterable[_Union[RetrievalResponseRecord, _Mapping]]] = ..., error: _Optional[_Union[RetrievalResponseError, _Mapping]] = ..., request_id: _Optional[str] = ...) -> None: ...

class CreateIndexRequest(_message.Message):
    __slots__ = ("unique_name",)
    UNIQUE_NAME_FIELD_NUMBER: _ClassVar[int]
    unique_name: str
    def __init__(self, unique_name: _Optional[str] = ...) -> None: ...

class CreateIndexResponse(_message.Message):
    __slots__ = ("unique_name",)
    UNIQUE_NAME_FIELD_NUMBER: _ClassVar[int]
    unique_name: str
    def __init__(self, unique_name: _Optional[str] = ...) -> None: ...

class ChatRequest(_message.Message):
    __slots__ = ("knowledge_id", "question", "retrieval_setting", "metadata_condition")
    KNOWLEDGE_ID_FIELD_NUMBER: _ClassVar[int]
    QUESTION_FIELD_NUMBER: _ClassVar[int]
    RETRIEVAL_SETTING_FIELD_NUMBER: _ClassVar[int]
    METADATA_CONDITION_FIELD_NUMBER: _ClassVar[int]
    knowledge_id: str
    question: str
    retrieval_setting: _chat_pb2.RetrievalSetting
    metadata_condition: MetadataCondition
    def __init__(self, knowledge_id: _Optional[str] = ..., question: _Optional[str] = ..., retrieval_setting: _Optional[_Union[_chat_pb2.RetrievalSetting, _Mapping]] = ..., metadata_condition: _Optional[_Union[MetadataCondition, _Mapping]] = ...) -> None: ...

class ChatResponse(_message.Message):
    __slots__ = ("answer", "records", "error")
    ANSWER_FIELD_NUMBER: _ClassVar[int]
    RECORDS_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    answer: str
    records: _containers.RepeatedCompositeFieldContainer[RetrievalResponseRecord]
    error: RetrievalResponseError
    def __init__(self, answer: _Optional[str] = ..., records: _Optional[_Iterable[_Union[RetrievalResponseRecord, _Mapping]]] = ..., error: _Optional[_Union[RetrievalResponseError, _Mapping]] = ...) -> None: ...

class RAGDocument(_message.Message):
    __slots__ = ("content", "title", "metadata", "type", "source")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: _any_pb2.Any
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[_any_pb2.Any, _Mapping]] = ...) -> None: ...
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    TITLE_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    content: str
    title: str
    metadata: _containers.MessageMap[str, _any_pb2.Any]
    type: str
    source: str
    def __init__(self, content: _Optional[str] = ..., title: _Optional[str] = ..., metadata: _Optional[_Mapping[str, _any_pb2.Any]] = ..., type: _Optional[str] = ..., source: _Optional[str] = ...) -> None: ...

class AddDocumentsRequest(_message.Message):
    __slots__ = ("knowledge_id", "documents")
    KNOWLEDGE_ID_FIELD_NUMBER: _ClassVar[int]
    DOCUMENTS_FIELD_NUMBER: _ClassVar[int]
    knowledge_id: str
    documents: _containers.RepeatedCompositeFieldContainer[RAGDocument]
    def __init__(self, knowledge_id: _Optional[str] = ..., documents: _Optional[_Iterable[_Union[RAGDocument, _Mapping]]] = ...) -> None: ...

class AddDocumentsResponse(_message.Message):
    __slots__ = ("documents", "error", "request_id")
    DOCUMENTS_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    documents: _containers.RepeatedCompositeFieldContainer[RAGDocument]
    error: RetrievalResponseError
    request_id: str
    def __init__(self, documents: _Optional[_Iterable[_Union[RAGDocument, _Mapping]]] = ..., error: _Optional[_Union[RetrievalResponseError, _Mapping]] = ..., request_id: _Optional[str] = ...) -> None: ...

class DeleteDocumentsRequest(_message.Message):
    __slots__ = ("knowledge_id", "document_ids")
    KNOWLEDGE_ID_FIELD_NUMBER: _ClassVar[int]
    DOCUMENT_IDS_FIELD_NUMBER: _ClassVar[int]
    knowledge_id: str
    document_ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, knowledge_id: _Optional[str] = ..., document_ids: _Optional[_Iterable[str]] = ...) -> None: ...

class DeleteDocumentsResponse(_message.Message):
    __slots__ = ("deleted_ids", "error", "request_id")
    DELETED_IDS_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    deleted_ids: _containers.RepeatedScalarFieldContainer[str]
    error: RetrievalResponseError
    request_id: str
    def __init__(self, deleted_ids: _Optional[_Iterable[str]] = ..., error: _Optional[_Union[RetrievalResponseError, _Mapping]] = ..., request_id: _Optional[str] = ...) -> None: ...

class HealthResponse(_message.Message):
    __slots__ = ("healthy", "message")
    HEALTHY_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    healthy: bool
    message: str
    def __init__(self, healthy: bool = ..., message: _Optional[str] = ...) -> None: ...
