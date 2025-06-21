from google.protobuf import any_pb2 as _any_pb2
from protobuf_pydantic_gen import pydantic_pb2 as _pydantic_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ChatMessage(_message.Message):
    __slots__ = ("role", "content", "name")
    ROLE_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    role: str
    content: str
    name: str
    def __init__(self, role: _Optional[str] = ..., content: _Optional[str] = ..., name: _Optional[str] = ...) -> None: ...

class RetrievalSetting(_message.Message):
    __slots__ = ("top_k", "score_threshold")
    TOP_K_FIELD_NUMBER: _ClassVar[int]
    SCORE_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    top_k: int
    score_threshold: float
    def __init__(self, top_k: _Optional[int] = ..., score_threshold: _Optional[float] = ...) -> None: ...

class CreateChatCompletionRequest(_message.Message):
    __slots__ = ("model", "messages", "max_tokens", "temperature", "metadata", "user", "stream", "response_format", "knowledge_id", "retrieval_setting")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: _any_pb2.Any
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[_any_pb2.Any, _Mapping]] = ...) -> None: ...
    MODEL_FIELD_NUMBER: _ClassVar[int]
    MESSAGES_FIELD_NUMBER: _ClassVar[int]
    MAX_TOKENS_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    USER_FIELD_NUMBER: _ClassVar[int]
    STREAM_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_FORMAT_FIELD_NUMBER: _ClassVar[int]
    KNOWLEDGE_ID_FIELD_NUMBER: _ClassVar[int]
    RETRIEVAL_SETTING_FIELD_NUMBER: _ClassVar[int]
    model: str
    messages: _containers.RepeatedCompositeFieldContainer[ChatMessage]
    max_tokens: int
    temperature: float
    metadata: _containers.MessageMap[str, _any_pb2.Any]
    user: str
    stream: bool
    response_format: str
    knowledge_id: str
    retrieval_setting: RetrievalSetting
    def __init__(self, model: _Optional[str] = ..., messages: _Optional[_Iterable[_Union[ChatMessage, _Mapping]]] = ..., max_tokens: _Optional[int] = ..., temperature: _Optional[float] = ..., metadata: _Optional[_Mapping[str, _any_pb2.Any]] = ..., user: _Optional[str] = ..., stream: bool = ..., response_format: _Optional[str] = ..., knowledge_id: _Optional[str] = ..., retrieval_setting: _Optional[_Union[RetrievalSetting, _Mapping]] = ...) -> None: ...

class ChatCompletionChoice(_message.Message):
    __slots__ = ("index", "message", "finish_reason")
    INDEX_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    FINISH_REASON_FIELD_NUMBER: _ClassVar[int]
    index: str
    message: ChatMessage
    finish_reason: float
    def __init__(self, index: _Optional[str] = ..., message: _Optional[_Union[ChatMessage, _Mapping]] = ..., finish_reason: _Optional[float] = ...) -> None: ...

class ChatCompletionUsage(_message.Message):
    __slots__ = ("prompt_tokens", "completion_tokens", "total_tokens")
    PROMPT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    COMPLETION_TOKENS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_TOKENS_FIELD_NUMBER: _ClassVar[int]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    def __init__(self, prompt_tokens: _Optional[int] = ..., completion_tokens: _Optional[int] = ..., total_tokens: _Optional[int] = ...) -> None: ...

class ChatCompletionReference(_message.Message):
    __slots__ = ("knowledge_id", "chunk_id", "content", "type", "metadata", "source")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: _any_pb2.Any
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[_any_pb2.Any, _Mapping]] = ...) -> None: ...
    KNOWLEDGE_ID_FIELD_NUMBER: _ClassVar[int]
    CHUNK_ID_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    knowledge_id: str
    chunk_id: str
    content: str
    type: str
    metadata: _containers.MessageMap[str, _any_pb2.Any]
    source: str
    def __init__(self, knowledge_id: _Optional[str] = ..., chunk_id: _Optional[str] = ..., content: _Optional[str] = ..., type: _Optional[str] = ..., metadata: _Optional[_Mapping[str, _any_pb2.Any]] = ..., source: _Optional[str] = ...) -> None: ...

class CreateChatCompletionResponse(_message.Message):
    __slots__ = ("id", "object", "created", "model", "choices", "metadata", "usage", "references")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: _any_pb2.Any
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[_any_pb2.Any, _Mapping]] = ...) -> None: ...
    ID_FIELD_NUMBER: _ClassVar[int]
    OBJECT_FIELD_NUMBER: _ClassVar[int]
    CREATED_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    CHOICES_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    USAGE_FIELD_NUMBER: _ClassVar[int]
    REFERENCES_FIELD_NUMBER: _ClassVar[int]
    id: str
    object: str
    created: int
    model: str
    choices: _containers.RepeatedCompositeFieldContainer[ChatCompletionChoice]
    metadata: _containers.MessageMap[str, _any_pb2.Any]
    usage: ChatCompletionUsage
    references: _containers.RepeatedCompositeFieldContainer[ChatCompletionReference]
    def __init__(self, id: _Optional[str] = ..., object: _Optional[str] = ..., created: _Optional[int] = ..., model: _Optional[str] = ..., choices: _Optional[_Iterable[_Union[ChatCompletionChoice, _Mapping]]] = ..., metadata: _Optional[_Mapping[str, _any_pb2.Any]] = ..., usage: _Optional[_Union[ChatCompletionUsage, _Mapping]] = ..., references: _Optional[_Iterable[_Union[ChatCompletionReference, _Mapping]]] = ...) -> None: ...
