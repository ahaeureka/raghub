# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: chat.proto
# Protobuf Python Version: 4.25.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import any_pb2 as google_dot_protobuf_dot_any__pb2
from protobuf_pydantic_gen import pydantic_pb2 as protobuf__pydantic__gen_dot_pydantic__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\nchat.proto\x12\x11raghub_interfaces\x1a\x19google/protobuf/any.proto\x1a$protobuf_pydantic_gen/pydantic.proto\"\xe2\x01\n\x0b\x43hatMessage\x12T\n\x04role\x18\x01 \x01(\tB@\xaa\xbb\x18<\n8Role of the message, e.g., \'user\', \'assistant\', \'system\'0\x01R\x04role\x12\x38\n\x07\x63ontent\x18\x02 \x01(\tB\x1e\xaa\xbb\x18\x1a\n\x16\x43ontent of the message0\x01R\x07\x63ontent\x12\x43\n\x04name\x18\x03 \x01(\tB/\xaa\xbb\x18+\n)Name of the message sender, if applicableR\x04name\"\xd4\x01\n\x10RetrievalSetting\x12\x44\n\x05top_k\x18\x01 \x01(\x05\x42.\xaa\xbb\x18*\n#Maximum number of retrieved results\x1a\x01\x35\x30\x01R\x05top_k\x12z\n\x0fscore_threshold\x18\x02 \x01(\x01\x42P\xaa\xbb\x18L\nCThe score limit of relevance of the result to the query, scope: 0~1\x1a\x03\x30.20\x01R\x0fscore_threshold\"\xca\x08\n\x1b\x43reateChatCompletionRequest\x12\x42\n\x05model\x18\x01 \x01(\tB,\xaa\xbb\x18(\n$The model to use for chat completion0\x01R\x05model\x12m\n\x08messages\x18\x02 \x03(\x0b\x32\x1e.raghub_interfaces.ChatMessageB1\xaa\xbb\x18-\n)List of messages in the chat conversation0\x01R\x08messages\x12`\n\nmax_tokens\x18\x03 \x01(\x05\x42@\xaa\xbb\x18<\n4Maximum number of tokens to generate in the response\x1a\x04\x31\x30\x30\x30R\nmax_tokens\x12Y\n\x0btemperature\x18\x04 \x01(\x01\x42\x37\xaa\xbb\x18\x33\n,Sampling temperature for response generation\x1a\x03\x30.7R\x0btemperature\x12\x92\x01\n\x08metadata\x18\x05 \x03(\x0b\x32<.raghub_interfaces.CreateChatCompletionRequest.MetadataEntryB8\xaa\xbb\x18\x34\n2Additional options for the chat completion requestR\x08metadata\x12\x44\n\x04user\x18\x06 \x01(\tB0\xaa\xbb\x18,\n*Identifier for the user making the requestR\x04user\x12H\n\x06stream\x18\x07 \x01(\x08\x42\x30\xaa\xbb\x18,\n#Whether to stream the response back\x1a\x05\x46\x61lseR\x06stream\x12\x62\n\x0fresponse_format\x18\x08 \x01(\tB8\xaa\xbb\x18\x34\n,Format of the response, e.g., \'json\', \'text\'\x1a\x04jsonR\x0fresponse_format\x12Z\n\x0cknowledge_id\x18\t \x01(\tB6\xaa\xbb\x18\x32\n0Knowledge base ID to use for the chat completionR\x0cknowledge_id\x12\x8e\x01\n\x11retrieval_setting\x18\n \x01(\x0b\x32#.raghub_interfaces.RetrievalSettingB;\xaa\xbb\x18\x37\n!Knowledge base retrieval settings\x1a\x12RetrievalSetting()R\x11retrieval_setting\x1a\x45\n\rMetadataEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12#\n\x05value\x18\x02 \x01(\x0b\x32\x14.google.protobuf.Any:\x02\x38\x01\"\xb1\x02\n\x14\x43hatCompletionChoice\x12\x41\n\x05index\x18\x01 \x01(\tB+\xaa\xbb\x18\'\n#Index of the choice in the response0\x01R\x05index\x12m\n\x07message\x18\x02 \x01(\x0b\x32\x1e.raghub_interfaces.ChatMessageB3\xaa\xbb\x18/\n+The message generated as part of the choice0\x01R\x07message\x12g\n\rfinish_reason\x18\x03 \x01(\x01\x42\x41\xaa\xbb\x18=\n;Reason for finishing the generation, e.g., \'stop\', \'length\'R\rfinish_reason\"\x95\x02\n\x13\x43hatCompletionUsage\x12L\n\rprompt_tokens\x18\x01 \x01(\x03\x42&\xaa\xbb\x18\"\n\x1eNumber of tokens in the prompt0\x01R\rprompt_tokens\x12X\n\x11\x63ompletion_tokens\x18\x02 \x01(\x03\x42*\xaa\xbb\x18&\n\"Number of tokens in the completion0\x01R\x11\x63ompletion_tokens\x12V\n\x0ctotal_tokens\x18\x03 \x01(\x03\x42\x32\xaa\xbb\x18.\n*Total number of tokens used in the request0\x01R\x0ctotal_tokens\"\xa0\x05\n\x17\x43hatCompletionReference\x12Z\n\x0cknowledge_id\x18\x01 \x01(\tB6\xaa\xbb\x18\x32\n.Knowledge base ID used for the chat completion0\x01R\x0cknowledge_id\x12O\n\x08\x63hunk_id\x18\x02 \x01(\tB3\xaa\xbb\x18/\n+ID of the chunk used in the chat completion0\x01R\x08\x63hunk_id\x12R\n\x07\x63ontent\x18\x03 \x01(\tB8\xaa\xbb\x18\x34\n0Content of the chunk used in the chat completion0\x01R\x07\x63ontent\x12L\n\x04type\x18\x04 \x01(\tB8\xaa\xbb\x18\x34\n,Type of the reference, e.g., \'text\', \'image\'\x1a\x04textR\x04type\x12\x83\x01\n\x08metadata\x18\x05 \x03(\x0b\x32\x38.raghub_interfaces.ChatCompletionReference.MetadataEntryB-\xaa\xbb\x18)\n\'Additional metadata about the referenceR\x08metadata\x12i\n\x06source\x18\x06 \x01(\tBQ\xaa\xbb\x18M\n;Source of the reference, e.g., \'knowledge_base\', \'external\'\x1a\x0eknowledge_baseR\x06source\x1a\x45\n\rMetadataEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12#\n\x05value\x18\x02 \x01(\x0b\x32\x14.google.protobuf.Any:\x02\x38\x01\"\xc7\x07\n\x1c\x43reateChatCompletionResponse\x12I\n\x02id\x18\x01 \x01(\tB9\xaa\xbb\x18\x35\n1Unique identifier for the chat completion request0\x01R\x02id\x12P\n\x06object\x18\x02 \x01(\tB8\xaa\xbb\x18\x34\n0Type of object returned, e.g., \'chat.completion\'0\x01R\x06object\x12S\n\x07\x63reated\x18\x03 \x01(\x03\x42\x39\xaa\xbb\x18\x35\n1Timestamp of when the chat completion was created0\x01R\x07\x63reated\x12K\n\x05model\x18\x04 \x01(\tB5\xaa\xbb\x18\x31\n-Model used for generating the chat completion0\x01R\x05model\x12{\n\x07\x63hoices\x18\x05 \x03(\x0b\x32\'.raghub_interfaces.ChatCompletionChoiceB8\xaa\xbb\x18\x34\n0List of choices generated in the chat completion0\x01R\x07\x63hoices\x12\x97\x01\n\x08metadata\x18\x06 \x03(\x0b\x32=.raghub_interfaces.CreateChatCompletionResponse.MetadataEntryB<\xaa\xbb\x18\x38\n6Additional metadata about the chat completion responseR\x08metadata\x12v\n\x05usage\x18\x07 \x01(\x0b\x32&.raghub_interfaces.ChatCompletionUsageB8\xaa\xbb\x18\x34\n0Usage statistics for the chat completion request0\x01R\x05usage\x12\x91\x01\n\nreferences\x18\x08 \x03(\x0b\x32*.raghub_interfaces.ChatCompletionReferenceBE\xaa\xbb\x18\x41\n?References to knowledge base chunks used in the chat completionR\nreferences\x1a\x45\n\rMetadataEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12#\n\x05value\x18\x02 \x01(\x0b\x32\x14.google.protobuf.Any:\x02\x38\x01\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'chat_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_CHATMESSAGE'].fields_by_name['role']._options = None
  _globals['_CHATMESSAGE'].fields_by_name['role']._serialized_options = b'\252\273\030<\n8Role of the message, e.g., \'user\', \'assistant\', \'system\'0\001'
  _globals['_CHATMESSAGE'].fields_by_name['content']._options = None
  _globals['_CHATMESSAGE'].fields_by_name['content']._serialized_options = b'\252\273\030\032\n\026Content of the message0\001'
  _globals['_CHATMESSAGE'].fields_by_name['name']._options = None
  _globals['_CHATMESSAGE'].fields_by_name['name']._serialized_options = b'\252\273\030+\n)Name of the message sender, if applicable'
  _globals['_RETRIEVALSETTING'].fields_by_name['top_k']._options = None
  _globals['_RETRIEVALSETTING'].fields_by_name['top_k']._serialized_options = b'\252\273\030*\n#Maximum number of retrieved results\032\00150\001'
  _globals['_RETRIEVALSETTING'].fields_by_name['score_threshold']._options = None
  _globals['_RETRIEVALSETTING'].fields_by_name['score_threshold']._serialized_options = b'\252\273\030L\nCThe score limit of relevance of the result to the query, scope: 0~1\032\0030.20\001'
  _globals['_CREATECHATCOMPLETIONREQUEST_METADATAENTRY']._options = None
  _globals['_CREATECHATCOMPLETIONREQUEST_METADATAENTRY']._serialized_options = b'8\001'
  _globals['_CREATECHATCOMPLETIONREQUEST'].fields_by_name['model']._options = None
  _globals['_CREATECHATCOMPLETIONREQUEST'].fields_by_name['model']._serialized_options = b'\252\273\030(\n$The model to use for chat completion0\001'
  _globals['_CREATECHATCOMPLETIONREQUEST'].fields_by_name['messages']._options = None
  _globals['_CREATECHATCOMPLETIONREQUEST'].fields_by_name['messages']._serialized_options = b'\252\273\030-\n)List of messages in the chat conversation0\001'
  _globals['_CREATECHATCOMPLETIONREQUEST'].fields_by_name['max_tokens']._options = None
  _globals['_CREATECHATCOMPLETIONREQUEST'].fields_by_name['max_tokens']._serialized_options = b'\252\273\030<\n4Maximum number of tokens to generate in the response\032\0041000'
  _globals['_CREATECHATCOMPLETIONREQUEST'].fields_by_name['temperature']._options = None
  _globals['_CREATECHATCOMPLETIONREQUEST'].fields_by_name['temperature']._serialized_options = b'\252\273\0303\n,Sampling temperature for response generation\032\0030.7'
  _globals['_CREATECHATCOMPLETIONREQUEST'].fields_by_name['metadata']._options = None
  _globals['_CREATECHATCOMPLETIONREQUEST'].fields_by_name['metadata']._serialized_options = b'\252\273\0304\n2Additional options for the chat completion request'
  _globals['_CREATECHATCOMPLETIONREQUEST'].fields_by_name['user']._options = None
  _globals['_CREATECHATCOMPLETIONREQUEST'].fields_by_name['user']._serialized_options = b'\252\273\030,\n*Identifier for the user making the request'
  _globals['_CREATECHATCOMPLETIONREQUEST'].fields_by_name['stream']._options = None
  _globals['_CREATECHATCOMPLETIONREQUEST'].fields_by_name['stream']._serialized_options = b'\252\273\030,\n#Whether to stream the response back\032\005False'
  _globals['_CREATECHATCOMPLETIONREQUEST'].fields_by_name['response_format']._options = None
  _globals['_CREATECHATCOMPLETIONREQUEST'].fields_by_name['response_format']._serialized_options = b'\252\273\0304\n,Format of the response, e.g., \'json\', \'text\'\032\004json'
  _globals['_CREATECHATCOMPLETIONREQUEST'].fields_by_name['knowledge_id']._options = None
  _globals['_CREATECHATCOMPLETIONREQUEST'].fields_by_name['knowledge_id']._serialized_options = b'\252\273\0302\n0Knowledge base ID to use for the chat completion'
  _globals['_CREATECHATCOMPLETIONREQUEST'].fields_by_name['retrieval_setting']._options = None
  _globals['_CREATECHATCOMPLETIONREQUEST'].fields_by_name['retrieval_setting']._serialized_options = b'\252\273\0307\n!Knowledge base retrieval settings\032\022RetrievalSetting()'
  _globals['_CHATCOMPLETIONCHOICE'].fields_by_name['index']._options = None
  _globals['_CHATCOMPLETIONCHOICE'].fields_by_name['index']._serialized_options = b'\252\273\030\'\n#Index of the choice in the response0\001'
  _globals['_CHATCOMPLETIONCHOICE'].fields_by_name['message']._options = None
  _globals['_CHATCOMPLETIONCHOICE'].fields_by_name['message']._serialized_options = b'\252\273\030/\n+The message generated as part of the choice0\001'
  _globals['_CHATCOMPLETIONCHOICE'].fields_by_name['finish_reason']._options = None
  _globals['_CHATCOMPLETIONCHOICE'].fields_by_name['finish_reason']._serialized_options = b'\252\273\030=\n;Reason for finishing the generation, e.g., \'stop\', \'length\''
  _globals['_CHATCOMPLETIONUSAGE'].fields_by_name['prompt_tokens']._options = None
  _globals['_CHATCOMPLETIONUSAGE'].fields_by_name['prompt_tokens']._serialized_options = b'\252\273\030\"\n\036Number of tokens in the prompt0\001'
  _globals['_CHATCOMPLETIONUSAGE'].fields_by_name['completion_tokens']._options = None
  _globals['_CHATCOMPLETIONUSAGE'].fields_by_name['completion_tokens']._serialized_options = b'\252\273\030&\n\"Number of tokens in the completion0\001'
  _globals['_CHATCOMPLETIONUSAGE'].fields_by_name['total_tokens']._options = None
  _globals['_CHATCOMPLETIONUSAGE'].fields_by_name['total_tokens']._serialized_options = b'\252\273\030.\n*Total number of tokens used in the request0\001'
  _globals['_CHATCOMPLETIONREFERENCE_METADATAENTRY']._options = None
  _globals['_CHATCOMPLETIONREFERENCE_METADATAENTRY']._serialized_options = b'8\001'
  _globals['_CHATCOMPLETIONREFERENCE'].fields_by_name['knowledge_id']._options = None
  _globals['_CHATCOMPLETIONREFERENCE'].fields_by_name['knowledge_id']._serialized_options = b'\252\273\0302\n.Knowledge base ID used for the chat completion0\001'
  _globals['_CHATCOMPLETIONREFERENCE'].fields_by_name['chunk_id']._options = None
  _globals['_CHATCOMPLETIONREFERENCE'].fields_by_name['chunk_id']._serialized_options = b'\252\273\030/\n+ID of the chunk used in the chat completion0\001'
  _globals['_CHATCOMPLETIONREFERENCE'].fields_by_name['content']._options = None
  _globals['_CHATCOMPLETIONREFERENCE'].fields_by_name['content']._serialized_options = b'\252\273\0304\n0Content of the chunk used in the chat completion0\001'
  _globals['_CHATCOMPLETIONREFERENCE'].fields_by_name['type']._options = None
  _globals['_CHATCOMPLETIONREFERENCE'].fields_by_name['type']._serialized_options = b'\252\273\0304\n,Type of the reference, e.g., \'text\', \'image\'\032\004text'
  _globals['_CHATCOMPLETIONREFERENCE'].fields_by_name['metadata']._options = None
  _globals['_CHATCOMPLETIONREFERENCE'].fields_by_name['metadata']._serialized_options = b'\252\273\030)\n\'Additional metadata about the reference'
  _globals['_CHATCOMPLETIONREFERENCE'].fields_by_name['source']._options = None
  _globals['_CHATCOMPLETIONREFERENCE'].fields_by_name['source']._serialized_options = b'\252\273\030M\n;Source of the reference, e.g., \'knowledge_base\', \'external\'\032\016knowledge_base'
  _globals['_CREATECHATCOMPLETIONRESPONSE_METADATAENTRY']._options = None
  _globals['_CREATECHATCOMPLETIONRESPONSE_METADATAENTRY']._serialized_options = b'8\001'
  _globals['_CREATECHATCOMPLETIONRESPONSE'].fields_by_name['id']._options = None
  _globals['_CREATECHATCOMPLETIONRESPONSE'].fields_by_name['id']._serialized_options = b'\252\273\0305\n1Unique identifier for the chat completion request0\001'
  _globals['_CREATECHATCOMPLETIONRESPONSE'].fields_by_name['object']._options = None
  _globals['_CREATECHATCOMPLETIONRESPONSE'].fields_by_name['object']._serialized_options = b'\252\273\0304\n0Type of object returned, e.g., \'chat.completion\'0\001'
  _globals['_CREATECHATCOMPLETIONRESPONSE'].fields_by_name['created']._options = None
  _globals['_CREATECHATCOMPLETIONRESPONSE'].fields_by_name['created']._serialized_options = b'\252\273\0305\n1Timestamp of when the chat completion was created0\001'
  _globals['_CREATECHATCOMPLETIONRESPONSE'].fields_by_name['model']._options = None
  _globals['_CREATECHATCOMPLETIONRESPONSE'].fields_by_name['model']._serialized_options = b'\252\273\0301\n-Model used for generating the chat completion0\001'
  _globals['_CREATECHATCOMPLETIONRESPONSE'].fields_by_name['choices']._options = None
  _globals['_CREATECHATCOMPLETIONRESPONSE'].fields_by_name['choices']._serialized_options = b'\252\273\0304\n0List of choices generated in the chat completion0\001'
  _globals['_CREATECHATCOMPLETIONRESPONSE'].fields_by_name['metadata']._options = None
  _globals['_CREATECHATCOMPLETIONRESPONSE'].fields_by_name['metadata']._serialized_options = b'\252\273\0308\n6Additional metadata about the chat completion response'
  _globals['_CREATECHATCOMPLETIONRESPONSE'].fields_by_name['usage']._options = None
  _globals['_CREATECHATCOMPLETIONRESPONSE'].fields_by_name['usage']._serialized_options = b'\252\273\0304\n0Usage statistics for the chat completion request0\001'
  _globals['_CREATECHATCOMPLETIONRESPONSE'].fields_by_name['references']._options = None
  _globals['_CREATECHATCOMPLETIONRESPONSE'].fields_by_name['references']._serialized_options = b'\252\273\030A\n?References to knowledge base chunks used in the chat completion'
  _globals['_CHATMESSAGE']._serialized_start=99
  _globals['_CHATMESSAGE']._serialized_end=325
  _globals['_RETRIEVALSETTING']._serialized_start=328
  _globals['_RETRIEVALSETTING']._serialized_end=540
  _globals['_CREATECHATCOMPLETIONREQUEST']._serialized_start=543
  _globals['_CREATECHATCOMPLETIONREQUEST']._serialized_end=1641
  _globals['_CREATECHATCOMPLETIONREQUEST_METADATAENTRY']._serialized_start=1572
  _globals['_CREATECHATCOMPLETIONREQUEST_METADATAENTRY']._serialized_end=1641
  _globals['_CHATCOMPLETIONCHOICE']._serialized_start=1644
  _globals['_CHATCOMPLETIONCHOICE']._serialized_end=1949
  _globals['_CHATCOMPLETIONUSAGE']._serialized_start=1952
  _globals['_CHATCOMPLETIONUSAGE']._serialized_end=2229
  _globals['_CHATCOMPLETIONREFERENCE']._serialized_start=2232
  _globals['_CHATCOMPLETIONREFERENCE']._serialized_end=2904
  _globals['_CHATCOMPLETIONREFERENCE_METADATAENTRY']._serialized_start=1572
  _globals['_CHATCOMPLETIONREFERENCE_METADATAENTRY']._serialized_end=1641
  _globals['_CREATECHATCOMPLETIONRESPONSE']._serialized_start=2907
  _globals['_CREATECHATCOMPLETIONRESPONSE']._serialized_end=3874
  _globals['_CREATECHATCOMPLETIONRESPONSE_METADATAENTRY']._serialized_start=1572
  _globals['_CREATECHATCOMPLETIONRESPONSE_METADATAENTRY']._serialized_end=1641
# @@protoc_insertion_point(module_scope)
