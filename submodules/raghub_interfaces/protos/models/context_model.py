# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   context_model.py
@Time    :   2025-07-02 09:16:48
@Desc    :   Generated Pydantic models from protobuf definitions
"""

from google.protobuf import message as _message, message_factory
from protobuf_pydantic_gen.ext import model2protobuf, pool, protobuf2model
from pydantic import BaseModel, ConfigDict, Field as _Field
from typing import Type


class RequestContextHeaders(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    def to_protobuf(self) -> _message.Message:
        """Convert Pydantic model to protobuf message"""
        _proto = pool.FindMessageTypeByName("RequestContextHeaders")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls, src: _message.Message) -> "RequestContextHeaders":
        """Convert protobuf message to Pydantic model"""
        return protobuf2model(cls, src)
