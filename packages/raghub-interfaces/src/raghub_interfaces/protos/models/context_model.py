# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   context.py
@Time    :
@Desc    :
"""

from typing import Type

from google.protobuf import message as _message
from google.protobuf import message_factory
from protobuf_pydantic_gen.ext import PydanticModel, model2protobuf, pool, protobuf2model
from pydantic import BaseModel, ConfigDict


class RequestContextHeaders(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    def to_protobuf(self) -> _message.Message:
        _proto = pool.FindMessageTypeByName(".RequestContextHeaders")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls: Type[PydanticModel], src: _message.Message) -> PydanticModel:
        return protobuf2model(cls, src)
