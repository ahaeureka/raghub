#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   web.py
@Time    :   2025/06/13 18:49:07
@Desc    :
"""

from fastapi import FastAPI

from raghub_interfaces.config import interface_config


class WebAPI:
    """
    Web API for Raghub.
    """

    def __init__(self, config: interface_config.InerfaceConfig):
        self._app = FastAPI(
            title="Raghub Interfaces API",
            description="API for Raghub Interfaces",
            version="1.0.0",
            openapi_tags=[
                {
                    "name": "RAG",
                    "description": "RAG related operations",
                },
            ],
        )
        self._config = config

    async def initialize(self):
        """
        Initialize the web API.
        This method can be used to perform any startup tasks.
        """
        pass

    @property
    def app(self) -> FastAPI:
        """
        Returns the FastAPI application instance.
        """
        return self._app
