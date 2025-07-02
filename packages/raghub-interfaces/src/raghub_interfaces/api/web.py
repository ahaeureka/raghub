#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   web.py
@Time    :   2025/06/13 18:49:07
@Desc    :
"""

import asyncio
import time
import traceback
import uuid
from datetime import datetime

import psutil
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from raghub_interfaces.config import interface_config


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log all incoming requests for debugging purposes.
    """

    async def dispatch(self, request: Request, call_next):
        """
        Log request details and measure response time.
        """
        start_time = time.time()
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())

        # Log incoming request
        logger.info(
            f"🔍 Incoming request: {request.method} {request.url} "
            f"[request_id: {request_id}] "
            f"[client: {request.client.host if request.client else 'unknown'}]"
        )

        try:
            response = await call_next(request)

            # Calculate response time
            process_time = time.time() - start_time

            # Log successful response
            logger.info(
                f"✅ Request completed: {request.method} {request.url} "
                f"[request_id: {request_id}] "
                f"[status: {response.status_code}] "
                f"[time: {process_time:.3f}s]"
            )

            # Add timing header
            response.headers["x-process-time"] = str(process_time)

            return response

        except Exception as exc:
            # Calculate error time
            process_time = time.time() - start_time

            # Log failed request
            logger.error(
                f"❌ Request failed: {request.method} {request.url} "
                f"[request_id: {request_id}] "
                f"[error: {type(exc).__name__}: {str(exc)}] "
                f"[time: {process_time:.3f}s]",
                exc_info=True,
            )

            # Re-raise to let other handlers catch it
            raise


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add a unique request ID to each incoming request.
    This helps with request tracing and debugging.
    """

    async def dispatch(self, request: Request, call_next):
        """
        Process the request and add a request ID.

        Args:
            request: The incoming request
            call_next: The next middleware or route handler

        Returns:
            Response with added request ID header
        """
        # Generate a unique request ID if not provided
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())

        # Store request ID in request state for access in exception handlers
        request.state.request_id = request_id

        # Call the next middleware/handler
        response = await call_next(request)

        # Add request ID to response headers for client tracking
        response.headers["x-request-id"] = request_id

        return response


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
        self._setup_middleware()
        self._setup_exception_handlers()
        self._setup_routes()

    def _setup_middleware(self):
        """
        Setup middleware for the FastAPI application.
        """
        # Add request ID middleware
        self._app.add_middleware(RequestIDMiddleware)
        self._app.add_middleware(RequestLoggingMiddleware)

    def _setup_exception_handlers(self):
        """
        Setup global exception handlers for the FastAPI application.
        This method registers exception handlers for different types of exceptions.
        """

        @self._app.exception_handler(StarletteHTTPException)
        async def http_exception_handler(request: Request, exc: StarletteHTTPException):
            """
            Handle HTTP exceptions (4xx and 5xx status codes).

            Args:
                request: The incoming request that caused the exception
                exc: The HTTP exception that was raised

            Returns:
                JSONResponse with structured error information
            """
            logger.warning(
                f"HTTP exception occurred: {exc.status_code} - {exc.detail} "
                f"for {request.method} {request.url} "
                f"[request_id: {getattr(request.state, 'request_id', 'unknown')}]"
            )

            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": {"code": exc.status_code, "message": exc.detail, "type": "http_error"},
                    "success": False,
                    "request_id": getattr(request.state, "request_id", None),
                },
            )

        @self._app.exception_handler(HTTPException)
        async def fastapi_http_exception_handler(request: Request, exc: HTTPException):
            """
            Handle FastAPI HTTP exceptions.

            Args:
                request: The incoming request that caused the exception
                exc: The FastAPI HTTP exception that was raised

            Returns:
                JSONResponse with structured error information
            """
            logger.warning(
                f"FastAPI HTTP exception occurred: {exc.status_code} - {exc.detail} "
                f"for {request.method} {request.url} "
                f"[request_id: {getattr(request.state, 'request_id', 'unknown')}]"
            )

            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": {
                        "code": exc.status_code,
                        "message": exc.detail,
                        "type": "validation_error" if exc.status_code == 422 else "http_error",
                    },
                    "success": False,
                    "request_id": getattr(request.state, "request_id", None),
                },
            )

        @self._app.exception_handler(ValueError)
        async def value_error_handler(request: Request, exc: ValueError):
            """
            Handle ValueError exceptions (typically from invalid input data).

            Args:
                request: The incoming request that caused the exception
                exc: The ValueError that was raised

            Returns:
                JSONResponse with structured error information
            """
            logger.error(
                f"ValueError occurred: {str(exc)} for {request.method} {request.url} "
                f"[request_id: {getattr(request.state, 'request_id', 'unknown')}]",
                exc_info=True,
            )

            return JSONResponse(
                status_code=400,
                content={
                    "error": {"code": 400, "message": f"Invalid input: {str(exc)}", "type": "value_error"},
                    "success": False,
                    "request_id": getattr(request.state, "request_id", None),
                },
            )

        @self._app.exception_handler(Exception)
        async def general_exception_handler(request: Request, exc: Exception):
            """
            Handle all other unhandled exceptions.
            This is a catch-all handler for any exception that wasn't handled by more specific handlers.

            Args:
                request: The incoming request that caused the exception
                exc: The exception that was raised

            Returns:
                JSONResponse with structured error information
            """
            # Log the full traceback for debugging
            logger.error(
                f"Unhandled exception occurred: {type(exc).__name__}: {str(exc)} "
                f"for {request.method} {request.url} "
                f"[request_id: {getattr(request.state, 'request_id', 'unknown')}]\n{traceback.format_exc()}"
            )

            # Don't expose internal error details in production
            error_message = "An internal server error occurred"
            if self._config and hasattr(self._config, "debug") and self._config.debug:
                error_message = f"{type(exc).__name__}: {str(exc)}"

            return JSONResponse(
                status_code=500,
                content={
                    "error": {"code": 500, "message": error_message, "type": "internal_error"},
                    "success": False,
                    "request_id": getattr(request.state, "request_id", None),
                },
            )

    def _setup_routes(self):
        """
        Setup routes for the FastAPI application.
        """

        @self._app.get("/health", tags=["RAG"])
        async def health_check():
            """
            Health check endpoint.
            Returns a simple JSON response to indicate that the service is up and running.
            """
            return {"status": "healthy"}

        @self._app.get("/diagnostic", tags=["RAG"])
        async def diagnostic_info():
            """
            Diagnostic endpoint.
            Returns detailed information about the server's health and metrics.
            """
            # Gather some basic metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_info = psutil.virtual_memory()
            disk_info = psutil.disk_usage("/")

            # Simulate some async work
            await asyncio.sleep(0)

            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "metrics": {
                    "cpu_usage": cpu_usage,
                    "memory_usage": memory_info.percent,
                    "disk_usage": disk_info.percent,
                },
            }

    async def initialize(self):
        """
        Initialize the web API.
        This method can be used to perform any startup tasks.
        """
        logger.info("WebAPI initialized with global exception handlers and middleware")

    @property
    def app(self) -> FastAPI:
        """
        Returns the FastAPI application instance.
        """
        return self._app
