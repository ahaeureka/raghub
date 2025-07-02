"""
Auto-generated gRPC FastAPI client
Generated from services.json
"""

import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Dict, Optional
from urllib.parse import urlparse, urlunparse

import httpx
from pydantic import ValidationError
from raghub_protos.models.chat_model import CreateChatCompletionRequest, CreateChatCompletionResponse

# Import all required models
from raghub_protos.models.rag_model import (
    AddDocumentsRequest,
    AddDocumentsResponse,
    CreateIndexRequest,
    CreateIndexResponse,
    DeleteDocumentsRequest,
    DeleteDocumentsResponse,
    RetrievalRequest,
    RetrievalResponse,
)

logger = logging.getLogger(__name__)


class RAGHubClient:
    """Generated gRPC FastAPI client with type safety"""

    def __init__(self, base_url: str, api_key: str = "", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _build_headers(self, extra_headers: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """Build request headers with authentication"""
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if extra_headers:
            headers.update(extra_headers)
        return headers

    def _build_websocket_uri(self, path: str) -> str:
        """Build WebSocket URI from HTTP base URL"""
        parsed = urlparse(self.base_url)
        ws_scheme = "wss" if parsed.scheme == "https" else "ws"
        return urlunparse((ws_scheme, parsed.netloc, path.lstrip("/"), parsed.params, parsed.query, parsed.fragment))

    # RAGService methods

    async def rag_service_retrieval(
        self, request: RetrievalRequest, headers: Optional[Dict[str, Any]] = None
    ) -> RetrievalResponse:
        """Retrieval - Unary RPC call"""
        url = f"{self.base_url}/v1/rag/retrieval"
        request_headers = self._build_headers(headers)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=request.model_dump(exclude_none=True), headers=request_headers)

            if response.status_code >= 400:
                raise httpx.HTTPStatusError(f"HTTP {response.status_code}", request=response.request, response=response)

            data = response.json()
            return RetrievalResponse(**data.get("data", data))

    async def rag_service_create_index(
        self, request: CreateIndexRequest, headers: Optional[Dict[str, Any]] = None
    ) -> CreateIndexResponse:
        """CreateIndex - Unary RPC call"""
        url = f"{self.base_url}/v1/rag/index"
        request_headers = self._build_headers(headers)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=request.model_dump(exclude_none=True), headers=request_headers)

            if response.status_code >= 400:
                raise httpx.HTTPStatusError(f"HTTP {response.status_code}", request=response.request, response=response)

            data = response.json()
            return CreateIndexResponse(**data.get("data", data))

    async def rag_service_chat(
        self, request: CreateChatCompletionRequest, headers: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[CreateChatCompletionResponse, None]:
        """Chat - Server streaming RPC call"""
        url = f"{self.base_url}/v1/rag/chat/completions"
        request_headers = self._build_headers(headers)
        request_headers["Accept"] = "text/event-stream"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST", url, json=request.model_dump(exclude_none=True), headers=request_headers
            ) as response:
                if response.status_code >= 400:
                    raise httpx.HTTPStatusError(
                        f"HTTP {response.status_code}", request=response.request, response=response
                    )
                try:
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]  # Remove "data: " prefix
                            if data_str.strip():
                                try:
                                    data = json.loads(data_str)
                                    yield CreateChatCompletionResponse(**data)
                                except (json.JSONDecodeError, ValidationError) as e:
                                    if data_str.startswith("data") and len(line) > 6:
                                        logger.error(f"Failed to parse SSE data: {e} with {line}")
                                        raise ValueError(f"Invalid SSE data: {line}") from e
                except asyncio.CancelledError:
                    logger.warning("Stream cancelled, closing connection.")
                    await response.aclose()
                    await client.aclose()
                except Exception as e:
                    logger.error(f"Error during streaming: {e}")
                    raise httpx.HTTPStatusError(f"Streaming error: {e}", request=response.request, response=response)

    async def rag_service_add_documents(
        self, request: AddDocumentsRequest, headers: Optional[Dict[str, Any]] = None
    ) -> AddDocumentsResponse:
        """AddDocuments - Unary RPC call"""
        url = f"{self.base_url}/v1/rag/documents"
        request_headers = self._build_headers(headers)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=request.model_dump(exclude_none=True), headers=request_headers)
            if response.status_code >= 400:
                raise httpx.HTTPStatusError(f"HTTP {response.status_code}", request=response.request, response=response)

            data = response.json()
            return AddDocumentsResponse(**data.get("data", data))

    async def rag_service_delete_documents(
        self, request: DeleteDocumentsRequest, headers: Optional[Dict[str, Any]] = None
    ) -> DeleteDocumentsResponse:
        """DeleteDocuments - Unary RPC call"""
        url = f"{self.base_url}/v1/rag/documents"
        request_headers = self._build_headers(headers)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.delete(url, params=request.model_dump(exclude_none=True), headers=request_headers)

            if response.status_code >= 400:
                print(f"HTTP {response.status_code} error: {response.text}")
                raise httpx.HTTPStatusError(f"HTTP {response.status_code}", request=response.request, response=response)

            data = response.json()
            return DeleteDocumentsResponse(**data.get("data", data))
