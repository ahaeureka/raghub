from typing import AsyncIterator, List

import grpc
from loguru import logger
from protobuf_pydantic_gen.any_type_transformer import AnyTransformer
from raghub_app.apps.app_rag_base import BaseRAGApp
from raghub_core.schemas.chat_response import QAChatResponse
from raghub_core.schemas.document import Document
from raghub_core.schemas.graph_model import Namespace
from raghub_core.utils.class_meta import ClassFactory
from raghub_core.utils.misc import compute_mdhash_id

from raghub_interfaces.config.interface_config import InerfaceConfig
from raghub_interfaces.protos.models.rag_model import RAGDocument
from raghub_interfaces.protos.pb import chat_pb2, rag_pb2
from raghub_interfaces.protos.pb.rag_pb2_grpc import RAGServiceServicer
from raghub_interfaces.services.service_base import ServiceBase


class RAGServiceImpl(RAGServiceServicer, ServiceBase):
    """Implementation of the DifyRetrieveService."""

    name = "RAGService"

    def __init__(self, config: InerfaceConfig):
        """Initialize the DifyRetrieveServiceImpl."""
        super().__init__()
        self._config = config
        self.app: BaseRAGApp = ClassFactory.get_instance(
            config.interfaces.rag_provider,
            BaseRAGApp,
            config=config,
        )
        logger.debug(f"RAGServiceImpl initialized with {config.database.db_url}")

    async def initialize(self):
        """
        Initialize the DifyRetrieveService.
        This method should be overridden in subclasses to perform any necessary initialization.
        """
        # Here you can add any initialization logic needed for the service.
        await self.app.init()

    async def Retrieval(
        self, request: rag_pb2.RetrievalRequest, context: grpc.ServicerContext
    ) -> rag_pb2.RetrievalResponse:
        """Retrieve method implementation."""
        rsp = await self.app.retrieve(
            unique_name=request.knowledge_id,
            queries=[request.query],
            retrieve_top_k=request.retrieval_setting.top_k,
        )
        records: List[rag_pb2.RetrievalResponseRecord] = []
        items = rsp.get(request.query, [])
        for item in items:
            record = rag_pb2.RetrievalResponseRecord(
                content=item.document.content,
                score=item.score,
                title=item.document.metadata.get("title", ""),
                metadata=item.document.metadata,
            )
            records.append(record)
        response = rag_pb2.RetrievalResponse(
            records=records,
            request_id=self.get_request_id(context),
        )
        return response

    async def CreateIndex(
        self, request: rag_pb2.CreateIndexRequest, context: grpc.ServicerContext
    ) -> rag_pb2.CreateIndexResponse:
        """CreateIndex method implementation."""
        await self.app.create(request.unique_name)
        response = rag_pb2.CreateIndexResponse(unique_name=request.unique_name)
        return response

    async def DeleteDocuments(
        self, request: rag_pb2.DeleteDocumentsRequest, context
    ) -> rag_pb2.DeleteDocumentsResponse:
        """DeleteDocuments method implementation."""
        await self.app.delete(unique_name=request.knowledge_id, docs_to_delete=request.document_ids)
        response = rag_pb2.DeleteDocumentsResponse(deleted_ids=request.document_ids)
        return response

    async def Chat(
        self, request: chat_pb2.CreateChatCompletionRequest, context: grpc.ServicerContext
    ) -> AsyncIterator[chat_pb2.CreateChatCompletionResponse]:
        """Chat method implementation."""
        # async for ans in self.app.QA()
        metadata = request.metadata
        unique_name = request.knowledge_id or metadata.get("knowledge_id", "")
        if not unique_name:
            raise ValueError("knowledge_id is required in the request metadata or as a parameter.")
        messages = request.messages
        lasted_user_message = messages[-1] if messages else None
        if not lasted_user_message or lasted_user_message.role != "user":
            raise ValueError("The last message must be from the user.")
        query = lasted_user_message.content
        if not query:
            raise ValueError("The user message must contain a query.")
        default_settings = chat_pb2.RetrievalSetting()
        settings = request.retrieval_setting or metadata.get("retrieval_setting", default_settings)
        if isinstance(settings, dict):
            settings = chat_pb2.RetrievalSetting(**settings)
        index = 0
        # llm = None
        # if request.m
        # auth = self.get_auth(context)
        # if auth:
        #     auth = get_openai_key_from_auth(auth)
        # model = request.model
        # if model:
        #     _llm = ClassFactory.get_instance(
        #     "openai-proxy",
        #     BaseChat,
        #     model_name=config.rag.llm.model,
        #     api_key=config.rag.llm.api_key,
        #     base_url=config.rag.llm.base_url,
        #     temperature=config.rag.llm.temperature,
        #     timeout=config.rag.llm.timeout,
        # )
        # qa_resulst = self.app.QA(
        #     unique_name=unique_name,
        #     question=query,
        #     retrieve_top_k=settings.top_k,
        # )
        r = self.app.QA(
            unique_name=unique_name,
            question=query,
            retrieve_top_k=settings.top_k,
        )
        async for resp in r:
            if resp:
                ans: QAChatResponse = resp
                response_message = chat_pb2.ChatMessage(
                    role="assistant",
                    content=ans.answer,
                    name="",
                )
                response = chat_pb2.CreateChatCompletionResponse(
                    choices=[chat_pb2.ChatCompletionChoice(index=str(index), message=response_message)],
                    usage=chat_pb2.ChatCompletionUsage(
                        total_tokens=ans.tokens,
                    ),
                )
                index += 1
                logger.debug(f"Chat response: {response}")
                yield response

    async def AddDocuments(
        self, request: rag_pb2.AddDocumentsRequest, context: grpc.ServicerContext
    ) -> rag_pb2.AddDocumentsResponse:
        """AddDocuments method implementation."""
        docs: List[Document] = []
        rag_docs = request.documents
        for rag_doc in rag_docs:
            rag_doc_model = RAGDocument.from_protobuf(rag_doc)  # Ensure the RAGDocument is properly converted
            rag_doc_model.metadata["source"] = rag_doc.source
            rag_doc_model.metadata["type"] = rag_doc.type
            doc = Document(
                content=rag_doc.content,
                metadata=rag_doc_model.metadata,
                uid=compute_mdhash_id(request.knowledge_id, rag_doc.content, Namespace.DOC.value),
            )
            docs.append(doc)
        new_docs: List[Document] = await self.app.add_documents(unique_name=request.knowledge_id, texts=docs)
        # response = rag_pb2.AddDocumentsResponse()
        rsp_rag_docs: List[rag_pb2.RAGDocument] = []
        for doc in new_docs:
            logger.debug(f"Document added services: {doc}..")
            rag_doc = rag_pb2.RAGDocument(
                content=doc.content,
                metadata={key: AnyTransformer.any_type_to_protobuf(value) for key, value in doc.metadata.items()},
                source=doc.metadata.get("source", ""),
                type=doc.metadata.get("type", ""),
            )
            rsp_rag_docs.append(rag_doc)
        response = rag_pb2.AddDocumentsResponse(
            documents=rsp_rag_docs,
        )
        return response
