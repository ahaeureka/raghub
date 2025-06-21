from raghub_app.config.config_models import APPConfig
from raghub_core.config.base import BaseParameters


class ServerConfig(BaseParameters):
    name: str = BaseParameters.field(
        default="RAGHub Server",
        description="Name of the RAGHub server",
        tags=["server"],
    )
    address: str = BaseParameters.field(
        default="127.0.0.1",
        description="Address of the RAGHub server",
        tags=["server"],
    )
    port: int = BaseParameters.field(
        default=8000,
        description="Port of the RAGHub server",
        tags=["server"],
    )


class Interfaces(BaseParameters):
    rag_provider: str = BaseParameters.field(
        default="hipporag_app",
        description="RAG Provider",
        tags=["rag"],
    )
    server: ServerConfig = BaseParameters.field(
        default=ServerConfig(),
        description="Server Configuration",
        tags=["server"],
    )


class InerfaceConfig(APPConfig):
    interfaces: Interfaces = BaseParameters.field(
        default=Interfaces(),
        description="Interfaces Configuration",
        tags=["interfaces"],
    )
