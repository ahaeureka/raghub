from typing import Optional

from raghub_core.config.base import BaseParameters
from raghub_core.utils.file.project import ProjectHelper


class LoggerConfig(BaseParameters):
    """
    配置类，用于存储日志相关的参数和设置.
    """

    # 日志级别
    log_level: str = BaseParameters.field(
        default="DEBUG",
        description="Log Level",
        tags=["log"],
    )
    # 日志文件路径
    log_dir: str = BaseParameters.field(
        default="logs",
        description="Log File Path",
        tags=["log"],
    )


class ProxyLLMConfig(BaseParameters):
    """ """

    api_key: Optional[str] = BaseParameters.field(
        default=None,
        description="OpenAI API Secret Key",
        tags=["secret"],
        env="OPENAI_API_KEY",
        env_description="OpenAI API Secret Key",
    )
    base_url: Optional[str] = BaseParameters.field(
        default=None,
        description="OpenAI API Base URL",
        tags=["url"],
    )
    model: str = BaseParameters.field(
        default="gpt-3.5-turbo",
        description="OpenAI Model Name",
        tags=["model"],
    )
    timeout: int = BaseParameters.field(
        default=60,
        description="OpenAI API Timeout in seconds",
        tags=["timeout"],
    )
    provider: str = BaseParameters.field(
        default="openai-proxy",
        description="Provider of the embedding model",
        tags=["provider"],
    )
    temperature: float = BaseParameters.field(
        default=0.3,
        description="Temperature for the OpenAI model",
        tags=["temperature"],
    )


class EmbbedingModelConfig(ProxyLLMConfig):
    batch_size: int = BaseParameters.field(
        default=32,
        description="Batch size for embedding",
        tags=["batch_size"],
    )
    n_dims: Optional[int] = BaseParameters.field(
        default=None,
        description="Number of dimensions for the embedding",
        tags=["n_dims"],
    )
    provider: str = BaseParameters.field(
        default="openai-proxy-embedding",
        description="Provider of the embedding model",
        tags=["provider"],
    )
    embedding_key_prefix: Optional[str] = BaseParameters.field(
        default="embedding",
        description="Prefix for the embedding key",
        tags=["embedding_key_prefix"],
    )


class RAGConfig(BaseParameters):
    """
    RAG（Retrieval-Augmented Generation)
    """

    llm: ProxyLLMConfig = BaseParameters.field(
        default=ProxyLLMConfig(),
        description="LLM Configuration",
        tags=["llm"],
    )
    embbeding: EmbbedingModelConfig = BaseParameters.field(
        default=EmbbedingModelConfig(),
        description="Embedding Configuration",
        tags=["embedding"],
    )
    lang: str = BaseParameters.field(
        default="en",
        description="Language for the RAG system",
        tags=["lang"],
    )


class VectorStorageConfig(BaseParameters):
    """
    配置类，用于存储向量存储的参数和设置.
    """

    provider: str = BaseParameters.field(
        default="chromadb",
        description="Vector storage provider",
        tags=["vector_storage"],
    )
    collection_name: str = BaseParameters.field(
        default="default_collection",
        description="Name of the collection in the vector storage",
        tags=["vector_storage"],
    )
    persist_directory: Optional[str] = BaseParameters.field(
        default=str(ProjectHelper.get_project_root() / "cache/chroma_db"),
        description="Directory to persist the vector storage",
        tags=["vector_storage"],
    )
    host: Optional[str] = BaseParameters.field(
        default="localhost",
        description="Host for the vector storage",
        tags=["vector_storage"],
    )
    port: Optional[int] = BaseParameters.field(
        default=8000,
        description="Port for the vector storage",
        tags=["vector_storage"],
    )
    username: Optional[str] = BaseParameters.field(
        default=None,
        description="Username for vector storage authentication",
        tags=["vector_storage"],
    )
    password: Optional[str] = BaseParameters.field(
        default=None,
        description="Password for vector storage authentication",
        tags=["vector_storage"],
    )
    token: Optional[str] = BaseParameters.field(
        default=None,
        description="Token for vector storage authentication",
        tags=["vector_storage"],
    )
    embedding_key_prefix: Optional[str] = BaseParameters.field(
        default="embedding",
        description="Prefix for the embedding key",
        tags=["vector_storage"],
    )
    # 其他向量存储相关的参数可以在这里添加
    # ...


class GraphStorageConfig(BaseParameters):
    """
    配置类，用于存储图存储的参数和设置.
    """

    provider: str = BaseParameters.field(
        default="igraph",
        description="Graph storage provider",
        tags=["graph_storage"],
    )
    graph_path: Optional[str] = BaseParameters.field(
        default=str(ProjectHelper.get_project_root() / "storage/graphs/default_graph.pkl"),
        description="Path to the graph file",
        tags=["graph_storage"],
    )
    url: Optional[str] = BaseParameters.field(
        default="http://localhost:8000",
        description="Graph storage URL",
        tags=["graph_storage"],
    )
    username: Optional[str] = BaseParameters.field(
        default=None,
        description="Username for graph storage authentication",
        tags=["graph_storage"],
    )
    password: Optional[str] = BaseParameters.field(
        default=None,
        description="Password for graph storage authentication",
        tags=["graph_storage"],
    )


class RAGHubConfig(BaseParameters):
    """
    配置类，用于存储应用程序的参数和设置.
    """

    # 应用程序名称
    app_name: str = BaseParameters.field(
        default="RAGHub",
        description="Application Name",
        tags=["app"],
    )
    # 应用程序版本
    app_version: str = BaseParameters.field(
        default="0.1.0",
        description="Application Version",
        tags=["app"],
    )
    # 应用程序描述
    app_description: str = BaseParameters.field(
        default="RAGHub Application",
        description="Application Description",
        tags=["app"],
    )

    logger: LoggerConfig = BaseParameters.field(
        default=LoggerConfig(),
        description="Logger Configuration",
        tags=["logger"],
    )
    rag: RAGConfig = BaseParameters.field(
        default=RAGConfig(),
        description="RAG Configuration",
        tags=["rag"],
    )
    vector_storage: VectorStorageConfig = BaseParameters.field(
        default=VectorStorageConfig(),
        description="Vector Storage Configuration",
        tags=["vector_storage"],
    )
    graph: GraphStorageConfig = BaseParameters.field(
        default=GraphStorageConfig(),
        description="Graph Configuration",
        tags=["graph"],
    )
    # 其他配置项...


class SearchEngineConfig(BaseParameters):
    """
    Configuration for Search Engine
    """

    provider: str = BaseParameters.field(
        default="elasticsearch",
        description="Search Engine Provider",
        tags=["search"],
    )
    host: str = BaseParameters.field(
        default="localhost",
        description="Search Engine Host",
        tags=["search"],
    )
    port: int = BaseParameters.field(
        default=9200,
        description="Search Engine Port",
        tags=["search"],
    )
    index_name_prefix: str = BaseParameters.field(
        default="raghub_index",
        description="Search Engine Index Name",
        tags=["search"],
    )
    username: Optional[str] = BaseParameters.field(
        default=None,
        description="Search Engine Username",
        tags=["search"],
    )
    password: Optional[str] = BaseParameters.field(
        default=None,
        description="Search Engine Password",
        tags=["search"],
    )
    use_ssl: bool = BaseParameters.field(
        default=False,
        description="Use SSL for Search Engine",
        tags=["search"],
    )
    verify_certs: bool = BaseParameters.field(
        default=False,
        description="Verify SSL certificates for Search Engine",
        tags=["search"],
    )


class DatabaseConfig(BaseParameters):
    """
    Configuration for Database
    """

    provider: str = BaseParameters.field(
        default="sqlite",
        description="Database Provider",
        tags=["database"],
    )
    db_url: str = BaseParameters.field(
        default="sqlite:///app.db",
        description="Database URL",
        tags=["database"],
    )


class CacheConfig(BaseParameters):
    """
    Configuration for Cache
    """

    provider: str = BaseParameters.field(
        default="memory",
        description="Cache Provider",
        tags=["cache"],
    )
    cache_dir: Optional[str] = BaseParameters.field(
        default=(ProjectHelper.get_project_root() / "cache").as_posix(),
        description="Cache Directory",
        tags=["cache"],
    )
    host: Optional[str] = BaseParameters.field(
        default="localhost",
        description="Cache Host",
        tags=["cache"],
    )
    port: Optional[int] = BaseParameters.field(
        default=6379,
        description="Cache Port",
        tags=["cache"],
    )
    auth: Optional[str] = BaseParameters.field(
        default=None,
        description="Cache Authentication",
        tags=["cache"],
    )
