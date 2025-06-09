from typing import Optional

from raghub_core.config.base import BaseParameters
from raghub_core.config.raghub_config import CacheConfig, DatabaseConfig, RAGHubConfig, SearchEngineConfig


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


class HippoRAGConfig(BaseParameters):
    """
    Configuration for HippoRAG
    """

    synonymy_edge_topk: int = BaseParameters.field(
        default=2047,
        description="Number of top-k synonyms to consider for each node",
        tags=["synonymy"],
    )
    synonymy_edge_query_batch_size: int = BaseParameters.field(
        default=1000,
        description="Batch size for query embeddings during synonymy edge construction",
        tags=["synonymy"],
    )
    synonymy_edge_key_batch_size: int = BaseParameters.field(
        default=1000,
        description="Batch size for key embeddings during synonymy edge construction",
        tags=["synonymy"],
    )
    synonymy_edge_sim_threshold: float = BaseParameters.field(
        default=0.8,
        description="Similarity threshold for synonymy edges",
        tags=["synonymy"],
    )
    passage_node_weight: float = BaseParameters.field(
        default=0.05,
        description="Weight of passage nodes in the graph",
        tags=["graph"],
    )
    linking_top_k: int = BaseParameters.field(
        default=10,
        description="Number of top-k documents to link to each node",
        tags=["linking"],
    )
    embedding_prefix: str = BaseParameters.field(
        default="entity_embeddings",
        description="Prefix for embedding files",
        tags=["embeddings"],
    )
    dspy_file_path: Optional[str] = BaseParameters.field(
        default=None,
        description="Path to the dspy file",
        tags=["dspy"],
    )
    storage_provider: str = BaseParameters.field(
        default="hipporag_storage_local",
        description="Storage provider for HippoRAG",
        tags=["storage"],
    )


class GraphRAGConfig(BaseParameters):
    dao_provider: str = BaseParameters.field(
        default="default_graph_rag_dao",
        description="DAO provider for GraphRAG",
        tags=["dao"],
    )


class APPConfig(RAGHubConfig):
    """
    配置类，用于存储应用程序相关的参数和设置.
    """

    database: DatabaseConfig = BaseParameters.field(
        default=DatabaseConfig(),
        description="Database Configuration",
        tags=["database"],
    )
    hipporag: HippoRAGConfig = BaseParameters.field(
        default=HippoRAGConfig(),
        description="HippoRAG Configuration",
        tags=["hipporag"],
    )
    cache: CacheConfig = BaseParameters.field(
        default=CacheConfig(),
        description="Cache Configuration",
        tags=["cache"],
    )
    search_engine: SearchEngineConfig = BaseParameters.field(
        default=SearchEngineConfig(),
        description="Search Engine Configuration",
        tags=["search_engine"],
    )
    graphrag: GraphRAGConfig = BaseParameters.field(
        default=GraphRAGConfig(),
        description="GraphRAG Configuration",
        tags=["graphrag"],
    )
