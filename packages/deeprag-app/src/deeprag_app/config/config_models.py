from typing import Optional

from deeprag_core.config.base import BaseParameters
from deeprag_core.config.deeprag_config import DeepRAGConfig
from deeprag_core.utils.file.project import ProjectHelper


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

    # synonymy_edge_topk: int = 2047,  # Number of top-k synonyms to consider for each node
    # synonymy_edge_query_batch_size: int = 1000,  # Batch size for query embeddings during synonymy edge construction
    # synonymy_edge_key_batch_size: int = 1000,  # Batch size for key embeddings during synonymy edge construction
    # synonymy_edge_sim_threshold: float = 0.8,
    # passage_node_weight:float = 0.05,
    # linking_top_k: int = 10,  # Number of top-k documents to link to each node
    # embedding_prefix="entity_embeddings",
    # dspy_file_path: Optional[str] = None,


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


class DatabaseConfig(BaseParameters):
    """
    Configuration for Database
    """

    provider: str = BaseParameters.field(
        default="sqlite",
        description="Database Provider",
        tags=["database"],
    )
    url: str = BaseParameters.field(
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


class APPConfig(DeepRAGConfig):
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
