# RAGHub 配置项说明

## 日志配置 (LoggerConfig)
- **log_level**: 日志级别，默认为 "DEBUG"。标签：log。
- **log_dir**: 日志文件路径，默认为 "logs"。标签：log。

## 代理LLM配置 (ProxyLLMConfig)
- **api_key**: OpenAI API密钥，环境变量为 `OPENAI_API_KEY`。默认值为 None。标签：secret。
- **base_url**: OpenAI API基础URL。默认值为 None。标签：url。
- **model**: 使用的OpenAI模型名称，默认为 "gpt-3.5-turbo"。标签：model。
- **timeout**: 请求超时时间（秒），默认60秒。标签：timeout。
- **provider**: 嵌入模型提供者，默认为 "openai-proxy"。标签：provider。
- **temperature**: 模型生成的随机性度量，默认0.3。标签：temperature。

## 嵌入模型配置 (EmbbedingModelConfig)
继承自 `ProxyLLMConfig`，并添加了：
- **batch_size**: 批处理大小，默认32。标签：batch_size。
- **n_dims**: 嵌入维度数，默认为 None。标签：n_dims。
- **embedding_key_prefix**: 嵌入键前缀，默认 "embedding"。标签：embedding_key_prefix。

## RAG配置 (RAGConfig)
- **llm**: LLM配置，默认使用 `ProxyLLMConfig()`。标签：llm。
- **embbeding**: 嵌入配置，默认使用 `EmbbedingModelConfig()`。标签：embedding。
- **lang**: 系统语言，默认为 "en"。标签：lang。

## 向量存储配置 (VectorStorageConfig)
- **provider**: 向量存储提供者，默认 "chromadb"。标签：vector_storage。
- **collection_name**: 集合名称，默认 "default_collection"。标签：vector_storage。
- **persist_directory**: 存储持久化目录。标签：vector_storage。
- **host**: 主机地址，默认 "localhost"。标签：vector_storage。
- **port**: 端口号，默认8000。标签：vector_storage。
- **username**: 用户名，默认为 None。标签：vector_storage。
- **password**: 密码，默认为 None。标签：vector_storage。
- **token**: 认证令牌，默认为 None。标签：vector_storage。
- **embedding_key_prefix**: 嵌入键前缀，默认 "embedding"。标签：vector_storage。

## 图存储配置 (GraphStorageConfig)
- **provider**: 图存储提供者，默认 "igraph"。标签：graph_storage。
- **graph_path**: 图文件路径。标签：graph_storage。
- **url**: 图存储URL，默认 "http://localhost:8000"。标签：graph_storage。
- **username**: 用户名，默认为 None。标签：graph_storage。
- **password**: 密码，默认为 None。标签：graph_storage。
- **database**: 数据库名称，默认为 None。标签：graph_storage。

## 应用程序配置 (RAGHubConfig)
- **app_name**: 应用名称，默认 "RAGHub"。标签：app。
- **app_version**: 应用版本，默认 "0.1.0"。标签：app。
- **app_description**: 应用描述，默认 "RAGHub Application"。标签：app。
- **logger**: 日志配置，默认 `LoggerConfig()`。标签：logger。
- **rag**: RAG配置，默认 `RAGConfig()`。标签：rag。
- **vector_storage**: 向量存储配置，默认 `VectorStorageConfig()`。标签：vector_storage。
- **graph**: 图配置，默认 `GraphStorageConfig()`。标签：graph。

## 搜索引擎配置 (SearchEngineConfig)
- **provider**: 搜索引擎提供者，默认 "elasticsearch"。标签：search。
- **host**: 主机地址，默认 "localhost"。标签：search。
- **port**: 端口号，默认9200。标签：search。
- **index_name_prefix**: 索引名称前缀，默认 "raghub_index"。标签：search。
- **use_ssl**: 是否使用SSL，默认 False。标签：search。
- **verify_certs**: 是否验证SSL证书，默认 False。标签：search。

## 数据库配置 (DatabaseConfig)
- **provider**: 数据库提供者，默认 "sqlite"。标签：database。
- **db_url**: 数据库连接URL，默认 "sqlite:///app.db"。标签：database。

## 缓存配置 (CacheConfig)
- **provider**: 缓存提供者，默认 "memory"。标签：cache。
- **cache_dir**: 缓存目录。标签：cache。
- **host**: 主机地址，默认 "localhost"。标签：cache。
- **port**: 端口号，默认6379。标签：cache。
- **auth**: 认证信息，默认为 None。标签：cache。


