# 接口配置文件文档  
**自动生成于 2025-06-26 05:27:51**  
**描述：配置参数说明**

# 根级配置

| 配置项 | 类型 | 必填 | 描述 | 默认值 |
|-------|------|------|------|--------|
| app_name | str | 是 | 应用名称 | "RAGHub" |
| app_version | str | 是 | 应用版本 | "0.1.0" |
| app_description | str | 是 | 应用描述 | "RAGHub应用程序" |

## [logger]  
**日志记录器配置**

| 配置项 | 类型 | 必填 | 描述 | 默认值 |
|-------|------|------|------|--------|
| log_level | str | 是 | 日志级别 | "DEBUG" |
| log_dir | str | 是 | 日志文件路径 | "logs" |

## [rag]  
**RAG系统配置**

| 配置项 | 类型 | 必填 | 描述 | 默认值 |
|-------|------|------|------|--------|
| lang | str | 是 | RAG系统使用语言 | "en" |

## [rag.llm]  
**大语言模型配置**

| 配置项 | 类型 | 必填 | 描述 | 默认值 |
|-------|------|------|------|--------|
| api_key | str | 否 | OpenAI API密钥 | null |
| base_url | str | 否 | OpenAI API基础地址 | null |
| model | str | 是 | OpenAI模型名称 | "gpt-3.5-turbo" |
| timeout | int | 是 | OpenAI API超时时间(秒) | 60 |
| provider | str | 是 | 嵌入模型供应商 | "openai-proxy" |
| temperature | float | 是 | OpenAI模型温度参数 | 0.3 |

## [rag.embbeding]  
**嵌入模型配置**

| 配置项 | 类型 | 必填 | 描述 | 默认值 |
|-------|------|------|------|--------|
| api_key | str | 否 | OpenAI API密钥 | null |
| base_url | str | 否 | OpenAI API基础地址 | null |
| model | str | 是 | OpenAI模型名称 | "gpt-3.5-turbo" |
| timeout | int | 是 | OpenAI API超时时间(秒) | 60 |
| provider | str | 是 | 嵌入模型供应商 | "openai-proxy-embedding" |
| temperature | float | 是 | OpenAI模型温度参数 | 0.3 |
| batch_size | int | 是 | 嵌入处理的批次大小 | 32 |
| n_dims | int | 否 | 嵌入维度数 | null |
| embedding_key_prefix | str | 否 | 嵌入键前缀 | "embedding" |

## [vector_storage]  
**向量存储配置**

| 配置项 | 类型 | 必填 | 描述 | 默认值 |
|-------|------|------|------|--------|
| provider | str | 是 | 向量存储供应商 | "chromadb" |
| collection_name | str | 是 | 向量存储集合名称 | "default_collection" |
| persist_directory | str | 否 | 向量存储持久化目录 | "/app/cache/chroma_db" |
| host | str | 否 | 向量存储主机地址 | "localhost" |
| port | int | 否 | 向量存储端口号 | 8000 |
| username | str | 否 | 向量存储认证用户名 | null |
| password | str | 否 | 向量存储认证密码 | null |
| token | str | 否 | 向量存储认证令牌 | null |
| embedding_key_prefix | str | 否 | 嵌入键前缀 | "embedding" |

## [graph]  
**图存储配置**

| 配置项 | 类型 | 必填 | 描述 | 默认值 |
|-------|------|------|------|--------|
| provider | str | 是 | 图存储供应商 | "igraph" |
| graph_path | str | 否 | 图文件存储路径 | "/app/storage/graphs/default_graph.pkl" |
| url | str | 否 | 图存储服务URL | "http://localhost:8000" |
| username | str | 否 | 图存储认证用户名 | null |
| password | str | 否 | 图存储认证密码 | null |
| database | str | 否 | 图存储数据库名称 | null |

## [database]  
**数据库配置**

| 配置项 | 类型 | 必填 | 描述 | 默认值 |
|-------|------|------|------|--------|
| provider | str | 是 | 数据库供应商 | "sqlite" |
| db_url | str | 是 | 数据库连接URL | "sqlite:///app.db" |

## [hipporag]  
**HippoRAG配置**

| 配置项 | 类型 | 必填 | 描述 | 默认值 |
|-------|------|------|------|--------|
| synonymy_edge_topk | int | 是 | 每个节点考虑的top-k同义词数量 | 2047 |
| synonymy_edge_query_batch_size | int | 是 | 同义词边构建时的查询嵌入批次大小 | 1000 |
| synonymy_edge_key_batch_size | int | 是 | 同义词边构建时的键嵌入批次大小 | 1000 |
| synonymy_edge_sim_threshold | float | 是 | 同义词边相似度阈值 | 0.8 |
| passage_node_weight | float | 是 | 图中段落节点的权重 | 0.05 |
| linking_top_k | int | 是 | 每个节点链接的top-k文档数量 | 10 |
| embedding_prefix | str | 是 | 嵌入文件前缀 | "entity_embeddings" |
| dspy_file_path | str | 否 | DSPy配置文件路径 | "/app/configs/filter_llama3.3-70B-Instruct.json" |
| storage_provider | str | 是 | HippoRAG存储供应商 | "hipporag_storage_local" |

## [cache]  
**缓存配置**

| 配置项 | 类型 | 必填 | 描述 | 默认值 |
|-------|------|------|------|--------|
| provider | str | 是 | 缓存供应商 | "memory" |
| cache_dir | str | 否 | 缓存目录 | "/app/cache" |
| host | str | 否 | 缓存主机地址 | "localhost" |
| port | int | 否 | 缓存服务端口 | 6379 |
| auth | str | 否 | 缓存认证信息 | null |

## [search_engine]  
**搜索引擎配置**

| 配置项 | 类型 | 必填 | 描述 | 默认值 |
|-------|------|------|------|--------|
| provider | str | 是 | 搜索引擎供应商 | "elasticsearch" |
| host | str | 是 | 搜索引擎主机地址 | "localhost" |
| port | int | 是 | 搜索引擎端口 | 9200 |
| index_name_prefix | str | 是 | 搜索引擎索引名称 | "raghub_index" |
| username | str | 否 | 搜索引擎用户名 | "elastic" |
| password | str | 否 | 搜索引擎密码 | null |
| use_ssl | bool | 是 | 是否使用SSL连接 | false |
| verify_certs | bool | 是 | 是否验证SSL证书 | false |

## [graphrag]  
**GraphRAG配置**

| 配置项 | 类型 | 必填 | 描述 | 默认值 |
|-------|------|------|------|--------|
| dao_provider | str | 是 | GraphRAG数据访问对象供应商 | "default_graph_rag_dao" |

## [interfaces]  
**接口配置**

| 配置项 | 类型 | 必填 | 描述 | 默认值 |
|-------|------|------|------|--------|
| rag_provider | str | 是 | RAG供应商 | "hipporag_app" |

## [interfaces.server]  
**服务器接口配置**

| 配置项 | 类型 | 必填 | 描述 | 默认值 |
|-------|------|------|------|--------|
| name | str | 是 | RAGHub服务器名称 | "RAGHub Server" |
| address | str | 是 | RAGHub服务器地址 | "127.0.0.1" |
| port | int | 是 | RAGHub服务器端口 | 8000 |