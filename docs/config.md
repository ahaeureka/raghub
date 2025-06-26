**InerfaceConfig Configuration File Documentation**  
**Auto-generated on 2025-06-26 05:42:32**  
**Description: Configuration settings** 

# Root Level Configuration

| Configuration | Type | Required | Description | Default |
|---------------|------|----------|-------------|---------|
| app_name | str | true | Application Name | "RAGHub" |
| app_version | str | true | Application Version | "0.1.0" |
| app_description | str | true | Application Description | "RAGHub Application" |

## [logger]
**Configuration for logger**

| Configuration | Type | Required | Description | Default |
|---------------|------|----------|-------------|---------|
| log_level | str | true | Log Level | "DEBUG" |
| log_dir | str | true | Log File Path | "logs" |

## [rag]
**Configuration for rag**

| Configuration | Type | Required | Description | Default |
|---------------|------|----------|-------------|---------|
| lang | str | true | Language for the RAG system | "en" |

## [rag.llm]
**Configuration for rag.llm**

| Configuration | Type | Required | Description | Default |
|---------------|------|----------|-------------|---------|
| api_key | str | false | OpenAI API Secret Key | null |
| base_url | str | false | OpenAI API Base URL | null |
| model | str | true | OpenAI Model Name | "gpt-3.5-turbo" |
| timeout | int | true | OpenAI API Timeout in seconds | 60 |
| provider | str | true | Provider of the embedding model | "openai-proxy" |
| temperature | float | true | Temperature for the OpenAI model | 0.3 |

## [rag.embbeding]
**Configuration for rag.embbeding**

| Configuration | Type | Required | Description | Default |
|---------------|------|----------|-------------|---------|
| api_key | str | false | OpenAI API Secret Key | null |
| base_url | str | false | OpenAI API Base URL | null |
| model | str | true | OpenAI Model Name | "gpt-3.5-turbo" |
| timeout | int | true | OpenAI API Timeout in seconds | 60 |
| provider | str | true | Provider of the embedding model | "openai-proxy-embedding" |
| temperature | float | true | Temperature for the OpenAI model | 0.3 |
| batch_size | int | true | Batch size for embedding | 32 |
| n_dims | int | false | Number of dimensions for the embedding | null |
| embedding_key_prefix | str | false | Prefix for the embedding key | "embedding" |

## [vector_storage]
**Configuration for vector_storage**

| Configuration | Type | Required | Description | Default |
|---------------|------|----------|-------------|---------|
| provider | str | true | Vector storage provider | "chromadb" |
| collection_name | str | true | Name of the collection in the vector storage | "default_collection" |
| persist_directory | str | false | Directory to persist the vector storage | "/app/cache/chroma_db" |
| host | str | false | Host for the vector storage | "localhost" |
| port | int | false | Port for the vector storage | 8000 |
| username | str | false | Username for vector storage authentication | null |
| password | str | false | Password for vector storage authentication | null |
| token | str | false | Token for vector storage authentication | null |
| embedding_key_prefix | str | false | Prefix for the embedding key | "embedding" |

## [graph]
**Configuration for graph**

| Configuration | Type | Required | Description | Default |
|---------------|------|----------|-------------|---------|
| provider | str | true | Graph storage provider | "igraph" |
| graph_path | str | false | Path to the graph file | "/app/storage/graphs/default_graph.pkl" |
| url | str | false | Graph storage URL | "http://localhost:8000" |
| username | str | false | Username for graph storage authentication | null |
| password | str | false | Password for graph storage authentication | null |
| database | str | false | Database name for graph storage | null |

## [database]
**Configuration for database**

| Configuration | Type | Required | Description | Default |
|---------------|------|----------|-------------|---------|
| provider | str | true | Database Provider | "sqlite" |
| db_url | str | true | Database URL | "sqlite:///app.db" |

## [hipporag]
**Configuration for hipporag**

| Configuration | Type | Required | Description | Default |
|---------------|------|----------|-------------|---------|
| synonymy_edge_topk | int | true | Number of top-k synonyms to consider for each node | 2047 |
| synonymy_edge_query_batch_size | int | true | Batch size for query embeddings during synonymy edge construction | 1000 |
| synonymy_edge_key_batch_size | int | true | Batch size for key embeddings during synonymy edge construction | 1000 |
| synonymy_edge_sim_threshold | float | true | Similarity threshold for synonymy edges | 0.8 |
| passage_node_weight | float | true | Weight of passage nodes in the graph | 0.05 |
| linking_top_k | int | true | Number of top-k documents to link to each node | 10 |
| embedding_prefix | str | true | Prefix for embedding files | "entity_embeddings" |
| dspy_file_path | str | false | Path to the dspy file | "/app/configs/filter_llama3.3-70B-Instruct.json" |
| storage_provider | str | true | Storage provider for HippoRAG | "hipporag_storage_local" |

## [cache]
**Configuration for cache**

| Configuration | Type | Required | Description | Default |
|---------------|------|----------|-------------|---------|
| provider | str | true | Cache Provider | "memory" |
| cache_dir | str | false | Cache Directory | "/app/cache" |
| host | str | false | Cache Host | "localhost" |
| port | int | false | Cache Port | 6379 |
| auth | str | false | Cache Authentication | null |

## [search_engine]
**Configuration for search_engine**

| Configuration | Type | Required | Description | Default |
|---------------|------|----------|-------------|---------|
| provider | str | true | Search Engine Provider | "elasticsearch" |
| host | str | true | Search Engine Host | "localhost" |
| port | int | true | Search Engine Port | 9200 |
| index_name_prefix | str | true | Search Engine Index Name | "raghub_index" |
| username | str | false | Search Engine Username | "elastic" |
| password | str | false | Search Engine Password | null |
| use_ssl | bool | true | Use SSL for Search Engine | false |
| verify_certs | bool | true | Verify SSL certificates for Search Engine | false |

## [graphrag]
**Configuration for graphrag**

| Configuration | Type | Required | Description | Default |
|---------------|------|----------|-------------|---------|
| dao_provider | str | true | DAO provider for GraphRAG | "default_graph_rag_dao" |

## [interfaces]
**Configuration for interfaces**

| Configuration | Type | Required | Description | Default |
|---------------|------|----------|-------------|---------|
| rag_provider | str | true | RAG Provider | "hipporag_app" |

## [interfaces.server]
**Configuration for interfaces.server**

| Configuration | Type | Required | Description | Default |
|---------------|------|----------|-------------|---------|
| name | str | true | Name of the RAGHub server | "RAGHub Server" |
| address | str | true | Address of the RAGHub server | "127.0.0.1" |
| port | int | true | Port of the RAGHub server | 8000 |
