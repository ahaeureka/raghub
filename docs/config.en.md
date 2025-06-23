# RAGHub Configuration Documentation

## Logger Configuration (LoggerConfig)
- **log_level**: Logging level, defaults to "DEBUG". Tags: log.
- **log_dir**: Path to the log file, defaults to "logs". Tags: log.

## Proxy LLM Configuration (ProxyLLMConfig)
- **api_key**: OpenAI API Secret Key, environment variable `OPENAI_API_KEY`. Defaults to None. Tags: secret.
- **base_url**: Base URL for OpenAI API. Defaults to None. Tags: url.
- **model**: Name of the OpenAI model, defaults to "gpt-3.5-turbo". Tags: model.
- **timeout**: Timeout for OpenAI API requests in seconds, defaults to 60. Tags: timeout.
- **provider**: Provider of the embedding model, defaults to "openai-proxy". Tags: provider.
- **temperature**: Temperature setting for the OpenAI model, defaults to 0.3. Tags: temperature.

## Embedding Model Configuration (EmbbedingModelConfig)
Inherits from `ProxyLLMConfig`, and adds:
- **batch_size**: Batch size for embeddings, defaults to 32. Tags: batch_size.
- **n_dims**: Number of dimensions for the embedding, defaults to None. Tags: n_dims.
- **embedding_key_prefix**: Prefix for the embedding key, defaults to "embedding". Tags: embedding_key_prefix.

## RAG Configuration (RAGConfig)
- **llm**: Configuration for the Language Learning Model, defaults to using `ProxyLLMConfig()`. Tags: llm.
- **embbeding**: Configuration for the embedding model, defaults to using `EmbbedingModelConfig()`. Tags: embedding.
- **lang**: Language for the RAG system, defaults to "en". Tags: lang.

## Vector Storage Configuration (VectorStorageConfig)
- **provider**: Vector storage provider, defaults to "chromadb". Tags: vector_storage.
- **collection_name**: Name of the collection in the vector storage, defaults to "default_collection". Tags: vector_storage.
- **persist_directory**: Directory to persist the vector storage. Tags: vector_storage.
- **host**: Host address for the vector storage, defaults to "localhost". Tags: vector_storage.
- **port**: Port number for the vector storage, defaults to 8000. Tags: vector_storage.
- **username**: Username for authentication with the vector storage, defaults to None. Tags: vector_storage.
- **password**: Password for authentication with the vector storage, defaults to None. Tags: vector_storage.
- **token**: Token for authentication with the vector storage, defaults to None. Tags: vector_storage.
- **embedding_key_prefix**: Prefix for the embedding key, defaults to "embedding". Tags: vector_storage.

## Graph Storage Configuration (GraphStorageConfig)
- **provider**: Graph storage provider, defaults to "igraph". Tags: graph_storage.
- **graph_path**: Path to the graph file. Tags: graph_storage.
- **url**: URL for the graph storage, defaults to "http://localhost:8000". Tags: graph_storage.
- **username**: Username for authentication with the graph storage, defaults to None. Tags: graph_storage.
- **password**: Password for authentication with the graph storage, defaults to None. Tags: graph_storage.
- **database**: Database name for the graph storage, defaults to None. Tags: graph_storage.

## Application Configuration (RAGHubConfig)
- **app_name**: Name of the application, defaults to "RAGHub". Tags: app.
- **app_version**: Version of the application, defaults to "0.1.0". Tags: app.
- **app_description**: Description of the application, defaults to "RAGHub Application". Tags: app.
- **logger**: Configuration for logging, defaults to `LoggerConfig()`. Tags: logger.
- **rag**: Configuration for RAG, defaults to `RAGConfig()`. Tags: rag.
- **vector_storage**: Configuration for vector storage, defaults to `VectorStorageConfig()`. Tags: vector_storage.
- **graph**: Configuration for graph storage, defaults to `GraphStorageConfig()`. Tags: graph.

## Search Engine Configuration (SearchEngineConfig)
- **provider**: Provider of the search engine, defaults to "elasticsearch". Tags: search.
- **host**: Host address for the search engine, defaults to "localhost". Tags: search.
- **port**: Port number for the search engine, defaults to 9200. Tags: search.
- **index_name_prefix**: Prefix for the index name in the search engine, defaults to "raghub_index". Tags: search.
- **use_ssl**: Whether to use SSL for the search engine, defaults to False. Tags: search.
- **verify_certs**: Whether to verify SSL certificates for the search engine, defaults to False. Tags: search.

## Database Configuration (DatabaseConfig)
- **provider**: Provider of the database, defaults to "sqlite". Tags: database.
- **db_url**: URL for the database connection, defaults to "sqlite:///app.db". Tags: database.

## Cache Configuration (CacheConfig)
- **provider**: Provider of the cache, defaults to "memory". Tags: cache.
- **cache_dir**: Directory for caching. Tags: cache.
- **host**: Host address for the cache, defaults to "localhost". Tags: cache.
- **port**: Port number for the cache, defaults to 6379. Tags: cache.
- **auth**: Authentication information for the cache, defaults to None. Tags: cache.