[English](README.md) | [中文](README.zh.md)  

# RAGHUB

A platform that integrates common RAG algorithms and frameworks with engineering implementations, providing RESTful API interfaces and gRPC services. It supports both offline and online storage solutions.

## Implemented RAG Algorithms

- RAG Algorithms

  - [x] GraphRAG

  - [x] HippoRAG

# Features  

- Supports Dify external knowledge base interface  

- Supports both RESTFul API and gRPC services

- Supports offline and online storage solutions

- Compatible with OPENAI SDK for LLM interfaces  

## Installation and Deployment

### Source Code Installation
- Clone the code repository
```bash
git clone --recurse-submodules https://github.com/ahaeureka/raghub.git
```

- Install dependencies
```bash
cd raghub && uv sync -v --active --all-packages --default-index https://mirrors.aliyun.com/pypi/simple/ --extra online --index-strategy unsafe-best-match --prerelease=allow --no-build-isolation
```

- Configure dependencies  

Modify `configs/offline.toml`

- Start the service
```bash
raghub start server -c /app/configs/offline.toml
```
### Docker Installation  

#### Build Docker Image

```bash
docker build -t raghub:latest .
```

#### Run Docker Container

```bash
docker run -d --name raghub -p 8000:8000 -v /path/to/configs/config.toml:/app/configs/config.toml raghub:latest

```

### Configuration

[Configuration docs](docs/config.md)

## RESTFul API && gRPC Server

[Interface Definition](https://github.com/ahaeureka/raghub-protos/blob/main/rag.proto)