# RAGHUB

A platform that integrates common RAG algorithms and frameworks with engineering implementations, providing RESTful API interfaces and gRPC services. It supports both offline and online storage solutions.

## Implemented RAG Algorithms

- RAG Algorithms

  - [x] GraphRAG

  - [x] HippoRAG

## Installation and Deployment

### Source Code Installation
- Clone the code repository
```bash
git clone https://github.com/ahaeureka/raghub.git
```

- Install dependencies
```bash
cd raghub && uv sync -v --active --all-packages --default-index https://mirrors.aliyun.com/pypi/simple/ --extra online --index-strategy unsafe-best-match --prerelease=allow --no-build-isolation
```

- Configure dependencies
Modify `configs/offline.toml`

- Start the service
```bash
cd packages/raghub-interfaces/src/raghub_interfaces && uv run raghub.py start server -c /app/configs/offline.toml
```
### Docker Installation
TODO

## RESTFul API && gRPC Server

[Interface Definition](packages/raghub-interfaces/src/protos/rag.proto)