# RAGHUB

集合常见RAG算法和框架的工程化实现，提供RESTful API接口和gRPC服务，支持离线存储和在线存储方案。

# 已经实现的RAG算法

- RAG算法

  - [x] GraphRAG

  - [x] HippoRAG

# 安装部署

### 源码安装
- 克隆代码仓库
```bash
git clone https://github.com/ahaeureka/raghub.git
```

- 安装依赖
```bash
cd raghub && uv sync -v --active --all-packages --default-index https://mirrors.aliyun.com/pypi/simple/ --extra online --index-strategy unsafe-best-match --prerelease=allow --no-build-isolation
```

- 配置依赖
修改 `configs/offline.toml`

- 启动服务
```bash
cd packages/raghub-interfaces/src/raghub_interfaces && uv run raghub.py start server -c /app/configs/offline.toml
```
### Docker安装
TODO

# RESTFul API && gRPC Server

[接口定义](https://github.com/ahaeureka/raghub-protos/blob/main/rag.proto)