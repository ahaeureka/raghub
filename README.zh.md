# RAGHUB

集合常见RAG算法和框架的工程化实现，提供RESTful API接口和gRPC服务，支持离线存储和在线存储方案。

# 已经实现的RAG算法

- RAG算法

  - [x] GraphRAG

  - [x] HippoRAG  

# 特性  

- 支持Dify外挂知识库接口  

- 同时支持RESTFul API和gRPC服务

- 支持离线存储和在线存储方案

- 支持兼容OPENAI SDK的LLM接口  

# 安装部署

### 源码安装
- 克隆代码仓库
```bash
git clone --recurse-submodules https://github.com/ahaeureka/raghub.git
```

- 安装依赖
```bash
cd raghub && uv sync -v --active --all-packages --default-index https://mirrors.aliyun.com/pypi/simple/ --extra online --index-strategy unsafe-best-match --prerelease=allow --no-build-isolation
```

- 配置依赖
修改 `configs/offline.toml`

- 启动服务
```bash
raghub start server -c /app/configs/offline.toml
```
### Docker安装

#### 构建Docker镜像

```bash
docker build -t raghub:latest .
```
#### 运行Docker容器

```bash
docker run -d --name raghub -p 8000:8000 -v /path/to/configs/config.toml:/app/configs/config.toml raghub:latest
```

### 配置

[配置文档](docs/config.zh.md)

# RESTFul API && gRPC Server

[接口定义](https://github.com/ahaeureka/raghub-protos/blob/main/rag.proto)