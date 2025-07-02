# RAGHub Client Test Suite

This is the complete test suite for the RAGHub client, designed to test various RAG service functionalities with automatic server management.

## Automatic Server Management

**NEW FEATURE**: The test suite now includes automatic RAGHub server management:

- 🚀 **Auto-start**: Server automatically starts before tests run
- 🛑 **Auto-stop**: Server stops after all tests complete  
- 🔍 **Health check**: Verifies server is healthy before proceeding
- ⚙️ **Configurable**: Can be disabled via environment variables

### Quick Start

```bash
# Run tests with automatic server management (default)
cd /app
python -m pytest packages/raghub-client/tests/ -v

# Disable auto-start if you prefer manual server management
export RAGHUB_AUTO_START_SERVER=false
raghub start server -c configs/test.toml  # Start manually
python -m pytest packages/raghub-client/tests/ -v
```

### Configuration

Set these environment variables to customize behavior:

- `RAGHUB_AUTO_START_SERVER=true` - Auto-start server (default)
- `RAGHUB_SERVER_CONFIG=configs/test.toml` - Server config file
- `RAGHUB_SERVER_STARTUP_TIMEOUT=60` - Startup timeout seconds

## 测试结构

```
tests/
├── __init__.py              # 测试包初始化
├── config.py               # 测试配置
├── base_test.py            # 测试基类
├── test_retrieval.py       # 检索服务测试
├── test_chat.py            # 聊天服务测试
├── test_documents.py       # 文档管理测试
├── test_index.py           # 索引管理测试
├── test_integration.py     # 集成测试
├── test_utils.py           # 测试工具函数
├── run_tests.py            # 测试运行器
├── pytest.ini             # pytest 配置
└── README.md               # 本文件
```

## 测试功能覆盖

### 1. 检索服务测试 (`test_retrieval.py`)
- 基本检索功能
- 不同 top_k 值的检索
- 带元数据过滤的检索
- 空查询处理
- 不存在知识库的错误处理
- 检索结果排序验证
- 响应字段完整性验证

### 2. 聊天服务测试 (`test_chat.py`)
- 基本问答功能
- 流式响应处理
- 基于上下文的聊天
- 空问题处理
- 复杂问题处理
- 响应结构验证

### 3. 文档管理测试 (`test_documents.py`)
- 批量添加文档
- 单个文档添加
- 带复杂元数据的文档
- 文档删除功能
- 各种边界情况处理
- 文档内容验证

### 4. 索引管理测试 (`test_index.py`)
- 基本索引创建
- 特殊字符索引名
- 重复索引处理
- 索引名长度限制
- 并发索引创建
- 索引创建后的使用

### 5. 集成测试 (`test_integration.py`)
- 完整工作流程测试
- 错误处理工作流程
- 性能测试
- 数据一致性测试
- 并发操作测试

## 环境配置

### 环境变量

在运行测试前，请设置以下环境变量：

```bash
# 必需配置
export RAGHUB_TEST_BASE_URL="http://localhost:8000"  # RAGHub 服务器地址

# 可选配置
export RAGHUB_TEST_API_KEY=""                        # API 密钥（如果需要）
export RAGHUB_TEST_TIMEOUT="60.0"                   # 请求超时时间（秒）
```

### 依赖安装

确保已安装必要的依赖：

```bash
pip install pytest pytest-asyncio httpx
```

## 运行测试

### 方法 1: 使用测试运行器（推荐）

```bash
# 交互式选择测试
cd /app/packages/raghub-client/tests
python run_tests.py

# 运行特定测试
python run_tests.py --interactive

# 运行全部测试
python run_tests.py --all
```

### 方法 2: 直接使用 pytest

```bash
cd /app/packages/raghub-client/tests

# 运行全部测试
pytest -v

# 运行特定测试文件
pytest test_retrieval.py -v

# 运行特定测试类
pytest test_retrieval.py::TestRetrievalService -v

# 运行特定测试方法
pytest test_retrieval.py::TestRetrievalService::test_retrieval_basic -v

# 按模式筛选测试
pytest -k "retrieval" -v

# 运行慢速测试
pytest -m "slow" -v

# 跳过慢速测试
pytest -m "not slow" -v
```

### 方法 3: 运行特定类型的测试

```bash
# 只运行单元测试
pytest -m unit -v

# 只运行集成测试
pytest -m integration -v

# 运行检索相关测试
pytest -k "retrieval" -v

# 运行聊天相关测试
pytest -k "chat" -v
```

## 测试配置

### 修改测试配置

编辑 `config.py` 文件来自定义测试设置：

```python
class TestConfig:
    # 服务器配置
    BASE_URL = "http://your-server:port"
    API_KEY = "your-api-key"
    TIMEOUT = 30.0
    
    # 测试数据
    TEST_KNOWLEDGE_ID = "your_test_kb"
    TEST_DOCUMENTS = [...]  # 自定义测试文档
```

### pytest 配置

`pytest.ini` 文件包含了 pytest 的默认配置，包括：
- 异步测试支持
- 测试发现模式
- 输出格式
- 标记定义

## 测试最佳实践

### 1. 环境隔离
- 每个测试使用独立的知识库ID
- 测试完成后自动清理资源
- 使用随机后缀避免冲突

### 2. 错误处理
- 验证正常情况和异常情况
- 记录详细的错误信息
- 使用适当的断言和日志

### 3. 性能考虑
- 包含性能基准测试
- 监控响应时间
- 测试并发场景

### 4. 数据验证
- 验证响应结构
- 检查数据一致性
- 确保内容相关性

## 故障排除

### 常见问题

1. **连接错误**
   ```
   确保 RAGHub 服务器正在运行
   检查 BASE_URL 配置是否正确
   验证网络连接
   ```

2. **认证错误**
   ```
   检查 API_KEY 配置
   确认认证方式是否正确
   ```

3. **超时错误**
   ```
   增加 TIMEOUT 配置值
   检查服务器性能
   减少测试数据量
   ```

4. **测试失败**
   ```
   查看详细日志输出
   检查测试数据是否有效
   确认服务器功能正常
   ```

### 日志和调试

测试运行时会生成详细日志：
- 控制台输出：实时显示测试进度
- `test_results.log`：完整的测试日志
- pytest 报告：测试结果汇总

启用详细日志：

```bash
pytest -v -s --log-cli-level=DEBUG
```

## 扩展测试

### 添加新测试

1. 创建新的测试文件：`test_your_feature.py`
2. 继承 `BaseRAGTest` 类
3. 使用适当的 pytest 标记
4. 添加到测试运行器中

### 自定义断言

扩展 `test_utils.py` 中的验证函数来支持新的响应类型。

### 性能基准

在 `test_integration.py` 中添加性能基准测试，监控关键指标。

## 贡献指南

1. 遵循现有代码风格
2. 添加适当的文档和注释
3. 确保新测试的独立性
4. 更新相关文档

## 联系和支持

如有问题或建议，请联系开发团队或提交 issue。
