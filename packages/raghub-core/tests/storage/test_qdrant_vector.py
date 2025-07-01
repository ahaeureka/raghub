"""
QdrantVector 单元测试

测试 QdrantVector 的各种功能，使用本地 Qdrant 实例
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import List, Optional

import numpy as np
import pytest

try:
    import qdrant_client  # noqa: F401
    from langchain_qdrant import QdrantVectorStore  # noqa: F401

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

from raghub_core.embedding.base_embedding import BaseEmbedding
from raghub_core.schemas.document import Document


def create_test_qdrant_vector(embedder, persist_directory):
    """为测试创建非单例的QdrantVector实例"""
    from raghub_core.storage.qdrant_vector import QdrantVector
    from raghub_core.utils.class_meta import SingletonRegisterMeta

    # 临时清除单例缓存中该类的实例（仅用于测试）
    if QdrantVector in SingletonRegisterMeta._instances:
        del SingletonRegisterMeta._instances[QdrantVector]

    # 创建新实例
    instance = QdrantVector(embedder=embedder, persist_directory=persist_directory)

    # 再次清除缓存，确保下次测试获得新实例
    if QdrantVector in SingletonRegisterMeta._instances:
        del SingletonRegisterMeta._instances[QdrantVector]

    return instance


class MockEmbedding(BaseEmbedding):
    """用于测试的简单embedding实现"""

    name = "mock_embedding"

    def __init__(self, embedding_dim: int = 384):
        super().__init__()
        self.n_dim = embedding_dim
        # 为一致性预设一些固定的embeddings
        self._fixed_embeddings = {
            "test document 1": np.random.RandomState(42).randn(embedding_dim).astype(np.float32),
            "test document 2": np.random.RandomState(43).randn(embedding_dim).astype(np.float32),
            "programming tutorial": np.random.RandomState(44).randn(embedding_dim).astype(np.float32),
            "machine learning": np.random.RandomState(45).randn(embedding_dim).astype(np.float32),
            "python guide": np.random.RandomState(46).randn(embedding_dim).astype(np.float32),
        }

    @property
    def embedding_dim(self):
        return self.n_dim

    def encode(self, texts: List[str], instruction: Optional[str] = None) -> np.ndarray:
        """返回固定的embeddings以保证测试一致性"""
        embeddings = []
        for text in texts:
            if text in self._fixed_embeddings:
                embeddings.append(self._fixed_embeddings[text])
            else:
                # 为新文本生成一致的embedding
                hash_seed = hash(text) % (2**31)
                embeddings.append(np.random.RandomState(hash_seed).randn(self.embedding_dim).astype(np.float32))
        return np.array(embeddings)

    def encode_query(self, query: str, instruction: Optional[str] = None) -> np.ndarray:
        """返回查询的embedding"""
        return self.encode([query])[0]

    async def aencode(self, texts: List[str], instruction: Optional[str] = None) -> np.ndarray:
        """异步版本的encode"""
        return self.encode(texts, instruction)

    async def aencode_query(self, query: List[str], instruction: Optional[str] = None) -> np.ndarray:
        """异步版本的encode_query"""
        if isinstance(query, list):
            return self.encode(query, instruction)
        else:
            return self.encode([query], instruction)[0]


@pytest.mark.skipif(not QDRANT_AVAILABLE, reason="qdrant-client or langchain-qdrant not installed")
class TestQdrantVector:
    """QdrantVector 测试类"""

    @pytest.fixture
    def mock_embedder(self):
        """创建mock embedder实例"""
        return MockEmbedding(embedding_dim=384)

    @pytest.fixture
    async def qdrant_vector(self, mock_embedder, tmp_path, request):
        """创建QdrantVector实例，为每个测试方法创建完全独立的实例"""
        import time
        import uuid

        from raghub_core.storage.qdrant_vector import QdrantVector
        from raghub_core.utils.class_meta import SingletonRegisterMeta

        # 清理单例缓存，确保获得全新实例
        if QdrantVector in SingletonRegisterMeta._instances:
            del SingletonRegisterMeta._instances[QdrantVector]

        # 为每个测试创建唯一的目录名，包含测试名和时间戳
        test_name = request.node.name.replace("[", "_").replace("]", "_")
        unique_id = str(uuid.uuid4())[:8]
        timestamp = str(int(time.time() * 1000))
        test_dir = tmp_path / f"qdrant_{test_name}_{unique_id}_{timestamp}"

        vector_store = QdrantVector(embedder=mock_embedder, persist_directory=test_dir)
        await vector_store.init()

        yield vector_store

        # 测试后清理：再次清理单例缓存，确保下个测试获得新实例
        try:
            if QdrantVector in SingletonRegisterMeta._instances:
                del SingletonRegisterMeta._instances[QdrantVector]
        except Exception:
            # 忽略清理错误
            pass

    @pytest.fixture
    def sample_documents(self):
        """创建测试用的文档"""
        return [
            Document(
                content="test document 1",
                metadata={"category": "programming", "priority": 5, "tags": ["python", "tutorial"]},
                uid="11111111-1111-1111-1111-111111111111",
                summary="A programming tutorial",
            ),
            Document(
                content="test document 2",
                metadata={"category": "machine-learning", "priority": 3, "tags": ["ai", "ml"]},
                uid="22222222-2222-2222-2222-222222222222",
                summary="An ML guide",
            ),
            Document(
                content="programming tutorial",
                metadata={"category": "programming", "priority": 4, "tags": ["coding", "beginner"]},
                uid="33333333-3333-3333-3333-333333333333",
                summary="Basic programming concepts",
            ),
        ]

    @pytest.mark.asyncio
    async def test_initialization(self, mock_embedder, tmp_path):
        """测试QdrantVector初始化"""
        import time
        import uuid

        from raghub_core.storage.qdrant_vector import QdrantVector
        from raghub_core.utils.class_meta import SingletonRegisterMeta

        # 清理单例缓存
        if QdrantVector in SingletonRegisterMeta._instances:
            del SingletonRegisterMeta._instances[QdrantVector]

        # 为初始化测试创建唯一的目录名
        unique_id = str(uuid.uuid4())[:8]
        timestamp = str(int(time.time() * 1000))
        test_dir = tmp_path / f"test_init_{unique_id}_{timestamp}"
        vector_store = QdrantVector(embedder=mock_embedder, persist_directory=test_dir)

        assert vector_store._embedder == mock_embedder
        assert vector_store._persist_directory == test_dir
        assert vector_store._client is None  # Lazy loading
        assert len(vector_store._vector_stores) == 0

        # 测试后清理单例缓存
        if QdrantVector in SingletonRegisterMeta._instances:
            del SingletonRegisterMeta._instances[QdrantVector]

    @pytest.mark.asyncio
    async def test_client_lazy_loading(self, qdrant_vector):
        """测试客户端懒加载"""
        # 第一次访问时创建客户端
        client1 = qdrant_vector.client
        assert client1 is not None

        # 第二次访问应该返回同一个客户端
        client2 = qdrant_vector.client
        assert client1 is client2

    @pytest.mark.asyncio
    async def test_create_index(self, qdrant_vector):
        """测试创建索引"""
        index_name = "test_index"

        # 创建索引
        vector_store = qdrant_vector.create_index(index_name)
        assert vector_store is not None
        assert index_name in qdrant_vector._vector_stores

        # 再次创建相同索引应该返回缓存的实例
        vector_store2 = qdrant_vector.create_index(index_name)
        assert vector_store is vector_store2

    @pytest.mark.asyncio
    async def test_add_documents(self, qdrant_vector, sample_documents):
        """测试添加文档"""
        index_name = "test_add_docs"

        # 添加文档
        added_docs = await qdrant_vector.add_documents(index_name, sample_documents)

        assert len(added_docs) == len(sample_documents)

        # 检查内容是否正确（不依赖UID顺序，因为LangChain可能重新生成ID）
        added_contents = {doc.content for doc in added_docs}
        expected_contents = {doc.content for doc in sample_documents}
        assert added_contents == expected_contents

        # 检查每个文档都有valid ID
        for doc in added_docs:
            assert doc.uid is not None
            assert doc.content is not None
            assert doc.metadata is not None

    @pytest.mark.asyncio
    async def test_get_document_by_id(self, qdrant_vector, sample_documents):
        """测试通过ID获取文档"""
        index_name = "test_get_doc"

        # 先添加文档
        await qdrant_vector.add_documents(index_name, sample_documents)

        # 通过ID获取文档
        doc = await qdrant_vector.get(index_name, "11111111-1111-1111-1111-111111111111")
        assert doc.uid == "11111111-1111-1111-1111-111111111111"
        assert doc.content == "test document 1"
        assert doc.metadata["category"] == "programming"

    @pytest.mark.asyncio
    async def test_get_documents_by_ids(self, qdrant_vector, sample_documents):
        """测试通过ID列表获取文档"""
        index_name = "test_get_docs"

        # 先添加文档
        await qdrant_vector.add_documents(index_name, sample_documents)

        # 通过ID列表获取文档
        docs = await qdrant_vector.get_by_ids(
            index_name, ["11111111-1111-1111-1111-111111111111", "33333333-3333-3333-3333-333333333333"]
        )
        assert len(docs) == 2

        doc_ids = [doc.uid for doc in docs]
        assert "11111111-1111-1111-1111-111111111111" in doc_ids
        assert "33333333-3333-3333-3333-333333333333" in doc_ids

    @pytest.mark.asyncio
    async def test_get_by_ids_empty_list(self, qdrant_vector):
        """测试空ID列表"""
        index_name = "test_empty_ids"
        docs = await qdrant_vector.get_by_ids(index_name, [])
        assert docs == []

    @pytest.mark.asyncio
    async def test_get_nonexistent_document(self, qdrant_vector, sample_documents):
        """测试获取不存在的文档"""
        index_name = "test_nonexistent"

        # 先添加文档
        await qdrant_vector.add_documents(index_name, sample_documents)

        # 尝试获取不存在的文档
        with pytest.raises(ValueError, match="Document with ID nonexistent not found"):
            await qdrant_vector.get(index_name, "nonexistent")

    @pytest.mark.asyncio
    async def test_delete_documents(self, qdrant_vector, sample_documents):
        """测试删除文档"""
        index_name = "test_delete"

        # 先添加文档
        await qdrant_vector.add_documents(index_name, sample_documents)

        # 删除部分文档
        success = await qdrant_vector.delete(
            index_name, ["11111111-1111-1111-1111-111111111111", "22222222-2222-2222-2222-222222222222"]
        )
        assert success is True

        # 验证文档已被删除
        with pytest.raises(ValueError):
            await qdrant_vector.get(index_name, "11111111-1111-1111-1111-111111111111")

        # 验证未删除的文档仍然存在
        doc3 = await qdrant_vector.get(index_name, "33333333-3333-3333-3333-333333333333")
        assert doc3.uid == "33333333-3333-3333-3333-333333333333"

    @pytest.mark.asyncio
    async def test_delete_empty_list(self, qdrant_vector):
        """测试删除空ID列表"""
        index_name = "test_delete_empty"
        success = await qdrant_vector.delete(index_name, [])
        assert success is False

    @pytest.mark.asyncio
    async def test_similarity_search_by_vector(self, qdrant_vector, sample_documents):
        """测试向量相似度搜索"""
        index_name = "test_similarity_vector"

        # 先添加文档
        await qdrant_vector.add_documents(index_name, sample_documents)

        # 创建查询向量
        query_embedding = qdrant_vector._embedder.encode_query("programming tutorial")

        # 执行相似度搜索
        results = await qdrant_vector.similarity_search_by_vector(index_name, query_embedding.tolist(), k=2)

        assert len(results) <= 2
        for doc, score in results:
            assert isinstance(doc, Document)
            assert isinstance(score, (int, float))
            assert 0 <= score <= 1

    @pytest.mark.asyncio
    async def test_similarity_search_with_query(self, qdrant_vector, sample_documents):
        """测试查询字符串相似度搜索"""
        index_name = "test_similarity_query"

        # 先添加文档
        await qdrant_vector.add_documents(index_name, sample_documents)

        # 执行相似度搜索
        results = await qdrant_vector.asimilar_search_with_scores(index_name, "programming tutorial", k=2)

        assert len(results) <= 2
        for doc, score in results:
            assert isinstance(doc, Document)
            assert isinstance(score, (int, float))

    @pytest.mark.asyncio
    async def test_metadata_condition_format_detection(self, qdrant_vector):
        """测试元数据条件格式检测"""
        # MetadataCondition格式
        metadata_condition = {
            "logical_operator": "and",
            "conditions": [{"name": ["category"], "comparison_operator": "is", "value": "programming"}],
        }
        assert qdrant_vector._is_metadata_condition_format(metadata_condition) is True

        # 简单格式
        simple_filter = {"category": "programming", "priority": 5}
        assert qdrant_vector._is_metadata_condition_format(simple_filter) is False

        # 空条件
        empty_condition = {"logical_operator": "and", "conditions": []}
        assert qdrant_vector._is_metadata_condition_format(empty_condition) is False

    @pytest.mark.asyncio
    async def test_select_on_metadata_simple_format(self, qdrant_vector, sample_documents):
        """测试简单格式的元数据选择"""
        index_name = "test_metadata_simple"

        # 先添加文档
        await qdrant_vector.add_documents(index_name, sample_documents)

        # 使用简单格式过滤
        docs = await qdrant_vector.select_on_metadata(index_name, {"category": "programming"})

        assert len(docs) >= 1
        for doc in docs:
            assert doc.metadata["category"] == "programming"

    @pytest.mark.asyncio
    async def test_select_on_metadata_condition_format(self, qdrant_vector, sample_documents):
        """测试MetadataCondition格式的元数据选择"""
        index_name = "test_metadata_condition"

        # 先添加文档
        await qdrant_vector.add_documents(index_name, sample_documents)

        # 使用MetadataCondition格式过滤
        metadata_condition = {
            "logical_operator": "and",
            "conditions": [{"name": ["category"], "comparison_operator": "is", "value": "programming"}],
        }
        docs = await qdrant_vector.select_on_metadata(index_name, metadata_condition)

        assert len(docs) >= 1
        for doc in docs:
            assert doc.metadata["category"] == "programming"

    @pytest.mark.asyncio
    async def test_select_on_metadata_list_values(self, qdrant_vector, sample_documents):
        """测试包含列表值的元数据选择"""
        index_name = "test_metadata_list"

        # 先添加文档
        await qdrant_vector.add_documents(index_name, sample_documents)

        # 使用列表值过滤（OR条件）
        docs = await qdrant_vector.select_on_metadata(index_name, {"category": ["programming", "machine-learning"]})

        assert len(docs) >= 2
        categories = [doc.metadata["category"] for doc in docs]
        assert "programming" in categories or "machine-learning" in categories

    @pytest.mark.asyncio
    async def test_select_on_metadata_empty_filter(self, qdrant_vector, sample_documents):
        """测试空过滤条件"""
        index_name = "test_metadata_empty"

        # 先添加文档
        await qdrant_vector.add_documents(index_name, sample_documents)

        # 空过滤条件应该返回空列表
        docs = await qdrant_vector.select_on_metadata(index_name, {})
        assert docs == []

    @pytest.mark.asyncio
    async def test_similarity_search_with_filter(self, qdrant_vector, sample_documents):
        """测试带过滤器的相似度搜索"""
        index_name = "test_similarity_filter"

        # 先添加文档
        await qdrant_vector.add_documents(index_name, sample_documents)

        # 执行带过滤器的相似度搜索
        results = await qdrant_vector.asimilar_search_with_scores(
            index_name, "tutorial", k=5, filter={"category": "programming"}
        )

        # 验证所有结果都符合过滤条件
        for doc, score in results:
            assert doc.metadata["category"] == "programming"

    @pytest.mark.asyncio
    async def test_get_vector_store(self, qdrant_vector):
        """测试获取vector store"""
        index_name = "test_get_vector_store"

        vector_store = qdrant_vector.get_vector_store(index_name)
        assert vector_store is not None
        assert index_name in qdrant_vector._vector_stores

    @pytest.mark.asyncio
    async def test_persist_directory_property(self, qdrant_vector):
        """测试persist_directory属性"""
        # 检查persist_directory属性是否正确设置
        assert qdrant_vector.persist_directory is not None
        assert isinstance(qdrant_vector.persist_directory, Path)

    @pytest.mark.asyncio
    async def test_build_docs_helper(self, qdrant_vector):
        """测试_build_docs辅助方法"""
        from langchain_core.documents import Document as LangchainDocument

        langchain_docs = [
            LangchainDocument(
                page_content="test content", metadata={"category": "test", "summary": "test summary"}, id="test_id"
            )
        ]

        docs = qdrant_vector._build_docs(langchain_docs)

        assert len(docs) == 1
        doc = docs[0]
        assert doc.content == "test content"
        assert doc.metadata["category"] == "test"
        assert doc.uid == "test_id"
        assert doc.summary == "test summary"

    @pytest.mark.asyncio
    async def test_error_handling_invalid_index(self, qdrant_vector):
        """测试无效索引的错误处理"""
        # 这里我们测试一些可能的错误情况
        # 注意：由于使用本地Qdrant，某些错误可能不会发生

        # 测试获取不存在的文档
        with pytest.raises(ValueError):
            await qdrant_vector.get("nonexistent_index", "nonexistent_doc")

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, qdrant_vector, sample_documents):
        """测试并发操作"""
        index_name = "test_concurrent"

        # 并发添加文档
        tasks = []
        for i in range(3):
            doc_batch = [
                Document(
                    content=f"concurrent document {i}-{j}",
                    metadata={"batch": i, "seq": j},
                    uid=f"{i:08d}-{j:04d}-{j:04d}-{j:04d}-{i:012d}",  # 生成 UUID 格式的 ID
                    summary=f"Summary {i}-{j}",
                )
                for j in range(2)
            ]
            tasks.append(qdrant_vector.add_documents(f"{index_name}_{i}", doc_batch))

        # 等待所有任务完成
        results = await asyncio.gather(*tasks)

        # 验证结果
        for i, result in enumerate(results):
            assert len(result) == 2
            for doc in result:
                assert doc.metadata["batch"] == i


@pytest.mark.skipif(not QDRANT_AVAILABLE, reason="qdrant-client or langchain-qdrant not installed")
class TestQdrantVectorEdgeCases:
    """QdrantVector边界情况测试"""

    @pytest.fixture
    def mock_embedder(self):
        """创建mock embedder实例"""
        return MockEmbedding(embedding_dim=384)

    @pytest.fixture
    async def qdrant_vector(self, mock_embedder, tmp_path, request):
        """创建QdrantVector实例，为每个测试方法创建完全独立的实例"""
        import time
        import uuid

        from raghub_core.storage.qdrant_vector import QdrantVector

        # 为边界情况测试创建唯一的目录名，包含测试名和时间戳
        test_name = request.node.name.replace("[", "_").replace("]", "_")
        unique_id = str(uuid.uuid4())[:8]
        timestamp = str(int(time.time() * 1000))
        test_dir = tmp_path / f"qdrant_edge_{test_name}_{unique_id}_{timestamp}"

        vector_store = QdrantVector(embedder=mock_embedder, persist_directory=test_dir)
        await vector_store.init()

        yield vector_store

        # 显式清理
        try:
            # 关闭所有可能的连接
            if hasattr(vector_store, "_vector_stores"):
                for store in vector_store._vector_stores.values():
                    if hasattr(store, "client") and hasattr(store.client, "_client"):
                        client = store.client._client
                        if hasattr(client, "_flock_file") and client._flock_file:
                            client._flock_file.close()
                        if hasattr(client, "close"):
                            client.close()
        except Exception:
            # 忽略清理错误
            pass

    @pytest.mark.asyncio
    async def test_empty_document_list(self, qdrant_vector):
        """测试添加空文档列表"""
        index_name = "test_empty_docs"
        result = await qdrant_vector.add_documents(index_name, [])
        assert result == []

    @pytest.mark.asyncio
    async def test_document_with_none_metadata(self, qdrant_vector):
        """测试包含None元数据的文档"""
        index_name = "test_none_metadata"
        doc = Document(content="test content", metadata=None, uid="test_none", summary="test summary")

        result = await qdrant_vector.add_documents(index_name, [doc])
        assert len(result) == 1
        assert result[0].metadata is not None  # 应该被转换为空字典

    @pytest.mark.asyncio
    async def test_special_characters_in_content(self, qdrant_vector):
        """测试包含特殊字符的文档内容"""
        index_name = "test_special_chars"
        doc = Document(
            content="测试中文，emoji 😀，特殊符号 @#$%^&*()",
            metadata={"type": "special"},
            uid="special_doc",
            summary="特殊字符测试",
        )

        result = await qdrant_vector.add_documents(index_name, [doc])
        assert len(result) == 1

        retrieved_doc = await qdrant_vector.get(index_name, "special_doc")
        assert retrieved_doc.content == doc.content

    @pytest.mark.asyncio
    async def test_very_long_document(self, qdrant_vector):
        """测试非常长的文档"""
        index_name = "test_long_doc"
        long_content = "Long document content. " * 1000  # 约23KB的内容

        doc = Document(content=long_content, metadata={"type": "long"}, uid="long_doc", summary="Very long document")

        result = await qdrant_vector.add_documents(index_name, [doc])
        assert len(result) == 1

        retrieved_doc = await qdrant_vector.get(index_name, "long_doc")
        assert len(retrieved_doc.content) == len(long_content)


@pytest.mark.skipif(not QDRANT_AVAILABLE, reason="qdrant-client or langchain-qdrant not installed")
class TestQdrantVectorSnowflakeID:
    """QdrantVector雪花ID转换测试"""

    @pytest.fixture
    def mock_embedder(self):
        """创建mock embedder实例"""
        return MockEmbedding(embedding_dim=384)

    @pytest.fixture
    async def qdrant_vector(self, mock_embedder, tmp_path, request):
        """创建QdrantVector实例，为每个测试方法创建完全独立的实例"""
        import time
        import uuid

        from raghub_core.storage.qdrant_vector import QdrantVector
        from raghub_core.utils.class_meta import SingletonRegisterMeta

        # 清理单例缓存，确保获得全新实例
        if QdrantVector in SingletonRegisterMeta._instances:
            del SingletonRegisterMeta._instances[QdrantVector]

        # 为每个测试创建唯一的目录名，包含测试名和时间戳
        test_name = request.node.name.replace("[", "_").replace("]", "_")
        unique_id = str(uuid.uuid4())[:8]
        timestamp = str(int(time.time() * 1000))
        test_dir = tmp_path / f"qdrant_snowflake_{test_name}_{unique_id}_{timestamp}"

        vector_store = QdrantVector(embedder=mock_embedder, persist_directory=test_dir)
        await vector_store.init()

        yield vector_store

        # 测试后清理：再次清理单例缓存，确保下个测试获得新实例
        try:
            if QdrantVector in SingletonRegisterMeta._instances:
                del SingletonRegisterMeta._instances[QdrantVector]
        except Exception:
            # 忽略清理错误
            pass

    @pytest.fixture
    def snowflake_documents(self):
        """创建使用雪花ID的测试文档"""
        return [
            Document(
                content="Test document with snowflake ID 1",
                metadata={"category": "test", "type": "snowflake", "priority": 1},
                uid="1234567890123456789",  # 典型19位雪花ID
                summary="First snowflake document",
            ),
            Document(
                content="Test document with snowflake ID 2",
                metadata={"category": "test", "type": "snowflake", "priority": 2},
                uid="987654321098765432",  # 另一个18位雪花ID
                summary="Second snowflake document",
            ),
            Document(
                content="Test document with shorter ID",
                metadata={"category": "test", "type": "snowflake", "priority": 3},
                uid="123456789",  # 较短的9位数字ID
                summary="Shorter ID document",
            ),
        ]

    @pytest.mark.asyncio
    async def test_is_snowflake_id(self, qdrant_vector):
        """测试雪花ID检测函数"""
        # 测试典型雪花ID
        assert qdrant_vector._is_snowflake_id("1234567890123456789") is True
        assert qdrant_vector._is_snowflake_id("987654321098765432") is True
        assert qdrant_vector._is_snowflake_id("123456789012") is True  # 12位数字

        # 测试边界情况
        assert qdrant_vector._is_snowflake_id("1234567890") is True  # 刚好10位
        assert qdrant_vector._is_snowflake_id("123456789") is True  # 9位，仍然是有效的雪花ID
        assert qdrant_vector._is_snowflake_id("123") is True  # 3位，最小长度

        # 测试太短的数字
        assert qdrant_vector._is_snowflake_id("12") is False  # 2位，小于最小阈值

        # 测试非数字字符串
        assert qdrant_vector._is_snowflake_id("abc123") is False
        assert qdrant_vector._is_snowflake_id("123abc456") is False
        assert qdrant_vector._is_snowflake_id("") is False

        # 测试UUID格式
        assert qdrant_vector._is_snowflake_id("550e8400-e29b-41d4-a716-446655440000") is False

    @pytest.mark.asyncio
    async def test_is_uuid_format(self, qdrant_vector):
        """测试UUID格式检测函数"""
        import uuid

        # 测试标准UUID
        test_uuid = str(uuid.uuid4())
        assert qdrant_vector._is_uuid_format(test_uuid) is True
        assert qdrant_vector._is_uuid_format("550e8400-e29b-41d4-a716-446655440000") is True

        # 测试非UUID格式
        assert qdrant_vector._is_uuid_format("1234567890123456789") is False
        assert qdrant_vector._is_uuid_format("not-a-uuid") is False
        assert qdrant_vector._is_uuid_format("") is False

    @pytest.mark.asyncio
    async def test_snowflake_to_uuid_conversion(self, qdrant_vector):
        """测试雪花ID到UUID的转换"""
        test_cases = [
            "1234567890123456789",  # 典型19位雪花ID
            "987654321098765432",  # 另一个18位雪花ID
            "123456789012345",  # 15位雪花ID
        ]

        for snowflake_id in test_cases:
            uuid_id = qdrant_vector._snowflake_to_uuid(snowflake_id)

            # 验证结果是有效的UUID
            assert qdrant_vector._is_uuid_format(uuid_id) is True

            # 验证转换的一致性（同样的输入产生同样的输出）
            uuid_id2 = qdrant_vector._snowflake_to_uuid(snowflake_id)
            assert uuid_id == uuid_id2

    @pytest.mark.asyncio
    async def test_uuid_to_snowflake_conversion(self, qdrant_vector):
        """测试UUID到雪花ID的转换"""
        test_snowflake_ids = [
            "1234567890123456789",
            "987654321098765432",
            "123456789012345",
        ]

        for original_snowflake_id in test_snowflake_ids:
            # 转换为UUID再转换回来
            uuid_id = qdrant_vector._snowflake_to_uuid(original_snowflake_id)
            recovered_snowflake_id = qdrant_vector._uuid_to_snowflake(uuid_id)

            # 验证能够正确恢复
            assert recovered_snowflake_id == original_snowflake_id

    @pytest.mark.asyncio
    async def test_uuid_passthrough(self, qdrant_vector):
        """测试UUID应该直接通过转换函数"""
        import uuid

        test_uuid = str(uuid.uuid4())

        # UUID应该直接通过snowflake_to_uuid
        result_uuid = qdrant_vector._snowflake_to_uuid(test_uuid)
        assert result_uuid == test_uuid

        # UUID应该直接通过uuid_to_snowflake
        result_uuid2 = qdrant_vector._uuid_to_snowflake(test_uuid)
        assert result_uuid2 == test_uuid

    @pytest.mark.asyncio
    async def test_add_documents_with_snowflake_ids(self, qdrant_vector, snowflake_documents):
        """测试使用雪花ID添加文档"""
        index_name = "test_snowflake_add"

        # 添加文档
        added_docs = await qdrant_vector.add_documents(index_name, snowflake_documents)

        # 验证返回的文档数量
        assert len(added_docs) == len(snowflake_documents)

        # 验证返回的文档ID仍然是原始的雪花ID格式
        for i, (original_doc, added_doc) in enumerate(zip(snowflake_documents, added_docs)):
            assert added_doc.uid == original_doc.uid, f"Document {i}: ID mismatch"
            assert added_doc.content == original_doc.content, f"Document {i}: Content mismatch"
            assert added_doc.metadata == original_doc.metadata, f"Document {i}: Metadata mismatch"

    @pytest.mark.asyncio
    async def test_get_document_by_snowflake_id(self, qdrant_vector, snowflake_documents):
        """测试通过雪花ID获取文档"""
        index_name = "test_snowflake_get"

        # 先添加文档
        await qdrant_vector.add_documents(index_name, snowflake_documents)

        # 通过雪花ID获取文档
        for original_doc in snowflake_documents:
            retrieved_doc = await qdrant_vector.get(index_name, original_doc.uid)

            assert retrieved_doc.uid == original_doc.uid
            assert retrieved_doc.content == original_doc.content
            assert retrieved_doc.metadata["category"] == original_doc.metadata["category"]

    @pytest.mark.asyncio
    async def test_get_documents_by_snowflake_ids(self, qdrant_vector, snowflake_documents):
        """测试通过雪花ID列表批量获取文档"""
        index_name = "test_snowflake_get_batch"

        # 先添加文档
        await qdrant_vector.add_documents(index_name, snowflake_documents)

        # 批量获取文档
        snowflake_ids = [doc.uid for doc in snowflake_documents]
        retrieved_docs = await qdrant_vector.get_by_ids(index_name, snowflake_ids)

        # 验证返回的文档数量和顺序
        assert len(retrieved_docs) == len(snowflake_documents)

        # 验证返回的文档顺序与请求的ID顺序一致
        for i, (requested_id, retrieved_doc) in enumerate(zip(snowflake_ids, retrieved_docs)):
            assert retrieved_doc.uid == requested_id, f"Document {i}: ID order mismatch"

    @pytest.mark.asyncio
    async def test_delete_documents_by_snowflake_ids(self, qdrant_vector, snowflake_documents):
        """测试通过雪花ID删除文档"""
        index_name = "test_snowflake_delete"

        # 先添加文档
        await qdrant_vector.add_documents(index_name, snowflake_documents)

        # 删除部分文档
        ids_to_delete = [snowflake_documents[0].uid, snowflake_documents[1].uid]
        success = await qdrant_vector.delete(index_name, ids_to_delete)
        assert success is True

        # 验证删除的文档不再存在
        for deleted_id in ids_to_delete:
            with pytest.raises(ValueError, match=f"Document with ID {deleted_id} not found"):
                await qdrant_vector.get(index_name, deleted_id)

        # 验证未删除的文档仍然存在
        remaining_doc = await qdrant_vector.get(index_name, snowflake_documents[2].uid)
        assert remaining_doc.uid == snowflake_documents[2].uid

    @pytest.mark.asyncio
    async def test_metadata_search_with_snowflake_ids(self, qdrant_vector, snowflake_documents):
        """测试使用雪花ID的元数据搜索"""
        index_name = "test_snowflake_metadata"

        # 先添加文档
        await qdrant_vector.add_documents(index_name, snowflake_documents)

        # 通过元数据搜索
        results = await qdrant_vector.select_on_metadata(index_name, {"category": "test"})

        # 验证搜索结果
        assert len(results) == len(snowflake_documents)

        # 验证返回的文档仍然使用雪花ID
        result_ids = [doc.uid for doc in results]
        expected_ids = [doc.uid for doc in snowflake_documents]

        for expected_id in expected_ids:
            assert expected_id in result_ids

    @pytest.mark.asyncio
    async def test_similarity_search_with_snowflake_ids(self, qdrant_vector, snowflake_documents):
        """测试使用雪花ID的相似度搜索"""
        index_name = "test_snowflake_similarity"

        # 先添加文档
        await qdrant_vector.add_documents(index_name, snowflake_documents)

        # 执行相似度搜索
        results = await qdrant_vector.asimilar_search_with_scores(index_name, "test document", k=2)

        # 验证搜索结果
        assert len(results) <= 2

        for doc, score in results:
            # 验证返回的文档使用雪花ID
            assert qdrant_vector._is_snowflake_id(doc.uid) or qdrant_vector._is_uuid_format(doc.uid)
            assert isinstance(score, (int, float))

    @pytest.mark.asyncio
    async def test_mixed_id_formats(self, qdrant_vector):
        """测试混合ID格式的处理"""
        import uuid

        index_name = "test_mixed_ids"

        # 创建包含不同ID格式的文档
        mixed_docs = [
            Document(
                content="Document with snowflake ID",
                metadata={"type": "snowflake"},
                uid="1234567890123456789",  # 雪花ID
                summary="Snowflake ID doc",
            ),
            Document(
                content="Document with UUID",
                metadata={"type": "uuid"},
                uid=str(uuid.uuid4()),  # UUID
                summary="UUID doc",
            ),
            Document(
                content="Document with string ID",
                metadata={"type": "string"},
                uid="custom_string_id_12345",  # 自定义字符串ID
                summary="String ID doc",
            ),
        ]

        # 添加文档
        added_docs = await qdrant_vector.add_documents(index_name, mixed_docs)
        assert len(added_docs) == len(mixed_docs)

        # 验证每个文档都能正确检索
        for original_doc in mixed_docs:
            retrieved_doc = await qdrant_vector.get(index_name, original_doc.uid)
            assert retrieved_doc.uid == original_doc.uid
            assert retrieved_doc.content == original_doc.content

    @pytest.mark.asyncio
    async def test_conversion_consistency(self, qdrant_vector):
        """测试转换一致性"""
        test_ids = [
            "1234567890123456789",
            "987654321098765432",
            "555444333222111000",
        ]

        for test_id in test_ids:
            # 多次转换应该产生相同结果
            uuid1 = qdrant_vector._snowflake_to_uuid(test_id)
            uuid2 = qdrant_vector._snowflake_to_uuid(test_id)
            assert uuid1 == uuid2

            # 往返转换应该恢复原始值
            recovered = qdrant_vector._uuid_to_snowflake(uuid1)
            assert recovered == test_id

    @pytest.mark.asyncio
    async def test_edge_case_ids(self, qdrant_vector):
        """测试边界情况的ID处理"""
        # 测试最小的10位雪花ID
        min_snowflake = "1234567890"
        uuid_result = qdrant_vector._snowflake_to_uuid(min_snowflake)
        assert qdrant_vector._is_uuid_format(uuid_result)
        recovered = qdrant_vector._uuid_to_snowflake(uuid_result)
        assert recovered == min_snowflake

        # 测试空字符串和None（应该被安全处理）
        empty_result = qdrant_vector._snowflake_to_uuid("")
        assert qdrant_vector._is_uuid_format(empty_result)

        # 测试非常长的数字字符串
        very_long_id = "1" * 50  # 50位数字
        long_uuid = qdrant_vector._snowflake_to_uuid(very_long_id)
        assert qdrant_vector._is_uuid_format(long_uuid)
        # 非常长的ID可能使用哈希，所以不一定能完全恢复
