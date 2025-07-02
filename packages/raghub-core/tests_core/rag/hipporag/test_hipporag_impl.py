"""
HippoRAG 实现单元测试
"""

import asyncio
from typing import Any, Dict, List, Optional, Set, Tuple
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from raghub_core.chat.base_chat import BaseChat
from raghub_core.embedding import BaseEmbedding
from raghub_core.rag.hipporag.hipporag_impl import HippoRAGImpl
from raghub_core.rag.hipporag.hipporag_storage import HipporagStorage
from raghub_core.rerank.base_rerank import BaseRerank
from raghub_core.schemas.chat_response import ChatResponse
from raghub_core.schemas.document import Document
from raghub_core.schemas.graph_model import GraphCommunity, GraphEdge, GraphVertex, Namespace, RelationType
from raghub_core.schemas.hipporag_models import OpenIEInfo
from raghub_core.schemas.openie_mdoel import OpenIEModel
from raghub_core.schemas.rag_model import RetrieveResultItem
from raghub_core.storage.graph import GraphStorage
from raghub_core.storage.vector import VectorStorage


class MockLLM(BaseChat):
    """模拟的语言模型"""

    name = "mock_llm"

    def __init__(self):
        pass

    async def astream(self, qa_prompt=None, input=None, **kwargs):
        """模拟流式响应"""
        # 处理不同类型的 prompt
        prompt_text = ""
        if qa_prompt is not None:
            if hasattr(qa_prompt, "format"):
                try:
                    prompt_text = qa_prompt.format(**(input or {}))
                except Exception:
                    prompt_text = str(qa_prompt)
            else:
                prompt_text = str(qa_prompt)

        # 模拟基于输入的响应
        if "Python" in prompt_text:
            response_content = (
                "Python is a high-level programming language that is widely used for "
                "web development, data analysis, artificial intelligence, and more."
            )
        else:
            response_content = "This is a mock response to your question."

        yield ChatResponse(content=response_content, tokens=25)

    async def achat(self, prompt, input=None, output_parser=None, **kwargs):
        """模拟异步聊天"""
        return ChatResponse(content="This is a mock response.", tokens=25)


class MockRerank(BaseRerank):
    """模拟的重排序器"""

    def __init__(self):
        super().__init__(model_name="mock_rerank")

    async def rerank(self, *args, **kwargs) -> Any:
        """模拟重排序，支持不同的参数格式"""
        if len(args) >= 2:
            # 处理 (query, documents) 格式
            _, documents = args[0], args[1]
            if isinstance(documents, list):
                return documents  # 直接返回文档，不重排序

        # 处理其他格式
        return kwargs.get("documents", [])


class MockEmbedding(BaseEmbedding):
    """模拟的嵌入模型"""

    name = "mock_embedding"

    def __init__(self):
        pass

    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """模拟编码文本"""
        return np.random.rand(len(texts), 384)

    def encode_query(self, query: str, **kwargs) -> np.ndarray:
        """模拟编码查询"""
        return np.random.rand(384)

    async def aencode(self, texts: List[str], **kwargs) -> np.ndarray:
        """模拟异步编码文本"""
        return self.encode(texts, **kwargs)

    async def aencode_query(self, queries, **kwargs) -> np.ndarray:
        """模拟异步编码查询"""
        if isinstance(queries, list):
            # 如果是列表，返回多个嵌入
            return np.random.rand(len(queries), 384)
        else:
            # 如果是单个查询，返回单个嵌入
            return np.random.rand(384)


class MockVectorStorage(VectorStorage):
    """模拟的向量存储"""

    name = "mock_vector_storage"

    def __init__(self):
        self.documents = {}  # index_name -> List[Document]

    async def init(self):
        """初始化"""
        pass

    async def create_index(self, index_name: str) -> None:
        """创建索引"""
        if index_name not in self.documents:
            self.documents[index_name] = []

    async def add_documents(self, index_name: str, documents: List[Document]) -> List[Document]:
        """添加文档"""
        if index_name not in self.documents:
            self.documents[index_name] = []

        for doc in documents:
            if not doc.embedding:
                doc.embedding = np.random.rand(384).tolist()

        self.documents[index_name].extend(documents)
        return documents

    async def get_by_ids(self, index_name: str, ids: List[str]) -> List[Document]:
        """根据ID获取文档"""
        if index_name not in self.documents:
            return []

        docs = self.documents[index_name]
        return [doc for doc in docs if doc.uid in ids]

    async def delete(self, index_name: str, ids: List[str]) -> bool:
        """删除文档"""
        if index_name not in self.documents:
            return False

        original_count = len(self.documents[index_name])
        self.documents[index_name] = [doc for doc in self.documents[index_name] if doc.uid not in ids]
        return len(self.documents[index_name]) < original_count

    async def get(self, index_name: str, uid: str) -> Document:
        """获取单个文档"""
        if index_name not in self.documents:
            raise ValueError(f"Index {index_name} not found")

        for doc in self.documents[index_name]:
            if doc.uid == uid:
                return doc
        raise ValueError(f"Document with uid {uid} not found")

    async def select_on_metadata(self, index_name: str, metadata_filter: Dict[str, Any]) -> List[Document]:
        """根据元数据选择文档"""
        if index_name not in self.documents:
            return []

        docs = self.documents[index_name]
        filtered_docs = []
        for doc in docs:
            if all(doc.metadata.get(key) == value for key, value in metadata_filter.items()):
                filtered_docs.append(doc)
        return filtered_docs

    async def similarity_search_by_vector(
        self, index_name: str, embedding: List[float], k: int, filter: Optional[Dict[str, str]] = None
    ) -> List[Tuple[Document, float]]:
        """根据向量搜索相似文档"""
        if index_name not in self.documents:
            return []

        docs = self.documents[index_name]
        # 应用过滤器
        if filter:
            filtered_docs = []
            for doc in docs:
                if all(doc.metadata.get(key) == value for key, value in filter.items()):
                    filtered_docs.append(doc)
            docs = filtered_docs

        # 返回前k个文档，分数递减
        results = []
        for i, doc in enumerate(docs[:k]):
            score = 0.9 - (i * 0.1)
            results.append((doc, score))
        return results

    async def asimilar_search_with_scores(
        self, index_name: str, query: str, k: int = 10, filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """模拟相似搜索"""
        if index_name not in self.documents:
            return []

        docs = self.documents[index_name]
        # 应用过滤器
        if filter:
            filtered_docs = []
            for doc in docs:
                if all(doc.metadata.get(key) == value for key, value in filter.items()):
                    filtered_docs.append(doc)
            docs = filtered_docs

        # 返回前k个文档，分数递减
        results = []
        for i, doc in enumerate(docs[:k]):
            score = 0.9 - (i * 0.1)
            results.append((doc, score))
        return results

    def knn(
        self, query_docs: List[Document], target_docs: List[Document], k: int, **kwargs
    ) -> Dict[str, Tuple[List[str], List[float]]]:
        """模拟KNN搜索"""
        result = {}
        for query_doc in query_docs:
            neighbors = []
            scores = []
            for i, target_doc in enumerate(target_docs[:k]):
                score = 0.9 - (i * 0.1)
                neighbors.append(target_doc.uid)
                scores.append(score)
            result[query_doc.uid] = (neighbors, scores)
        return result


class MockGraphStorage(GraphStorage):
    """模拟的图存储"""

    name = "mock_graph_storage"

    def __init__(self):
        self.vertices = {}  # index_name -> List[GraphVertex]
        self.edges = {}  # index_name -> List[GraphEdge]

    async def init(self):
        """初始化"""
        pass

    async def aadd_new_edges(self, label: str, edges: List[GraphEdge]):
        """添加新边"""
        if label not in self.edges:
            self.edges[label] = []
        self.edges[label].extend(edges)

    async def aadd_graph_edges(self, label: str, edges: List[GraphEdge]):
        """添加图边"""
        if label not in self.edges:
            self.edges[label] = []
        self.edges[label].extend(edges)

    async def aget_by_ids(self, label: str, ids: List[str]) -> List[Document]:
        """根据ID获取文档"""
        return []  # Mock implementation

    async def aadd_vertices(self, label: str, vertices: List[GraphVertex]):
        """添加顶点"""
        if label not in self.vertices:
            self.vertices[label] = []
        self.vertices[label].extend(vertices)

    async def aadd_graph_vertices(self, label: str, vertices: List[GraphVertex]):
        """添加图顶点"""
        if label not in self.vertices:
            self.vertices[label] = []
        self.vertices[label].extend(vertices)

    async def aselect_vertices(self, index_name: str, filter_dict: Dict[str, Any]) -> List[GraphVertex]:
        """选择顶点"""
        if index_name not in self.vertices:
            return []

        vertices = self.vertices[index_name]
        if "name_in" in filter_dict:
            names = filter_dict["name_in"]
            return [v for v in vertices if v.name in names]

        return vertices

    async def aupsert_virtices(self, index_name: str, vertices: List[GraphVertex]):
        """插入或更新顶点"""
        if index_name not in self.vertices:
            self.vertices[index_name] = []

        self.vertices[index_name].extend(vertices)

    async def aupsert_edges(self, index_name: str, edges: List[GraphEdge]):
        """插入或更新边"""
        if index_name not in self.edges:
            self.edges[index_name] = []

        self.edges[index_name].extend(edges)

    async def aselect_edges(self, index_name: str, filter_dict: Optional[Dict[str, Any]] = None) -> List[GraphEdge]:
        """选择边"""
        if index_name not in self.edges:
            return []
        return self.edges[index_name]

    async def aselect_vertices_group_by_graph(self, label: str, filter_dict: Dict[str, Any]) -> List[GraphVertex]:
        """按图分组选择顶点"""
        return await self.aselect_vertices(label, filter_dict)

    async def aupdate_edges(self, label: str, edges: List[GraphEdge]):
        """更新边"""
        await self.aadd_graph_edges(label, edges)

    async def aupdate_vertices(self, label: str, vertices: List[GraphVertex]):
        """更新顶点"""
        await self.aadd_graph_vertices(label, vertices)

    async def adelete_vertices(self, label: str, vertex_ids: List[str]) -> bool:
        """删除顶点"""
        if label not in self.vertices:
            return False

        original_count = len(self.vertices[label])
        self.vertices[label] = [v for v in self.vertices[label] if v.uid not in vertex_ids]
        return len(self.vertices[label]) < original_count

    async def apersonalized_pagerank(
        self, label: str, vertices_with_weight: Dict[str, float], damping: float = 0.85, top_k: int = 10, **kwargs
    ) -> Dict[str, float]:
        """个性化PageRank"""
        # Mock implementation - 返回输入权重的子集
        return dict(list(vertices_with_weight.items())[:top_k])

    async def asearch_neibors(self, label: str, node_id: str, hop: int = 1) -> List[str]:
        """搜索邻居"""
        return []  # Mock implementation

    def discover_communities(self, label: str) -> List[GraphCommunity]:
        """发现社区"""
        return []  # Mock implementation

    def freestyle_search(self, label: str, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """自由搜索"""
        return []  # Mock implementation

    def get_community(self, label: str, community_id: str) -> Optional[GraphCommunity]:
        """获取社区"""
        return None  # Mock implementation

    def multi_hop_search(self, label: str, start_node: str, end_node: str, max_hops: int = 3) -> List[List[str]]:
        """多跳搜索"""
        return []  # Mock implementation


class MockHipporagStorage(HipporagStorage):
    """模拟的 Hipporag 存储"""

    name = "mock_hipporag_storage"

    def __init__(self):
        self.indices = set()
        self.openie_data = {}  # label -> Dict[str, OpenIEInfo]
        self.ent_node_to_chunks = {}  # label -> Dict[str, List[str]]
        self.node_to_node_stats = {}  # label -> Dict[Tuple[str, str], float]
        self.triples_to_docs = {}  # label -> Dict[str, List[str]]
        self.nodes_cache = {}  # label -> Dict[str, Any]

    async def init(self):
        """初始化"""
        pass

    def create_new_index(self, label: str):
        """创建新索引"""
        self.indices.add(label)

    async def save_openie_info(self, label: str, openie_infos: List[OpenIEInfo]) -> List[Document]:
        """保存 OpenIE 信息"""
        if label not in self.openie_data:
            self.openie_data[label] = {}

        for info in openie_infos:
            self.openie_data[label][info.idx] = info

        return [Document(uid=info.idx, content=info.passage, metadata={"openie": True}) for info in openie_infos]

    async def get_openie_info(self, label: str, keys: List[str]) -> List[OpenIEInfo]:
        """获取 OpenIE 信息"""
        result = []
        if label in self.openie_data:
            for key in keys:
                if key in self.openie_data[label]:
                    result.append(self.openie_data[label][key])
        return result

    async def delete_openie_info(self, label: str, keys: List[str]):
        """删除 OpenIE 信息"""
        if label in self.openie_data:
            for key in keys:
                self.openie_data[label].pop(key, None)

    async def set_ent_node_to_chunk_ids(self, label: str, ent_node_id: str, ent_node_to_chunk_ids: List[str]):
        """设置实体节点到chunk ID的映射"""
        if label not in self.ent_node_to_chunks:
            self.ent_node_to_chunks[label] = {}
        self.ent_node_to_chunks[label][ent_node_id] = ent_node_to_chunk_ids

    async def get_ent_node_to_chunk_ids(self, label: str, ent_node_id: str) -> Optional[List[str]]:
        """获取实体节点到chunk ID的映射"""
        return self.ent_node_to_chunks.get(label, {}).get(ent_node_id)

    async def delete_ent_node_to_chunk_ids(self, label: str, ent_node_ids: List[str]):
        """删除实体节点到chunk ID的映射"""
        if label in self.ent_node_to_chunks:
            for ent_node_id in ent_node_ids:
                self.ent_node_to_chunks[label].pop(ent_node_id, None)

    async def get_ent_node_to_chunk_cache_key(self, label: str, ent_node_id: str) -> Optional[str]:
        """获取实体节点到chunk缓存的key"""
        return f"{label}:{ent_node_id}" if ent_node_id in self.ent_node_to_chunks.get(label, {}) else None

    async def set_node_to_node_stats(self, label: str, from_node_key: str, to_node_key: str, stats: float):
        """设置节点到节点的统计信息"""
        if label not in self.node_to_node_stats:
            self.node_to_node_stats[label] = {}
        self.node_to_node_stats[label][(from_node_key, to_node_key)] = stats

    async def get_node_to_node_stats(self, label: str, from_node_key: str, to_node_key: str) -> Optional[float]:
        """获取节点到节点的统计信息"""
        return self.node_to_node_stats.get(label, {}).get((from_node_key, to_node_key))

    async def delete_node_to_node_stats(self, label: str, node_pairs: List[Tuple[str, str]]):
        """删除节点到节点的统计信息"""
        if label in self.node_to_node_stats:
            for pair in node_pairs:
                self.node_to_node_stats[label].pop(pair, None)

    async def set_triples_to_docs(self, label: str, triples: Dict[str, Set[str]]):
        """设置三元组到文档的映射"""
        if label not in self.triples_to_docs:
            self.triples_to_docs[label] = {}
        for triple_key, doc_ids in triples.items():
            self.triples_to_docs[label][triple_key] = list(doc_ids)

    async def get_docs_from_triples(self, label: str, triple_key: str) -> Optional[List[str]]:
        """从三元组获取文档"""
        return self.triples_to_docs.get(label, {}).get(triple_key)

    async def delete_triples_to_docs(self, label: str, triple_keys: List[str]):
        """删除三元组到文档的映射"""
        if label in self.triples_to_docs:
            for key in triple_keys:
                self.triples_to_docs[label].pop(key, None)

    async def set_nodes_cache(self, label: str, node_id: str, data: Any):
        """设置节点缓存"""
        if label not in self.nodes_cache:
            self.nodes_cache[label] = {}
        self.nodes_cache[label][node_id] = data

    async def get_nodes_cache(self, label: str, node_id: str) -> Optional[Any]:
        """获取节点缓存"""
        return self.nodes_cache.get(label, {}).get(node_id)

    async def delete_nodes_cache(self, label: str, node_ids: List[str]):
        """删除节点缓存"""
        if label in self.nodes_cache:
            for node_id in node_ids:
                self.nodes_cache[label].pop(node_id, None)


class TestHippoRAGImpl:
    """HippoRAG 实现测试类"""

    @pytest.fixture(scope="function")
    def mock_llm(self):
        """创建模拟LLM"""
        return MockLLM()

    @pytest.fixture(scope="function")
    def mock_rerank(self):
        """创建模拟重排序器"""
        return MockRerank()

    @pytest.fixture(scope="function")
    def mock_embedding(self):
        """创建模拟嵌入模型"""
        return MockEmbedding()

    @pytest.fixture(scope="function")
    def mock_vector_storage(self):
        """创建模拟向量存储"""
        return MockVectorStorage()

    @pytest.fixture(scope="function")
    def mock_graph_storage(self):
        """创建模拟图存储"""
        return MockGraphStorage()

    @pytest.fixture(scope="function")
    def mock_hipporag_storage(self):
        """创建模拟 Hipporag 存储"""
        return MockHipporagStorage()

    @pytest.fixture(scope="function")
    def hipporag_impl(
        self, mock_llm, mock_rerank, mock_embedding, mock_vector_storage, mock_graph_storage, mock_hipporag_storage
    ):
        """创建 HippoRAG 实现实例"""
        return HippoRAGImpl(
            llm=mock_llm,
            reranker=mock_rerank,
            embedder=mock_embedding,
            embedding_store=mock_vector_storage,
            graph_store=mock_graph_storage,
            hipporag_store=mock_hipporag_storage,
            dspy_file_path="/app/configs/filter_llama3.3-70B-Instruct.json",
        )

    @pytest.fixture
    def sample_documents(self):
        """创建示例文档"""
        return [
            Document(
                uid="doc_1",
                content="Python is a high-level programming language known for its simplicity and readability.",
                summary="Introduction to Python programming",
                metadata={"category": "programming", "language": "Python"},
            ),
            Document(
                uid="doc_2",
                content="Machine learning is a subset of artificial intelligence that focuses on algorithms.",
                summary="Machine learning overview",
                metadata={"category": "AI", "topic": "machine learning"},
            ),
        ]

    @pytest.mark.asyncio
    async def test_initialization(self, hipporag_impl):
        """测试初始化"""
        assert hipporag_impl is not None
        assert hipporag_impl._llm is not None
        assert hipporag_impl._reranker is not None
        assert hipporag_impl._embedder is not None
        assert hipporag_impl._embedd_store is not None
        assert hipporag_impl._graph_store is not None
        assert hipporag_impl._db is not None

    @pytest.mark.asyncio
    async def test_create_index(self, hipporag_impl):
        """测试创建索引"""
        index_name = "test_index"

        # Mock create_new_index method to return None (async)
        hipporag_impl._db.create_new_index = AsyncMock()

        # Mock the init method
        with patch.object(hipporag_impl, "init", new_callable=AsyncMock):
            await hipporag_impl.create(index_name)

        hipporag_impl._db.create_new_index.assert_called_once_with(index_name)

    @pytest.mark.asyncio
    async def test_flatten_facts(self, hipporag_impl):
        """测试事实扁平化"""
        chunk_triples = [
            [("Python", "is", "language"), ("Python", "has", "syntax")],
            [("AI", "includes", "ML"), ("Python", "is", "language")],  # 重复的三元组
        ]

        result = hipporag_impl._flatten_facts(chunk_triples)

        # 应该去重并返回唯一的三元组
        assert len(result) == 3
        assert ("Python", "is", "language") in result
        assert ("Python", "has", "syntax") in result
        assert ("AI", "includes", "ML") in result

    @pytest.mark.asyncio
    async def test_extract_entity_nodes(self, hipporag_impl):
        """测试实体节点提取"""
        triples = [("Python", "is", "language"), ("Python", "has", "syntax"), ("AI", "includes", "ML")]

        graph_nodes, triple_entities = hipporag_impl._extract_entity_nodes(triples)

        # 检查唯一的实体节点
        expected_entities = {"Python", "language", "syntax", "AI", "ML"}
        assert set(graph_nodes) == expected_entities

        # 检查每个三元组的实体
        assert len(triple_entities) == 3
        assert set(triple_entities[0]) == {"Python", "language"}
        assert set(triple_entities[1]) == {"Python", "syntax"}
        assert set(triple_entities[2]) == {"AI", "ML"}

    @pytest.mark.asyncio
    async def test_add_embeddings(self, hipporag_impl):
        """测试添加嵌入"""
        index_name = "test_index"
        entity_nodes = ["Python", "AI", "Machine Learning"]
        prefix = "entity_"
        metadata = {"namespace": Namespace.ENTITY.value}

        result = await hipporag_impl.add_embeddings(index_name, entity_nodes, prefix, metadata)

        assert len(result) == 3
        for doc in result:
            assert doc.uid.startswith(prefix)
            assert doc.content in entity_nodes
            assert doc.metadata == metadata
            assert doc.embedding is not None

    @pytest.mark.asyncio
    async def test_add_document_with_openie(self, hipporag_impl):
        """测试添加文档并进行 OpenIE 提取"""
        index_name = "test_index"
        doc = Document(uid="test_doc", content="Python is a programming language used for AI.", metadata={})

        # Mock OpenIE extraction
        mock_ie_result = OpenIEModel(
            triples=[("Python", "is", "programming language"), ("Python", "used for", "AI")], ner=["Python", "AI"]
        )

        with patch.object(hipporag_impl._openie, "extract", new_callable=AsyncMock) as mock_extract:
            mock_extract.return_value = mock_ie_result

            result = await hipporag_impl._add_document(index_name, doc)

            assert result.uid == "test_doc"
            assert result.metadata["namespace"] == Namespace.PASSAGE.value
            assert "entities" in result.metadata
            assert "facts" in result.metadata

    @pytest.mark.asyncio
    async def test_add_documents_batch(self, hipporag_impl, sample_documents):
        """测试批量添加文档"""
        index_name = "test_index"

        # Mock OpenIE extraction for each document
        mock_ie_result = OpenIEModel(triples=[("Python", "is", "language")], ner=["Python"])

        with patch.object(hipporag_impl._openie, "extract", new_callable=AsyncMock) as mock_extract:
            mock_extract.return_value = mock_ie_result

            result = await hipporag_impl.add_documents(index_name, sample_documents)

            assert len(result) == 2
            for doc in result:
                assert doc.metadata["namespace"] == Namespace.PASSAGE.value

    @pytest.mark.asyncio
    async def test_get_query_embeddings(self, hipporag_impl):
        """测试获取查询嵌入"""
        queries = ["What is Python?", "How does AI work?"]

        result = await hipporag_impl._get_query_embeddings(queries)

        assert "triple" in result
        assert "passage" in result

        for query in queries:
            assert query in result["triple"]
            assert query in result["passage"]
            assert isinstance(result["triple"][query], np.ndarray)
            assert isinstance(result["passage"][query], np.ndarray)

    @pytest.mark.asyncio
    async def test_dense_passage_retrieval(self, hipporag_impl, sample_documents):
        """测试密集段落检索"""
        index_name = "test_index"
        query = "Python programming"

        # 先添加文档到存储
        await hipporag_impl._embedd_store.add_documents(index_name, sample_documents)

        # 提供查询嵌入，格式正确
        query_to_embedding = {"passage": {query: np.random.rand(384)}}

        result = await hipporag_impl.dense_passage_retrieval(index_name, query, query_to_embedding)

        assert isinstance(result, list)
        # 结果应该是 Document, score 的元组列表
        if result:
            assert isinstance(result[0], tuple)
            assert len(result[0]) == 2

    @pytest.mark.asyncio
    async def test_embedding_retrieve(self, hipporag_impl, sample_documents):
        """测试嵌入检索"""
        index_name = "test_index"
        queries = ["Python programming"]

        # 添加带有正确命名空间的文档
        for doc in sample_documents:
            doc.metadata = doc.metadata or {}
            doc.metadata["namespace"] = Namespace.PASSAGE.value

        await hipporag_impl._embedd_store.add_documents(index_name, sample_documents)

        result = await hipporag_impl.embedding_retrieve(index_name, queries)

        assert queries[0] in result
        assert isinstance(result[queries[0]], list)

        # 检查返回的项目类型
        if result[queries[0]]:
            item = result[queries[0]][0]
            assert isinstance(item, RetrieveResultItem)
            assert hasattr(item, "document")
            assert hasattr(item, "score")

    @pytest.mark.asyncio
    async def test_qa_basic(self, hipporag_impl, sample_documents):
        """测试基本问答功能"""
        index_name = "test_index"
        query = "What is Python?"

        # 设置文档的命名空间
        for doc in sample_documents:
            doc.metadata = doc.metadata or {}
            doc.metadata["namespace"] = Namespace.PASSAGE.value

        await hipporag_impl._embedd_store.add_documents(index_name, sample_documents)

        # Mock hybrid_retrieve
        mock_retrieve_result = {
            query: [RetrieveResultItem(document=sample_documents[0], score=0.9, query=query, metadata={})]
        }

        with patch.object(hipporag_impl, "hybrid_retrieve", new_callable=AsyncMock) as mock_retrieve:
            mock_retrieve.return_value = mock_retrieve_result

            responses = []
            async for response in hipporag_impl.qa(index_name, query):
                responses.append(response)

            assert len(responses) > 0
            assert isinstance(responses[0], ChatResponse)

    @pytest.mark.asyncio
    async def test_delete_documents(self, hipporag_impl, sample_documents):
        """测试删除文档"""
        index_name = "test_index"
        doc_id = sample_documents[0].uid

        # 先准备一些 OpenIE 信息来模拟删除
        openie_info = OpenIEInfo(
            idx=doc_id,
            passage=sample_documents[0].content,
            extracted_triples=[["Python", "is", "language"]],
            extracted_entities=["Python", "language"],
        )

        # Mock get_openie_info 返回数据
        with patch.object(hipporag_impl._db, "get_openie_info", new_callable=AsyncMock) as mock_get_openie:
            mock_get_openie.return_value = [openie_info]

            # Mock 其他相关方法
            with patch.object(hipporag_impl._db, "get_docs_from_triples", new_callable=AsyncMock) as mock_get_docs:
                mock_get_docs.return_value = [doc_id]

                with patch.object(
                    hipporag_impl._db, "get_ent_node_to_chunk_ids", new_callable=AsyncMock
                ) as mock_get_ent:
                    mock_get_ent.return_value = [doc_id]

                    # Mock 向量存储删除方法
                    with patch.object(hipporag_impl._embedd_store, "delete", new_callable=AsyncMock) as mock_delete:
                        await hipporag_impl.delete(index_name, [doc_id])

                        # 验证删除方法被调用
                        mock_delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_ppr(self, hipporag_impl):
        """测试 PageRank 算法"""
        index_name = "test_index"
        node_weights = {"node1": 0.5, "node2": 0.3, "node3": 0.2}
        damping = 0.85
        top_k = 2

        # Mock the graph storage to return some edges
        mock_edges = [
            GraphEdge(
                source="node1",
                target="node2",
                relation="connects",
                uid="edge1",
                weight=1.0,
                relation_type=RelationType.RELATION,
                label=index_name,
                metadata={},
            )
        ]

        with patch.object(hipporag_impl._graph_store, "aselect_edges", new_callable=AsyncMock) as mock_edges_query:
            mock_edges_query.return_value = mock_edges

            result = await hipporag_impl.run_ppr(index_name, node_weights, damping, top_k)

            assert isinstance(result, dict)
            # PPR 应该返回节点权重
            assert len(result) <= top_k

    @pytest.mark.asyncio
    async def test_vertices_nodes_conversion(self, hipporag_impl):
        """测试顶点节点转换"""
        index_name = "test_index"

        # 测试字典到 GraphVertex 的转换
        vertices_dict = [
            {
                "uid": "vertex1",
                "name": "Python",
                "content": "Python programming language",
                "metadata": {"namespace": Namespace.ENTITY.value},
                "embedding": [0.1, 0.2, 0.3],
            }
        ]

        graph_vertices = hipporag_impl._vertices_nodes_to_graph_vertices(index_name, vertices_dict)

        assert len(graph_vertices) == 1
        vertex = graph_vertices[0]
        assert isinstance(vertex, GraphVertex)
        assert vertex.label == index_name
        assert vertex.uid == "vertex1"
        assert vertex.name == "Python"
        assert vertex.content == "Python programming language"

        # 测试 GraphVertex 到字典的转换
        nodes_dict = hipporag_impl._graph_vertices_to_nodes(graph_vertices)

        assert len(nodes_dict) == 1
        node = nodes_dict[0]
        assert node["uid"] == "vertex1"
        assert node["name"] == "Python"
        assert node["content"] == "Python programming language"

    @pytest.mark.asyncio
    async def test_edge_to_graph_edge_conversion(self, hipporag_impl):
        """测试边到 GraphEdge 的转换"""
        index_name = "test_index"
        edge_nodes = {("node1", "node2"): 0.8, ("node2", "node3"): 0.6}

        graph_edges = hipporag_impl._edge_to_graph_edge(index_name, edge_nodes)

        assert len(graph_edges) == 2

        for edge in graph_edges:
            assert isinstance(edge, GraphEdge)
            assert edge.label == index_name
            assert edge.relation_type == RelationType.RELATION
            assert edge.weight in [0.8, 0.6]

    @pytest.mark.asyncio
    async def test_add_new_nodes(self, hipporag_impl):
        """测试添加新节点"""
        index_name = "test_index"

        entities = [Document(uid="entity1", content="Python", metadata={"namespace": Namespace.ENTITY.value})]
        passages = [
            Document(uid="doc_passage1", content="Python tutorial", metadata={"namespace": Namespace.PASSAGE.value})
        ]

        # Mock existing nodes (empty)
        with patch.object(hipporag_impl._graph_store, "aselect_vertices", new_callable=AsyncMock) as mock_select:
            mock_select.return_value = []

            with patch.object(hipporag_impl._graph_store, "aupsert_virtices", new_callable=AsyncMock) as mock_upsert:
                new_entities, new_passages = await hipporag_impl._add_new_nodes(index_name, entities, passages)

                assert len(new_entities) == 1
                assert len(new_passages) == 1
                assert new_entities[0].uid == "entity1"
                assert new_passages[0].uid == "doc_passage1"

                # 验证图存储被调用
                mock_upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_handling_empty_content(self, hipporag_impl):
        """测试空内容的错误处理"""
        index_name = "test_index"
        doc = Document(
            uid="empty_doc",
            content="",  # 空内容
            metadata={},
        )

        # Mock OpenIE 返回空结果
        mock_ie_result = OpenIEModel(triples=[], ner=[])

        with patch.object(hipporag_impl._openie, "extract", new_callable=AsyncMock) as mock_extract:
            mock_extract.return_value = mock_ie_result

            result = await hipporag_impl._add_document(index_name, doc)

            # 应该正常处理，但可能没有提取到实体
            assert result.uid == "empty_doc"

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, hipporag_impl, sample_documents):
        """测试并发操作"""
        index_name = "test_index"

        # 简化的并发测试 - 只测试嵌入存储的并发添加
        tasks = []
        for i in range(3):
            docs = [
                Document(uid=f"concurrent_doc_{i}_{j}", content=f"Content {i}-{j}", metadata={"batch": i})
                for j in range(2)
            ]

            # 直接添加到嵌入存储而不是通过复杂的文档处理
            tasks.append(hipporag_impl._embedd_store.add_documents(f"{index_name}_{i}", docs))

        results = await asyncio.gather(*tasks)

        # 验证所有任务完成
        assert len(results) == 3
        for result in results:
            assert len(result) == 2
            assert all(isinstance(doc, Document) for doc in result)

    @pytest.mark.asyncio
    async def test_batch_embedding_operations(self, hipporag_impl):
        """测试批量嵌入操作"""
        index_name = "test_index"
        entity_nodes = ["Entity1", "Entity2", "Entity3", "Entity4", "Entity5"]
        prefix = "batch_entity_"
        metadata = {"namespace": Namespace.ENTITY.value, "batch": "test"}

        result = await hipporag_impl.add_embeddings(index_name, entity_nodes, prefix, metadata)

        assert len(result) == 5
        for doc in result:
            assert doc.uid.startswith(prefix)
            assert doc.content in entity_nodes
            assert doc.metadata == metadata
            assert doc.embedding is not None

    @pytest.mark.asyncio
    async def test_graph_edge_conversion_with_weights(self, hipporag_impl):
        """测试带权重的图边转换"""
        index_name = "test_index"
        edge_nodes = {
            ("node1", "node2"): 0.8,
            ("node2", "node3"): 0.6,
            ("node3", "node1"): 0.9,
            ("node1", "node4"): 0.4,
        }

        graph_edges = hipporag_impl._edge_to_graph_edge(index_name, edge_nodes)

        assert len(graph_edges) == 4

        # 验证边的属性
        weights = [edge.weight for edge in graph_edges]
        assert set(weights) == {0.8, 0.6, 0.9, 0.4}

        for edge in graph_edges:
            assert isinstance(edge, GraphEdge)
            assert edge.label == index_name
            assert edge.relation_type == RelationType.RELATION
            assert edge.relation == ""
            assert edge.uid is not None

    @pytest.mark.asyncio
    async def test_fact_flattening_with_duplicates(self, hipporag_impl):
        """测试带重复的事实扁平化"""
        chunk_triples = [
            [("A", "relates", "B"), ("B", "connects", "C"), ("A", "relates", "B")],  # 重复
            [("C", "links", "D"), ("A", "relates", "B")],  # 又一个重复
        ]

        result = hipporag_impl._flatten_facts(chunk_triples)

        # 应该去除重复，只保留唯一的三元组
        expected_unique = {("A", "relates", "B"), ("B", "connects", "C"), ("C", "links", "D")}

        assert len(result) == 3
        assert set(result) == expected_unique

    @pytest.mark.asyncio
    async def test_entity_extraction_complex(self, hipporag_impl):
        """测试复杂实体提取"""
        triples = [
            ("Machine Learning", "is_part_of", "Artificial Intelligence"),
            ("Deep Learning", "is_subset_of", "Machine Learning"),
            ("Neural Networks", "implements", "Deep Learning"),
            ("TensorFlow", "is_framework_for", "Machine Learning"),
            ("Python", "used_with", "TensorFlow"),
        ]

        graph_nodes, triple_entities = hipporag_impl._extract_entity_nodes(triples)

        expected_entities = {
            "Machine Learning",
            "Artificial Intelligence",
            "Deep Learning",
            "Neural Networks",
            "TensorFlow",
            "Python",
        }

        assert set(graph_nodes) == expected_entities
        assert len(triple_entities) == 5

        # 验证每个三元组的实体提取
        assert set(triple_entities[0]) == {"Machine Learning", "Artificial Intelligence"}
        assert set(triple_entities[1]) == {"Deep Learning", "Machine Learning"}
        assert set(triple_entities[2]) == {"Neural Networks", "Deep Learning"}

    @pytest.mark.asyncio
    async def test_mixed_namespace_documents(self, hipporag_impl):
        """测试混合命名空间的文档处理"""
        index_name = "test_index"

        entities = [Document(uid="entity1", content="Python", metadata={"namespace": Namespace.ENTITY.value})]
        passages = [
            Document(uid="doc_passage1", content="Python tutorial", metadata={"namespace": Namespace.PASSAGE.value})
        ]

        # Mock图存储返回空的已存在节点（全部都是新节点）
        with patch.object(hipporag_impl._graph_store, "aselect_vertices", new_callable=AsyncMock) as mock_select:
            mock_select.return_value = []

            with patch.object(hipporag_impl._graph_store, "aupsert_virtices", new_callable=AsyncMock):
                new_entities, new_passages = await hipporag_impl._add_new_nodes(index_name, entities, passages)

                # 应该根据 uid 前缀正确分类
                assert len(new_entities) == 1
                assert len(new_passages) == 1
                assert new_entities[0].uid == "entity1"
                assert new_passages[0].uid == "doc_passage1"

    @pytest.mark.asyncio
    async def test_qa_with_empty_results(self, hipporag_impl):
        """测试没有检索结果的问答"""
        index_name = "test_index"
        query = "Unknown topic that doesn't exist"

        # Mock hybrid_retrieve 返回空结果
        with patch.object(hipporag_impl, "hybrid_retrieve", new_callable=AsyncMock) as mock_retrieve:
            mock_retrieve.return_value = {query: []}

            responses = []
            async for response in hipporag_impl.qa(index_name, query):
                responses.append(response)

            # 应该仍然产生响应，即使没有检索到相关文档
            assert len(responses) > 0
            assert isinstance(responses[0], ChatResponse)

    @pytest.mark.asyncio
    async def test_error_handling_with_invalid_triples(self, hipporag_impl):
        """测试三元组处理的一致性"""
        # 测试不同长度的三元组都被保留（_flatten_facts只去重，不过滤）
        chunk_triples = [
            [("A", "relates")],  # 短三元组
            [("B", "connects", "C")],  # 标准三元组
            [("D",)],  # 单元素
            [("E", "links", "F", "extra")],  # 长三元组
            [("B", "connects", "C")],  # 重复，应被去除
        ]

        result = hipporag_impl._flatten_facts(chunk_triples)

        # _flatten_facts 只去重，不过滤长度
        assert len(result) == 4  # 去除重复后应该有4个

        # 检查去重效果
        result_set = set(result)
        assert len(result_set) == len(result)  # 确保没有重复

        # 验证重复的三元组被去除
        assert result.count(("B", "connects", "C")) == 1
