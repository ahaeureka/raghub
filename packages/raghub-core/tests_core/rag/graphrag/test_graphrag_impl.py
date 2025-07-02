"""
GraphRAG 实现单元测试
"""

import asyncio
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

import pytest
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from raghub_core.chat.base_chat import BaseChat
from raghub_core.rag.base_rag import BaseGraphRAGDAO
from raghub_core.rag.graphrag.graphrag_impl import GraphRAGImpl
from raghub_core.rag.graphrag.operators import GraphRAGOperators
from raghub_core.rerank.base_rerank import BaseRerank
from raghub_core.schemas.chat_response import ChatResponse
from raghub_core.schemas.document import Document
from raghub_core.schemas.graph_extract_model import GraphExtractOperatorOutputModel
from raghub_core.schemas.graph_model import (
    GraphCommunity,
    GraphEdge,
    GraphModel,
    GraphRAGRetrieveResultItem,
    GraphVertex,
    Namespace,
    QueryIndentationModel,
)
from raghub_core.schemas.keywords_model import KeywordsOperatorOutputModel
from raghub_core.schemas.summarize_model import SummarizeOperatorOutputModel


class MockLLM(BaseChat):
    """模拟的语言模型"""

    name = "mock_llm"

    def __init__(self):
        pass

    def chat(
        self,
        prompt: "ChatPromptTemplate",
        input: Dict[str, str],
        output_parser: Optional[Callable] = None,
    ) -> "ChatResponse":
        """模拟同步聊天"""
        from raghub_core.schemas.chat_response import ChatResponse

        return ChatResponse(content="This is a mock response.", tokens=15)

    async def achat(
        self,
        prompt: "ChatPromptTemplate",
        input: Dict[str, str],
        output_parser: Optional[Callable] = None,
    ) -> "ChatResponse":
        """模拟异步聊天"""
        from raghub_core.schemas.chat_response import ChatResponse

        return ChatResponse(content="This is a mock response.", tokens=15)

    def preprocess_input(self, input: Dict[str, str]) -> Dict[str, str]:
        """预处理输入"""
        return input

    async def astream(
        self,
        prompt: ChatPromptTemplate,
        input: Dict[str, str],
        output_parser: Optional[Callable[[AIMessage], ChatResponse]] = None,
    ) -> AsyncIterator[ChatResponse]:
        """模拟流式响应"""
        from raghub_core.schemas.chat_response import ChatResponse

        # 处理不同类型的 prompt
        prompt_text = ""
        if prompt is not None:
            if hasattr(prompt, "format"):
                # ChatPromptTemplate or similar
                try:
                    prompt_text = prompt.format(**(input or {}))
                except Exception:
                    prompt_text = str(prompt)
            else:
                prompt_text = str(prompt)

        # 模拟基于输入的响应
        if "Python" in prompt_text:
            response_content = (
                "Python is a high-level programming language that is widely used for "
                "web development, data analysis, artificial intelligence, and more."
            )
        else:
            response_content = "This is a mock response to your question."

        yield ChatResponse(
            content=response_content,
            tokens=25,  # 总 token 数
        )


class MockRerank(BaseRerank):
    """模拟的重排序器"""

    def __init__(self):
        super().__init__(model_name="mock_rerank")

    async def rerank(self, *args, **kwargs) -> Any:
        """模拟重排序，支持不同的参数格式"""
        # 处理不同的调用模式
        if len(args) == 3:  # GraphRAG调用: (index_name, query, docs)
            index_name, query, docs = args
            return [(doc.uid, 0.8) for doc in docs]
        elif len(args) == 2:  # 标准调用: (query, documents)
            query, documents = args
            return {doc.uid: 0.8 for doc in documents}
        else:
            # 默认处理
            documents = args[-1] if args else kwargs.get("documents", [])
            return {doc.uid: 0.8 for doc in documents}


class MockGraphRAGDAO(BaseGraphRAGDAO):
    """模拟的图RAG数据访问对象"""

    def __init__(self):
        super().__init__()
        self.documents_storage = {}
        self.vertices_storage = {}
        self.edges_storage = {}
        self.communities_storage = {}

    async def init(self) -> None:
        """初始化存储系统"""
        pass

    async def create(self, index_name: str):
        """创建索引"""
        pass

    async def add_documents(self, index_name: str, documents: List[Document]) -> List[Document]:
        """模拟添加文档"""
        if index_name not in self.documents_storage:
            self.documents_storage[index_name] = []
        self.documents_storage[index_name].extend(documents)
        return documents

    async def add_virtices(self, index_name: str, vertices: List[GraphVertex]) -> List[GraphVertex]:
        """模拟添加顶点"""
        if index_name not in self.vertices_storage:
            self.vertices_storage[index_name] = []
        self.vertices_storage[index_name].extend(vertices)
        return vertices

    async def add_edges(self, index_name: str, edges: List[GraphEdge]) -> List[GraphEdge]:
        """模拟添加边"""
        if index_name not in self.edges_storage:
            self.edges_storage[index_name] = []
        self.edges_storage[index_name].extend(edges)
        return edges

    async def delete(self, index_name: str, doc_ids: List[str] | str) -> None:
        """模拟删除文档"""
        # 确保 doc_ids 是列表
        if isinstance(doc_ids, str):
            doc_ids = [doc_ids]

        # 删除文档
        docs_key = f"{index_name}_docs"
        if docs_key in self.documents_storage:
            self.documents_storage[docs_key] = [
                doc for doc in self.documents_storage[docs_key] if doc.uid not in doc_ids
            ]

        # 删除相关的顶点和边
        for storage in [self.vertices_storage, self.edges_storage]:
            if index_name in storage:
                storage[index_name] = [
                    item for item in storage[index_name] if getattr(item, "uid", None) not in doc_ids
                ]

    async def similar_search_with_scores(
        self, index_name: str, query: str, top_k: int = 10, filter: Optional[Dict[str, str]] = None
    ) -> List[tuple[Document, float]]:
        """模拟相似搜索"""
        if index_name not in self.documents_storage:
            return []

        docs = self.documents_storage[index_name]
        # 简单的模拟：返回前top_k个文档，分数递减
        results = []
        for i, doc in enumerate(docs[:top_k]):
            score = 0.9 - (i * 0.1)  # 分数从0.9递减
            results.append((doc, score))
        return results

    async def search_communities(
        self, label: str, query: str, top_k: int = 5, similar_threshold=0.55
    ) -> List[GraphCommunity]:
        """模拟社区搜索"""
        # 返回模拟的社区
        return [
            GraphCommunity(
                cid="community_1",
                name="Programming Community",
                summary="This is a mock community about programming.",
                graph=GraphModel(vertices=[], edges=[]),
            )
        ]

    async def search_graph_by_indent(
        self, index_name: str, query_indent: QueryIndentationModel
    ) -> Optional[GraphModel]:
        """模拟基于意图的图搜索"""
        if query_indent.entities:
            return GraphModel(
                vertices=[
                    GraphVertex(
                        uid="entity_1",
                        name="Python",
                        content="Python",
                        description={"doc_1": "Python programming language"},
                        metadata={},
                        namespace=Namespace.ENTITY.value,
                        label=index_name,
                    )
                ],
                edges=[],
            )
        return None

    async def explore_trigraph(self, index_name: str, entity_ids: List[str]) -> GraphModel:
        """模拟三元图探索"""
        return GraphModel(
            vertices=[
                GraphVertex(
                    uid=entity_id,
                    name=entity_id.split("_")[-1] if "_" in entity_id else entity_id,
                    content=entity_id.split("_")[-1] if "_" in entity_id else entity_id,
                    description={},
                    metadata={},
                    namespace=Namespace.ENTITY.value,
                    label=index_name,
                )
                for entity_id in entity_ids[:3]  # 限制返回数量
            ],
            edges=[],
        )

    async def get_docs_by_entities(self, index_name: str, entities: List[str]) -> List[Document]:
        """模拟通过实体获取文档"""
        return [
            Document(
                uid="doc_1",
                content="This is a document about Python programming.",
                summary="Python programming guide",
                metadata={"entities": entities[:2]},  # 模拟关联的实体
            )
        ]

    async def discover_communities(self, index_name: str) -> List[str]:
        """模拟发现社区"""
        return ["community_1", "community_2"]

    async def get_community(self, lable: str, community_id: str) -> GraphCommunity:
        """模拟获取社区"""
        return GraphCommunity(
            cid=community_id,
            name=f"Community {community_id}",
            summary="",  # 空摘要，等待后续填充
            graph=GraphModel(
                vertices=[
                    GraphVertex(
                        uid="entity_1",
                        name="Python",
                        content="Python",
                        description={},
                        metadata={},
                        namespace=Namespace.ENTITY.value,
                        label=lable,
                    )
                ],
                edges=[],
            ),
        )

    async def aselect_vertices_group_by_graph(
        self, index_name: str, filter: Dict[str, Any]
    ) -> Dict[str, List[GraphVertex]]:
        """模拟按图分组选择顶点"""
        vertices = self.vertices_storage.get(index_name, [])
        return {"mock_graph": vertices}

    async def get_verteices_by_ids(self, index_name: str, ids: List[str]) -> List[GraphVertex]:
        """模拟通过ID获取顶点"""
        vertices = self.vertices_storage.get(index_name, [])
        return [v for v in vertices if v.uid in ids]

    async def save_openie_info(self, unique_name: str, openie_info) -> List[Document]:
        """模拟保存OpenIE信息"""
        # 转换为文档格式
        docs = [
            Document(uid=f"openie_{i}", content=str(info), summary="OpenIE information", metadata={"type": "openie"})
            for i, info in enumerate(openie_info)
        ]
        return await self.add_documents(unique_name, docs)


class MockGraphRAGOperators(GraphRAGOperators):
    """模拟的图RAG操作器"""

    def __init__(self):
        super().__init__()

    async def extract_graph(self, input: Dict[str, Any], lang="zh") -> GraphExtractOperatorOutputModel:
        """模拟图提取"""
        return GraphExtractOperatorOutputModel(
            name="test_graph_extract_operator",
            entities=[
                ("Python", "Python is a programming language"),
                ("Programming", "Programming is the process of writing code"),
            ],
            triples=[
                ("Python", "is_a", "Programming Language", "Python is a high-level programming language"),
                ("Programming", "uses", "Python", "Programming can be done using Python"),
            ],
        )

    async def extract_keywords(self, input: Dict[str, Any], lang="zh") -> KeywordsOperatorOutputModel:
        """模拟关键词提取"""
        return KeywordsOperatorOutputModel(
            name="test_keywords_operator", keywords=["Python", "programming", "language", "code"]
        )

    async def detect_query_indent(self, input: Dict[str, Any], lang="zh") -> QueryIndentationModel:
        """模拟查询意图检测"""
        return QueryIndentationModel(category="SingleEntitySearch", entities=["Python", "programming"], relations=[])

    async def summarize_communities(self, input: Dict[str, Any], lang="zh") -> SummarizeOperatorOutputModel:
        """模拟社区摘要"""
        return SummarizeOperatorOutputModel(
            name="test_summarize_operator",
            summary="This community focuses on Python programming and related technologies.",
            keywords=["Python", "programming", "technology"],
        )


class TestGraphRAGImpl:
    """GraphRAG 实现测试类"""

    @pytest.fixture(scope="function")
    def mock_llm(self):
        """创建模拟LLM"""
        return MockLLM()

    @pytest.fixture(scope="function")
    def mock_rerank(self):
        """创建模拟重排序器"""
        return MockRerank()

    @pytest.fixture(scope="function")
    def mock_dao(self):
        """创建模拟DAO"""
        return MockGraphRAGDAO()

    @pytest.fixture(scope="function")
    def mock_operators(self):
        """创建模拟操作器"""
        return MockGraphRAGOperators()

    @pytest.fixture(scope="function")  # 改为 function 作用域
    def graphrag_impl(self, mock_llm, mock_rerank, mock_dao, mock_operators):
        """创建GraphRAG实现实例"""
        return GraphRAGImpl(llm=mock_llm, reranker=mock_rerank, dao=mock_dao, operators=mock_operators)

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
    async def test_initialization(self, graphrag_impl):
        """测试GraphRAG初始化"""
        assert graphrag_impl.llm is not None
        assert graphrag_impl.storage is not None
        assert graphrag_impl._operators is not None
        assert graphrag_impl._reranker is not None
        assert graphrag_impl._topk == 5
        assert graphrag_impl._score_threshold == 0.5

    @pytest.mark.asyncio
    async def test_create_index(self, graphrag_impl):
        """测试创建索引"""
        # create方法目前是空实现，测试它不抛出异常
        await graphrag_impl.create("test_index")
        # 由于是空实现，没有具体的断言

    @pytest.mark.asyncio
    async def test_add_documents_single(self, graphrag_impl, sample_documents):
        """测试添加单个文档"""
        index_name = "test_index"
        documents = [sample_documents[0]]

        result = await graphrag_impl.add_documents(index_name, documents)

        assert len(result) == 1
        assert result[0].uid == documents[0].uid
        assert result[0].content == documents[0].content

    @pytest.mark.asyncio
    async def test_add_documents_multiple(self, graphrag_impl, sample_documents):
        """测试添加多个文档"""
        index_name = "test_index"

        result = await graphrag_impl.add_documents(index_name, sample_documents)

        assert len(result) == len(sample_documents)
        for i, doc in enumerate(result):
            assert doc.uid == sample_documents[i].uid
            assert doc.content == sample_documents[i].content

    @pytest.mark.asyncio
    async def test_delete_documents(self, graphrag_impl, sample_documents):
        """测试删除文档"""
        index_name = "test_index"

        # 确保开始时存储是干净的
        graphrag_impl.storage.documents_storage.clear()

        # 先添加文档
        await graphrag_impl.add_documents(index_name, sample_documents)

        # 删除第一个文档
        doc_ids = [sample_documents[0].uid]  # 删除 doc_1
        await graphrag_impl.delete(index_name, doc_ids)

        # 验证删除操作：应该剩下 1 个文档（doc_2）
        remaining_docs = graphrag_impl.storage.documents_storage.get(f"{index_name}_docs", [])
        assert len(remaining_docs) == 1
        assert remaining_docs[0].uid == sample_documents[1].uid  # 确认剩下的是 doc_2

    @pytest.mark.asyncio
    async def test_retrieve_query(self, graphrag_impl, sample_documents):
        """测试查询检索"""
        index_name = "test_index"
        query = "What is Python programming?"

        # 先添加一些文档
        await graphrag_impl.add_documents(index_name, sample_documents)

        result = await graphrag_impl._retrieve_query(index_name, query, top_k=5)

        assert isinstance(result, GraphRAGRetrieveResultItem)
        assert result.query == query
        assert result.context is not None
        assert result.docs is not None
        assert len(result.docs) >= 0

    @pytest.mark.asyncio
    async def test_qa_basic(self, graphrag_impl, sample_documents):
        """测试基本问答功能"""
        index_name = "test_index"
        query = "What is Python?"

        # 先添加文档
        await graphrag_impl.add_documents(index_name, sample_documents)

        responses = []
        async for response in graphrag_impl.qa(index_name, query):
            responses.append(response)

        assert len(responses) > 0
        assert responses[0].question == query
        assert responses[0].answer is not None
        assert responses[0].tokens is not None

    @pytest.mark.asyncio
    async def test_qa_with_history(self, graphrag_impl, sample_documents):
        """测试带历史上下文的问答"""
        index_name = "test_index"
        query = "Tell me more about it"
        history = "We were discussing Python programming language."

        await graphrag_impl.add_documents(index_name, sample_documents)

        responses = []
        async for response in graphrag_impl.qa(index_name, query, history_context=history):
            responses.append(response)

        assert len(responses) > 0
        assert responses[0].question == query

    @pytest.mark.asyncio
    async def test_qa_with_custom_prompt(self, graphrag_impl, sample_documents):
        """测试自定义提示的问答"""
        index_name = "test_index"
        query = "What is Python?"
        custom_prompt = (
            "Answer the question: {question} using this context: {context} "
            "and knowledge graph: {knowledge_graph} and documents: {knowledge_graph_for_doc}"
        )

        await graphrag_impl.add_documents(index_name, sample_documents)

        responses = []
        async for response in graphrag_impl.qa(index_name, query, prompt=custom_prompt):
            responses.append(response)

        assert len(responses) > 0

    @pytest.mark.asyncio
    async def test_qa_with_invalid_prompt(self, graphrag_impl, sample_documents):
        """测试无效提示的问答"""
        index_name = "test_index"
        query = "What is Python?"
        # 缺少必需变量的提示
        invalid_prompt = "Answer the question: {question}"

        await graphrag_impl.add_documents(index_name, sample_documents)

        with pytest.raises(ValueError, match="Prompt is missing required variables"):
            responses = []
            async for response in graphrag_impl.qa(index_name, query, prompt=invalid_prompt):
                responses.append(response)

    @pytest.mark.asyncio
    async def test_retrieve_multiple_queries(self, graphrag_impl, sample_documents):
        """测试多查询检索"""
        index_name = "test_index"
        queries = ["What is Python?", "What is machine learning?"]

        await graphrag_impl.add_documents(index_name, sample_documents)

        results = await graphrag_impl.retrieve(index_name, queries)

        assert len(results) == len(queries)
        for query in queries:
            assert query in results
            assert isinstance(results[query], list)

    @pytest.mark.asyncio
    async def test_embedding_retrieve(self, graphrag_impl, sample_documents):
        """测试嵌入检索"""
        index_name = "test_index"
        queries = ["Python programming"]

        await graphrag_impl.add_documents(index_name, sample_documents)

        results = await graphrag_impl.embedding_retrieve(index_name, queries)

        assert len(results) == len(queries)
        for query in queries:
            assert query in results
            assert isinstance(results[query], list)

    @pytest.mark.asyncio
    async def test_summary_communities(self, graphrag_impl):
        """测试社区摘要"""
        index_name = "test_index"

        communities = await graphrag_impl.summary_communities(index_name)

        assert isinstance(communities, list)
        if communities:
            assert all(isinstance(c, GraphCommunity) for c in communities)
            assert all(c.summary for c in communities)

    @pytest.mark.asyncio
    async def test_search_similar_entities(self, graphrag_impl, sample_documents):
        """测试相似实体搜索"""
        index_name = "test_index"
        keywords = ["Python", "programming"]

        await graphrag_impl.add_documents(index_name, sample_documents)

        entities = await graphrag_impl._search_similar_entities(index_name, keywords)

        assert isinstance(entities, list)

    @pytest.mark.asyncio
    async def test_to_vertices(self, graphrag_impl, sample_documents):
        """测试转换为顶点"""
        index_name = "test_index"
        entities = [
            Document(
                uid="entity_1", content="Python", summary="Python programming language", metadata={"doc_id": "doc_1"}
            )
        ]
        doc = sample_documents[0]
        entities_facts = {"Python": ["Python#is_a#Programming Language#description"]}

        vertices = graphrag_impl._to_virtices(index_name, entities, doc, entities_facts)

        assert len(vertices) == len(entities) + 1  # entities + doc
        assert all(isinstance(v, GraphVertex) for v in vertices)

    @pytest.mark.asyncio
    async def test_to_edges(self, graphrag_impl):
        """测试转换为边"""
        index_name = "test_index"
        extract_result = GraphExtractOperatorOutputModel(
            name="test_extract_operator",
            entities=[("Python", "Programming language")],
            triples=[("Python", "is_a", "Programming Language", "Python is a programming language")],
        )
        doc_id = "doc_1"

        edges = graphrag_impl._to_edges(index_name, extract_result, doc_id)

        assert len(edges) == len(extract_result.triples)
        assert all(isinstance(e, GraphEdge) for e in edges)

    @pytest.mark.asyncio
    async def test_entities_bind_to_docs(self, graphrag_impl, sample_documents):
        """测试实体与文档绑定"""
        entities = [
            Document(uid="entity_1", content="Python", summary="", metadata={}),
            Document(uid="entity_2", content="Programming", summary="", metadata={}),
        ]
        doc = sample_documents[0]

        result = graphrag_impl._entities_bind_to_docs(entities, doc)

        assert "entities" in result.metadata
        assert len(result.metadata["entities"]) == len(entities)

    @pytest.mark.asyncio
    async def test_aload_chunk_context(self, graphrag_impl, sample_documents):
        """测试加载块上下文"""
        index_name = "test_index"

        # 先添加一些历史文档到上下文索引
        context_docs = [
            Document(uid="context_1", content="Python is versatile", summary="Python versatility", metadata={})
        ]
        await graphrag_impl.storage.add_documents(graphrag_impl._context_history_index.format(index_name), context_docs)

        context_map = await graphrag_impl.aload_chunk_context(index_name, sample_documents)

        assert isinstance(context_map, dict)
        assert len(context_map) == len(sample_documents)
        for doc in sample_documents:
            assert doc.content in context_map

    @pytest.mark.asyncio
    async def test_error_handling_add_documents(self, graphrag_impl):
        """测试添加文档时的错误处理"""
        index_name = "test_index"

        # 创建一个会导致错误的文档（例如，空内容可能导致提取失败）
        problematic_doc = Document(
            uid="problematic_doc",
            content="",  # 空内容可能导致提取失败
            summary="",
            metadata={},
        )

        # 根据实际实现，这可能抛出异常或返回空结果
        try:
            result = await graphrag_impl.add_documents(index_name, [problematic_doc])
            # 如果没有抛出异常，验证结果
            assert isinstance(result, list)
        except Exception as e:
            # 如果抛出异常，验证异常类型
            assert isinstance(e, (RuntimeError, ValueError))

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, graphrag_impl, sample_documents):
        """测试并发操作"""
        index_name = "test_index"

        # 并发添加文档
        tasks = []
        for i in range(3):
            docs = [
                Document(
                    uid=f"doc_{i}_{j}", content=f"Content {i}-{j}", summary=f"Summary {i}-{j}", metadata={"batch": i}
                )
                for j in range(2)
            ]
            tasks.append(graphrag_impl.add_documents(f"{index_name}_{i}", docs))

        results = await asyncio.gather(*tasks)

        assert len(results) == 3
        for result in results:
            assert len(result) == 2
