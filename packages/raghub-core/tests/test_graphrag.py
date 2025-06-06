import asyncio
import os

from raghub_core.chat.openai_chat import OpenAIProxyChat
from raghub_core.embedding.openai_embedding import OpenAIEmbedding
from raghub_core.rag.graphrag.graph_dao import GraphRAGDAO
from raghub_core.rag.graphrag.graphrag_impl import GraphRAGImpl
from raghub_core.rag.graphrag.operators import DefaultGraphRAGOperators
from raghub_core.schemas.document import Document
from raghub_core.storage.chromadb_vector import ChromaDBVectorStorage
from raghub_core.storage.igraph_store import IGraphStore
from raghub_core.storage.local_sql import LocalSQLStorage
from raghub_core.utils.file.project import ProjectHelper
from raghub_core.utils.misc import compute_mdhash_id


# @pytest.fixture
async def graphRAG():
    llm = OpenAIProxyChat(os.environ["LLM_MODEL_NAME"], os.environ["OPENAI_API_KEY"], os.environ["OPENAI_API_BASE"])
    embbedder = OpenAIEmbedding(
        os.environ["OPENAI_API_KEY"],
        model_name=os.environ["EMBEDDING_MODEL_NAME"],
        base_url=os.environ["OPENAI_API_BASE"],
    )
    vector = ChromaDBVectorStorage(embbedder)
    graph = IGraphStore((ProjectHelper.get_project_root() / "cache/graph").as_posix())
    db_url = (ProjectHelper.get_project_root() / "cache/sqlite.db").as_posix()
    sql = LocalSQLStorage(f"sqlite:///{db_url}")
    await vector.init()
    await graph.init()
    await sql.init()
    storage = GraphRAGDAO(vector, graph, sql)
    return GraphRAGImpl(llm, embbedder, storage, DefaultGraphRAGOperators(llm, vector))


# @pytest.mark.asyncio
async def test_add_documents(graphRAG: GraphRAGImpl):
    text = """
在2023年国际人工智能大会上，来自斯坦福大学的研究团队发布了最新研发的量子计算模型Q-Net。该模型由首席科学家张伟博士带领，团队成员包括李娜和王强。Q-Net在图像识别任务中达到了98.7%的准确率，其核心算法基于图神经网络（GNN）架构。
张伟博士此前曾在谷歌研究院工作三年，2020年加入斯坦福大学人工智能实验室。他的研究方向还包括自然语言处理和强化学习。李娜是团队中唯一拥有医学背景的成员，专注于医疗图像分析。
2024年3月，斯坦福大学与麻省理工学院（MIT）宣布合作开发新一代AI伦理框架。该框架由张伟博士和MIT的艾米丽·陈教授共同主导。项目预算为500万美元，预计持续36个月。
王强在业余时间运营着一个名为'AI Explained'的科普博客，该博客每月访问量超过10万次。最近他发布了一篇关于Q-Net模型的深度解析文章，引发学术界广泛讨论。
李娜的丈夫王磊是硅谷知名投资人，曾向斯坦福大学捐赠200万美元用于建设新实验室。该实验室将于2025年1月正式启用，配备最先进的量子计算设备
"""  # noqa: E501
    docs = [
        Document(
            content=text,
            uid=compute_mdhash_id(text, "doc"),
            metadata={
                "source": "https://example.com/article",
                "author": "张伟",
                "date": "2023-10-01",
                "tags": ["AI", "Quantum Computing", "GNN"],
            },
        )
    ]
    await graphRAG.add_documents("test_graph_rag", docs, "zh")


async def main():
    graph = await graphRAG()
    graph.init()
    await test_add_documents(graph)
    # You can add more tests here, like retrieval, deletion, etc.
    # await graph.retrieve("test_graph_rag", ["What is Q-Net?"])
    # await graph.delete("test_graph_rag", [doc.uid for doc in docs])


if __name__ == "__main__":
    asyncio.run(main())
