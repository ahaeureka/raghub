import asyncio
import os

from raghub_core.chat.openai_chat import OpenAIProxyChat
from raghub_core.embedding.openai_embedding import OpenAIEmbedding
from raghub_core.rag.graphrag.graph_dao import GraphRAGDAO
from raghub_core.rag.graphrag.graphrag_impl import GraphRAGImpl
from raghub_core.rag.graphrag.operators import DefaultGraphRAGOperators
from raghub_core.rag.graphrag.prompts import GraphRAGContextPrompt
from raghub_core.schemas.document import Document
from raghub_core.storage.chromadb_vector import ChromaDBVectorStorage
from raghub_core.storage.igraph_store import IGraphStore
from raghub_core.storage.local_sql import LocalSQLStorage
from raghub_core.utils.file.project import ProjectHelper
from raghub_core.utils.misc import compute_mdhash_id


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
    return GraphRAGImpl(llm, storage, DefaultGraphRAGOperators(llm, vector))


async def test_add_documents(graphRAG: GraphRAGImpl):
    text = """
在2023年国际人工智能大会上，来自斯坦福大学的研究团队发布了最新研发的量子计算模型Q-Net。该模型由首席科学家张伟博士带领，团队成员包括李娜和王强。Q-Net在图像识别任务中达到了98.7%的准确率，其核心算法基于图神经网络（GNN）架构。
张伟博士此前曾在谷歌研究院工作三年，2020年加入斯坦福大学人工智能实验室。他的研究方向还包括自然语言处理和强化学习。李娜是团队中唯一拥有医学背景的成员，专注于医疗图像分析。
2024年3月，斯坦福大学与麻省理工学院（MIT）宣布合作开发新一代AI伦理框架。该框架由张伟博士和MIT的艾米丽·陈教授共同主导。项目预算为500万美元，预计持续36个月。
王强在业余时间运营着一个名为'AI Explained'的科普博客，该博客每月访问量超过10万次。最近他发布了一篇关于Q-Net模型的深度解析文章，引发学术界广泛讨论。
李娜的丈夫王磊是硅谷知名投资人，曾向斯坦福大学捐赠200万美元用于建设新实验室。该实验室将于2025年1月正式启用，配备最先进的量子计算设备
"""
    text2 = """
星尘科技（Stardust Tech），一家总部位于深圳的人工智能初创公司，由李明和王华于2023年初共同创立。
该公司专注于开发用于数据中心优化的智能算法“Nebula Optimizer”。
凭借其创新技术，星尘科技在2023年第三季度获得了来自深港创投（Shenzhen-HK Ventures）和云海资本（CloudSea Capital）的 5000 万美元 A 轮融资。
与此同时，另一家人工智能公司幻影智能（Phantom Intelligence），位于上海，由张伟在 2022 年创立，也在快速发展。
幻影智能的核心产品是“Spectra Vision”，一个计算机视觉平台，主要应用于工业质检。两家公司在智能工业解决方案领域存在竞争关系。
2024 年第一季度，市场传出星尘科技有意收购幻影智能的消息，
旨在整合双方技术优势（“Nebula Optimizer”的算法优化和“Spectra Vision”的视觉能力）以打造更全面的工业 AI 解决方案平台。经过数月的谈判，星尘科技于 2024 年 6 月 15 日正式宣布以 8.2 亿美元全股票交易完成对幻影智能的收购。张伟加入星尘科技，担任首席技术官（CTO）。
然而，技术整合的复杂性和 2024 年下半年全球芯片供应链的波动，
导致原计划在 2024 年底推出的整合平台“Stellar Platform”被推迟。最终，“Stellar Platform”在 2025 年 3 月才成功发布，并迅速获得了市场认可。
"""
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
        ),
        Document(content=text2, uid=compute_mdhash_id(text2, "doc"), metadata={}),
    ]
    await graphRAG.add_documents("test_graph_rag", docs, "zh")


async def main():
    graph = await graphRAG()
    graph.init()
    await test_add_documents(graph)
    await graph.delete("test_graph_rag", ["doc-8476c90becef26e95227f23d8a710186"])
    result = await graph.retrieve(
        "test_graph_rag",
        [
            "请详细说明张伟博士在2020年加入斯坦福大学后参与的主要研究项目",
            "星尘科技在收购幻影智能后，计划推出的整合平台是什么?",
        ],
    )
    for key, value in result.items():
        context = (
            GraphRAGContextPrompt()
            .get("zh")
            .format_messages(
                context=value.context,
                query=value.query,
                knowledge_graph=value.subgraph,
                knowledge_graph_for_doc="\n".join([doc.content for doc in value.docs]),
            )[0]
            .content
        )
        from raghub_core.embedding.embedding_helper import tiktoken_encoder

        print(f"Query {value.query} for {context}")
        print(f"used {len(tiktoken_encoder.encode(context))} tokens")
    # You can add more tests here, like retrieval, deletion, etc.
    # await graph.retrieve("test_graph_rag", ["What is Q-Net?"])
    # await graph.delete("test_graph_rag", [doc.uid for doc in docs])


if __name__ == "__main__":
    asyncio.run(main())
