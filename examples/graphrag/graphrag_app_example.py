import asyncio

from raghub_app.apps.graphrag.graphrag_app import GraphRAG
from raghub_app.config.config_models import APPConfig
from raghub_core.config.manager import ConfigLoader
from raghub_core.schemas.document import Document
from raghub_core.utils.misc import compute_mdhash_id


async def main():
    # shutil.rmtree("/app/output/.raghub", ignore_errors=True)
    from raghub_core.rag.graphrag.graph_dao import GraphRAGDAO  # noqa: F401
    from raghub_ext import storage_ext  # noqa: F401

    # app_config = ConfigLoader.load(cls=APPConfig, file_path="/app/examples/hipporag/online.toml")
    app_config = ConfigLoader.load(cls=APPConfig, file_path="/app/.devcontainer/online.toml")

    impl = GraphRAG(app_config)
    await impl.init()
    unique_name = "example_graphrag_app_4"
    await impl.create_new_index(unique_name)
    docs = [
        """
在2023年国际人工智能大会上，来自斯坦福大学的研究团队发布了最新研发的量子计算模型Q-Net。该模型由首席科学家张伟博士带领，团队成员包括李娜和王强。Q-Net在图像识别任务中达到了98.7%的准确率，其核心算法基于图神经网络（GNN）架构。
张伟博士此前曾在谷歌研究院工作三年，2020年加入斯坦福大学人工智能实验室。他的研究方向还包括自然语言处理和强化学习。李娜是团队中唯一拥有医学背景的成员，专注于医疗图像分析。
2024年3月，斯坦福大学与麻省理工学院（MIT）宣布合作开发新一代AI伦理框架。该框架由张伟博士和MIT的艾米丽·陈教授共同主导。项目预算为500万美元，预计持续36个月。
王强在业余时间运营着一个名为'AI Explained'的科普博客，该博客每月访问量超过10万次。最近他发布了一篇关于Q-Net模型的深度解析文章，引发学术界广泛讨论。
李娜的丈夫王磊是硅谷知名投资人，曾向斯坦福大学捐赠200万美元用于建设新实验室。该实验室将于2025年1月正式启用，配备最先进的量子计算设备
"""  # noqa: E501
    ]
    # docs = await impl.add_documents(
    #     unique_name, [Document(content=doc, uid=compute_mdhash_id(unique_name, doc, "doc")) for doc in docs], "zh"
    # )
    queries = [
        "请详细说明张伟博士在2020年加入斯坦福大学后参与的主要研究项目?",
        # "How did Cinderella reach her happy ending?",
        "What county is Erik Hort's birthplace a part of?",
    ]
    async for ans in impl.QA(unique_name, queries[0], retrieve_top_k=5, lang="zh"):
        print(ans.answer)
    # await impl.delete(unique_name, [doc.uid for doc in docs])


if __name__ == "__main__":
    asyncio.run(main())
