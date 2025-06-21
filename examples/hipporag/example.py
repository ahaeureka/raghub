import asyncio

from raghub_app.apps.hipporag.hipporag_app import HippoRAG
from raghub_app.config.config_models import APPConfig
from raghub_core.config.manager import ConfigLoader
from raghub_core.schemas.document import Document
from raghub_core.utils.misc import compute_mdhash_id


async def main():
    # shutil.rmtree("/app/output/.raghub", ignore_errors=True)
    from raghub_ext import storage_ext  # noqa: F401

    # app_config = ConfigLoader.load(cls=APPConfig, file_path="/app/examples/hipporag/online.toml")
    app_config = ConfigLoader.load(cls=APPConfig, file_path="/app/.devcontainer/online.toml")

    impl = HippoRAG(app_config)
    await impl.init()
    unique_name = "hipporag_example7"
    await impl.create(unique_name)
    docs = [
        "北京故宫始建于明朝永乐四年（1406年），是明清两代的皇家宫殿，旧称紫禁城。其建筑布局严格遵循《周礼·考工记》的“前朝后市，左祖右社”原则，外朝以太和殿、中和殿、保和殿为核心，用于举行重大典礼。",
        "太和殿俗称“金銮殿”，殿顶采用重檐庑殿顶结构，屋脊安放十只脊兽，为中国古代建筑的最高形制。1900年庚子事变中，殿内陈设遭八国联军劫掠，现存龙椅为民国时期复刻品",
        "故宫文物南迁始于1933年，为躲避日军侵华战火，逾1.3万箱珍宝经铁路运抵上海、南京，后辗转存于四川乐山安谷乡。这批文物包含《清明上河图》等传世名作，战后部分运回北京，部分现存台北故宫博物院。",
        """在2023年国际人工智能大会上，来自斯坦福大学的研究团队发布了最新研发的量子计算模型Q-Net。该模型由首席科学家张伟博士带领，团队成员包括李娜和王强。Q-Net在图像识别任务中达到了98.7%的准确率，其核心算法基于图神经网络（GNN）架构。
张伟博士此前曾在谷歌研究院工作三年，2020年加入斯坦福大学人工智能实验室。他的研究方向还包括自然语言处理和强化学习。李娜是团队中唯一拥有医学背景的成员，专注于医疗图像分析。
2024年3月，斯坦福大学与麻省理工学院（MIT）宣布合作开发新一代AI伦理框架。该框架由张伟博士和MIT的艾米丽·陈教授共同主导。项目预算为500万美元，预计持续36个月。
王强在业余时间运营着一个名为'AI Explained'的科普博客，该博客每月访问量超过10万次。最近他发布了一篇关于Q-Net模型的深度解析文章，引发学术界广泛讨论。
李娜的丈夫王磊是硅谷知名投资人，曾向斯坦福大学捐赠200万美元用于建设新实验室。该实验室将于2025年1月正式启用，配备最先进的量子计算设备""",
    ]
    docs = await impl.add_documents(
        unique_name, [Document(content=doc, uid=compute_mdhash_id(unique_name, doc, "doc")) for doc in docs]
    )
    queries = [
        "太和殿在庚子事变中遭受的破坏，与故宫文物南迁有何间接关联？",
        "从紫禁城初建到文物南迁，历经哪些重大历史阶段？列出关键时间节点及事件。",
        # "What county is Erik Hort's birthplace a part of?",
    ]
    for query in queries:
        async for ans in impl.QA(unique_name, query, retrieve_top_k=5, lang="zh"):
            print(ans.answer)
    # retrieve_docs = await impl.retrieve(unique_name, queries=queries)
    # for query, doc in retrieve_docs.items():
    #     print(query, [(item.document.content, item.score) for item in doc])
    # await impl.delete(unique_name, [doc.uid for doc in docs])


if __name__ == "__main__":
    asyncio.run(main())
