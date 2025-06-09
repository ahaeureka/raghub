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
    ]
    docs = await impl.add_documents(unique_name, [Document(content=doc, uid=compute_mdhash_id(unique_name,doc, "doc")) for doc in docs])
    queries = [
        "太和殿在庚子事变中遭受的破坏，与故宫文物南迁有何间接关联？",
        # "How did Cinderella reach her happy ending?",
        # "What county is Erik Hort's birthplace a part of?",
    ]
    retrieve_docs = await impl.retrieve(unique_name, queries=queries)
    for query, doc in retrieve_docs.items():
        print(query, [(item.document.content, item.score) for item in doc])
    # await impl.delete(unique_name, [doc.uid for doc in docs])


if __name__ == "__main__":
    asyncio.run(main())
