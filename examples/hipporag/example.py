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
    app_config = ConfigLoader.load(cls=APPConfig, file_path="/app/.devcontainer/dev.toml")

    impl = HippoRAG(app_config)
    await impl.init()
    unique_name = "example3"
    await impl.create(unique_name)
    docs = [
        "Oliver Badman is a politician.",
        "George Rankin is a politician.",
        "Thomas Marwick is a politician.",
        "Cinderella attended the royal ball.",
        "The prince used the lost glass slipper to search the kingdom.",
        "When the slipper fit perfectly, Cinderella was reunited with the prince.",
        "Erik Hort's birthplace is Montebello.",
        "Marina is bom in Minsk.",
        "Montebello is a part of Rockland County.",
    ]
    docs = await impl.add_documents(unique_name, [Document(content=doc, uid=compute_mdhash_id(unique_name,doc, "doc")) for doc in docs])
    queries = [
        "What is George Rankin's occupation?",
        # "How did Cinderella reach her happy ending?",
        # "What county is Erik Hort's birthplace a part of?",
    ]
    retrieve_docs = await impl.retrieve(unique_name, queries=queries)
    for doc in retrieve_docs:
        print(doc.query, doc.document.content, doc.score)
    await impl.delete(unique_name, [doc.uid for doc in docs])


if __name__ == "__main__":
    asyncio.run(main())
