import shutil
from deeprag_app.apps.hipporag.hipporag_impl import HippoRAGImpl
from deeprag_app.config.config_models import APPConfig
from deeprag_core.config.manager import ConfigLoader
from deeprag_core.schemas.document import Document
from deeprag_core.utils.misc import compute_mdhash_id


def main():
    # shutil.rmtree("/app/output/.deeprag", ignore_errors=True)
    
    app_config = ConfigLoader.load(
        cls=APPConfig,
        file_path=ConfigLoader.default_config_paths()
    )
    impl = HippoRAGImpl(app_config)
    impl.init()
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
    # impl.add_documents([Document(content=doc, uid=compute_mdhash_id(doc, "doc")) for doc in docs])
    queries = [
        "What is George Rankin's occupation?",
        # "How did Cinderella reach her happy ending?",
        # "What county is Erik Hort's birthplace a part of?",
    ]
    docs = impl.retrieve(queries=queries)
    for doc in docs:
        print(doc.query,doc.document.content, doc.score)
if __name__ == "__main__":
    main()

