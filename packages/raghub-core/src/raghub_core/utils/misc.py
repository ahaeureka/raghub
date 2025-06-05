from hashlib import md5
from typing import List, Type

import numpy as np
from raghub_core.schemas.document import Document
from sqlmodel import SQLModel


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """
    Compute the MD5 hash of the given content string and optionally prepend a prefix.

    Args:
        content (str): The input string to be hashed.
        prefix (str, optional): A string to prepend to the resulting hash. Defaults to an empty string.

    Returns:
        str: A string consisting of the prefix followed by the hexadecimal representation of the MD5 hash.
    """
    if prefix.endswith("-"):
        prefix = prefix.removesuffix("-")  # Remove trailing hyphen if present
    if not prefix:
        return md5(content.encode()).hexdigest()
    return f"{prefix}-{md5(content.encode()).hexdigest()}"


def min_max_normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def get_primary_key_names(model_class: Type[SQLModel]) -> List[str]:
    """
    Get the primary key names of a SQLModel class.
    Args:
        model_class (Type[SQLModel]): The SQLModel class to inspect.
    Returns:
        List[str]: A list of primary key names.
    """
    primary_key_names = [col.name for col in model_class.__table__.primary_key.columns.values()]
    return primary_key_names
    # mapper = inspect(model_class.__t)
    # if hasattr(mapper, "primary_key"):
    #     # If the model has a primary key, return its names
    #     return [col.name for col in mapper.primary_key]
    # return []


def detect_language(text) -> str:
    import langid

    # langid.classify 返回一个元组 (语言代码, 置信度)
    lang, _ = langid.classify(text)
    return lang


def docs_duplicate_filter(docs: List[Document]) -> List[Document]:
    """
    Remove duplicates from a list while preserving the order.

    Args:
        docs (List[Document]): The input list of Document objects.

    Returns:
        List[Document]: A new list with duplicates removed.
    """
    seen_uids = set()
    unique_docs = []

    for doc in docs:
        if doc.uid not in seen_uids:
            seen_uids.add(doc.uid)
            unique_docs.append(doc)

    return unique_docs
