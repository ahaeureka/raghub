from hashlib import md5
from typing import List, Type

import numpy as np
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
