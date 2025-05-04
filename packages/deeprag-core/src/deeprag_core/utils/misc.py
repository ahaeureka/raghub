from hashlib import md5

import numpy as np


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
    return f"{prefix}-{md5(content.encode()).hexdigest()}"


def min_max_normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))
