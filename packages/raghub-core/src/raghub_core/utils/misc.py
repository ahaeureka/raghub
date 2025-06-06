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


def flatten_dict(d, parent_key="", sep="."):
    """
    将嵌套字典展平为单层字典，键名使用点号分隔路径

    Args:
        d: 要展平的字典
        parent_key: 父级键名
        sep: 分隔符，默认为点号

    Returns:
        展平后的字典
    """
    items = []

    for k, v in d.items():
        # 构建新的键名
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if isinstance(v, dict):
            # 如果值是字典，递归处理
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # 处理列表，为每个元素创建索引路径
            for i, item in enumerate(v):
                list_key = f"{new_key}[{i}]"
                if isinstance(item, dict):
                    items.extend(flatten_dict(item, list_key, sep=sep).items())
                else:
                    items.append((list_key, item))
        else:
            # 普通值直接添加
            items.append((new_key, v))

    return dict(items)


def unflatten_dict(flat_dict, sep="."):
    """
    将展平的字典还原为嵌套结构

    Args:
        flat_dict: 展平的字典
        sep: 分隔符，默认为点号

    Returns:
        还原后的嵌套字典
    """
    result = {}

    for key, value in flat_dict.items():
        # 分割键路径
        parts = key.split(sep)
        current = result

        # 遍历路径的每一部分
        for i, part in enumerate(parts[:-1]):
            # 检查是否是列表索引格式 key[index]
            if "[" in part and "]" in part:
                # 分离键名和索引
                base_key = part.split("[")[0]
                index_str = part.split("[")[1].rstrip("]")
                index = int(index_str)

                # 确保基础键存在且为列表
                if base_key not in current:
                    current[base_key] = []

                # 扩展列表到所需长度
                while len(current[base_key]) <= index:
                    current[base_key].append({})

                current = current[base_key][index]
            else:
                # 普通键处理
                if part not in current:
                    current[part] = {}
                current = current[part]

        # 处理最后一个键
        last_part = parts[-1]
        if "[" in last_part and "]" in last_part:
            base_key = last_part.split("[")[0]
            index_str = last_part.split("[")[1].rstrip("]")
            index = int(index_str)

            if base_key not in current:
                current[base_key] = []

            while len(current[base_key]) <= index:
                current[base_key].append(None)

            current[base_key][index] = value
        else:
            current[last_part] = value

    return result
