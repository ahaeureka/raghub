import os
from typing import List

import tiktoken
from raghub_core.utils.file.project import ProjectHelper

tiktoken_cache_dir = ProjectHelper.get_project_root().joinpath("models", "tiktoken").as_posix()
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir
tiktoken_encoder = tiktoken.get_encoding("cl100k_base")


class EmbeddingHelper:
    @staticmethod
    def truncate(string: str, max_len: int) -> List[str]:
        if tiktoken_encoder is None:
            raise ValueError("Model is not initialized. Please provide a valid model.")
        tokens = tiktoken_encoder.encode(string)
        if len(tokens) <= max_len:
            return [string]
        chunks = []
        for i in range(0, len(tokens), max_len):
            chunk_tokens = tokens[i : i + max_len]
            chunk_str = tiktoken_encoder.decode(chunk_tokens)
            chunks.append(chunk_str)
        return chunks
