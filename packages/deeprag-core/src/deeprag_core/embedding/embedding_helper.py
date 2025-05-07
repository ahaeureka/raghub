import os

import tiktoken
from deeprag_core.utils.file.project import ProjectHelper
from loguru import logger

tiktoken_cache_dir = ProjectHelper.get_project_root().joinpath("models", "tiktoken").as_posix()
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir
tiktoken_encoder = tiktoken.get_encoding("cl100k_base")


class EmbeddingHelper:
    @staticmethod
    def truncate(string: str, max_len: int) -> str:
        """
        Returns truncated text if the length of text exceeds max_len.
        """
        if tiktoken_encoder is None:
            raise ValueError("Model is not initialized. Please provide a valid model.")
        logger.debug(f"Truncating string to max length:{string} {max_len}")
        return tiktoken_encoder.decode(tiktoken_encoder.encode(string)[:max_len])
