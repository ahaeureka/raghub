from typing import Optional

from langchain_core.caches import RETURN_VAL_TYPE, BaseCache


class ChatCache(BaseCache):
    def __init__(self, cache_size: int = 100):
        pass

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        # Generate a key from the prompt and llm_string
        pass
