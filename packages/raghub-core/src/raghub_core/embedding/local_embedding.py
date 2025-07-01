from abc import abstractmethod
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from langchain_core.runnables.config import run_in_executor
from raghub_core.embedding.base_embedding import BaseEmbedding


class LocalEmbedding(BaseEmbedding):
    def __init__(self, model_name_or_path: str, batch_size: int = 32, n_dims: Optional[int] = None):
        super().__init__()
        self._model_name_or_path = model_name_or_path
        self._batch_size = batch_size
        self._model = self._load_model()
        self._n_dims = n_dims

    @abstractmethod
    def _load_model(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

    @property
    def embedding_dim(self):
        return self._n_dims


class SentenceTransformersEmbedding(LocalEmbedding):
    name = "sentence_transformers_embbeding"

    # def __init__(self, models_dir: str, model_name: str, batch_size: int = 32):
    #     super().__init__()
    #     from sentence_transformers import SentenceTransformer
    #     self._model = SentenceTransformer(os.path.join(models_dir, model_name))
    #     self._batch_size = batch_size
    def _load_model(self):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self._model_name_or_path)
        return self._model

    def encode(self, texts: List[str], instruction: Optional[str] = None) -> np.ndarray:
        normalize_embeddings = True if self._n_dims is None else False
        embeddings = self._model.encode(
            texts, batch_size=self._batch_size, normalize_embeddings=normalize_embeddings, prompt=instruction
        )
        if normalize_embeddings:
            return embeddings
        from sklearn.preprocessing import normalize

        return normalize(embeddings[:, : self._n_dims])

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode([query])[0]

    async def aencode(self, texts: List[str], instruction: Optional[str] = None) -> np.ndarray:
        await run_in_executor(None, self.encode, texts, instruction)

    async def aencode_query(self, text: str, instruction: Optional[str] = None) -> np.ndarray:
        results = await run_in_executor(None, self.encode_query, text, instruction)
        return np.array(results.tolist()[0])


class TransformersEmbedding(LocalEmbedding):
    name = "transformers-embedding"

    def _load_model(self):
        from transformers import AutoModel, AutoTokenizer

        self._model = AutoModel.from_pretrained(self._model_name_or_path)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name_or_path)
        self._task = "Given a web search query, retrieve relevant passages that answer the query"

        return self._model

    def last_token_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def encode(self, texts: List[str], instruction: Optional[str] = None) -> np.ndarray:
        inputs = self._tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self._model.device)

        with torch.no_grad():
            outputs = self._model(**inputs, output_hidden_states=True)
            embeddings = self.last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
            return F.normalize(embeddings, p=2, dim=1).cpu().numpy()

    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        return f"Instruct: {task_description}\nQuery:{query}"

    def encode_query(self, text: str, instruction: Optional[str] = None) -> np.ndarray:
        return self.encode([text])[0]

    async def aencode(self, texts: List[str], instruction: Optional[str] = None) -> np.ndarray:
        await run_in_executor(None, self.encode, texts, instruction)

    async def aencode_query(self, text: str, instruction: Optional[str] = None) -> np.ndarray:
        results = await run_in_executor(None, self.encode_query, text, instruction)
        return np.array(results.tolist()[0])


class BGEEmbedding(LocalEmbedding):
    name = "beg-embedding"

    def _load_model(self):
        from FlagEmbedding import FlagModel

        self._model = FlagModel(
            self._model_name_or_path,
            query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
            use_fp16=torch.cuda.is_available(),
        )
        return self._model

    def encode(self, texts: List[str], instruction: Optional[str] = None) -> np.ndarray:
        ress = []
        for i in range(0, len(texts), self._batch_size):
            ress.extend(self._model.encode(texts[i : i + self._batch_size], instruction=instruction).tolist())
        return np.array(ress)

    def encode_query(self, text: str, instruction: str = "为这个句子生成表示以用于检索相关文章：") -> np.ndarray:
        results = self._model.encode_queries([text], instruction=instruction)
        return np.array(results.tolist()[0])

    async def aencode(self, texts: List[str], instruction: Optional[str] = None) -> np.ndarray:
        ress = []
        for i in range(0, len(texts), self._batch_size):
            ress.extend(
                await run_in_executor(
                    None, self._model.encode, texts[i : i + self._batch_size], instruction=instruction
                ).tolist()
            )
        return np.array(ress)

    async def aencode_query(
        self, texts: List[str], instruction: str = "为这个句子生成表示以用于检索相关文章："
    ) -> np.ndarray:
        ress = []
        for i in range(0, len(texts), self._batch_size):
            ress.extend(
                await run_in_executor(
                    None, self._model.encode_queries, texts[i : i + self._batch_size], instruction=instruction
                ).tolist()
            )
        return np.array(ress)


if __name__ == "__main__":
    # Example usage
    embedding_model = SentenceTransformersEmbedding(model_name_or_path="Qwen/Qwen3-Embedding-0.6B")
    texts = ["Hello world", "This is a test"]
    embeddings = embedding_model.encode(texts)
    print(embeddings.tolist())
    query_embedding = embedding_model.encode_query("Hello world")
    print(query_embedding.tolist())

    # Example usage for TransformersEmbedding
    transformers_embedding_model = TransformersEmbedding(model_name_or_path="Qwen/Qwen3-Embedding-0.6B")
    texts = ["Hello world", "This is a test"]
    embeddings = transformers_embedding_model.encode(texts)
    print(embeddings.tolist())
    query_embedding = transformers_embedding_model.encode_query("Hello world")
    print(query_embedding.tolist())
