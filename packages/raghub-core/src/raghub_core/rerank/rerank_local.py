from typing import Dict, List

import torch
from langchain_core.runnables.config import run_in_executor
from raghub_core.rerank.base_rerank import BaseRerank
from raghub_core.schemas.document import Document


class LocalRerankerByTransformers(BaseRerank):
    """
    A class to handle local reranking of documents based on a given query.
    """

    name = "transformers-reranker"

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-0.6B",
        max_length: int = 4096,
        prefix: str = "",
        suffix: str = "",
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self._max_length = max_length
        prefix = (
            prefix
            or "<|im_start|>system\nJudge whether the Document meets the requirements based on"
            ' the Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
            "<|im_end|>\n<|im_start|>user\n"
        )
        suffix = suffix or "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self._prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self._suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)
        self._task = "Given a web search query, retrieve relevant passages that answer the query"
        self._model = AutoModelForCausalLM.from_pretrained(model_name).eval()
        self._token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self._token_true_id = self.tokenizer.convert_tokens_to_ids("yes")

    def format_instructionse(self, instruction, query, doc):
        if instruction is None:
            instruction = "Given a web search query, retrieve relevant passages that answer the query"
        output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
            instruction=instruction, query=query, doc=doc
        )
        return output

    def process_inputs(self, pairs):
        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=self._max_length - len(self._prefix_tokens) - len(self._suffix_tokens),
        )
        for i, ele in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = self._prefix_tokens + ele + self._suffix_tokens
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self._max_length)
        for key in inputs:
            inputs[key] = inputs[key].to(self._model.device)
        return inputs

    @torch.no_grad()
    def compute_logits(self, inputs, **kwargs):
        batch_scores = self._model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self._token_true_id]
        false_vector = batch_scores[:, self._token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores

    async def rerank(self, query, documents: List[Document]) -> Dict[str, float]:
        """
        Rerank the provided documents based on the query.

        Args:
            query (str): The search query to rerank documents against.
            documents (list): List of documents to be reranked.

        Returns:
            dict: A dictionary with document IDs as keys and their reranked scores as values.
        """
        # Placeholder for actual reranking logic
        return await run_in_executor(None, self.sync_rerank, query, documents)

    def sync_rerank(self, query, documents: List[Document]) -> Dict[str, float]:
        """
        Synchronous reranking of documents based on the query.

        Args:
            query (str): The search query.
            documents (list): List of documents to rerank.

        Returns:
            dict: Reranked documents with their scores.
        """
        pairs = [self.format_instructionse(self._task, query, doc.content) for doc in documents]
        inputs = self.process_inputs(pairs)
        scores = self.compute_logits(inputs)
        reranked_docs = {}
        for i, doc in enumerate(documents):
            reranked_docs[doc.uid] = scores[i]
        reranked_docs = dict(sorted(reranked_docs.items(), key=lambda item: item[1], reverse=True))
        return reranked_docs


if __name__ == "__main__":
    # Example usage
    reranker = LocalRerankerByTransformers()
    query = "What is the capital of China?"
    documents = [
        Document(uid="1", content="The capital of China is Beijing."),
        Document(
            uid="2",
            content="Gravity is a force that attracts two bodies towards each other. "
            "It gives weight to physical objects and is responsible for the movement of planets around the sun.",
        ),
    ]
    reranked_docs = reranker.sync_rerank(query, documents)
    print(reranked_docs)
    # Output: {'1': score1, '2': score2} where score1
