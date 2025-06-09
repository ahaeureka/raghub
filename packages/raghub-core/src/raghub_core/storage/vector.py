from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from raghub_core.schemas.document import Document
from raghub_core.utils.class_meta import SingletonRegisterMeta


class VectorStorage(metaclass=SingletonRegisterMeta):
    # name = None
    # _registry: Dict[str, Type["VectorStorage"]] = {}  # 注册表
    @abstractmethod
    async def init(self):
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    async def create_index(self, index_name: str) -> None:
        """
        Create a new index in the vector storage.
        Args:
            index_name: Name of the index to create.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    async def add_documents(self, index_name: str, texts: List[Document]) -> List[Document]:
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    async def get_by_ids(self, index_name: str, ids: List[str]) -> List[Document]:
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    async def get(self, index_name: str, uid: str) -> Document:
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    async def select_on_metadata(self, index_name: str, metadata_filter: Dict[str, Any]) -> List[Document]:
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    async def delete(self, index_name: str, ids: List[str]) -> bool:
        """
        Delete documents by their IDs from the vector storage.
        Args:
            index_name: Name of the index from which to delete documents.
            ids: List of document IDs to delete.

        Returns:
            bool: True if deletion was successful, False otherwise.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    async def similarity_search_by_vector(
        self, index_name: str, embedding: List[float], k: int, filter: Optional[Dict[str, str]] = None
    ) -> List[Tuple[Document, float]]:
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    async def asimilar_search_with_scores(
        self, index_name: str, query: str, k: int, filter: Optional[Dict[str, str]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Perform a similarity search in the vector storage using a query string.
        Args:
            index_name: Name of the index to search in.
            query: Query string to search for.
            k: Number of top results to return.
            filter: Optional filter for the search.
        Returns:
            List[Tuple[Document, float]]: List of tuples containing the matching documents and their similarity scores.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    async def add_documents_filter_exists(self, index_name: str, texts: List[Document]) -> List[Document]:
        """
        Add documents to the vector storage, filtering out those that already exist.
        Args:
            texts: List of Document objects to add.

        Returns:
            List of Document objects that were added.
        """
        # Check if the document already exists in the storage
        existing_docs = await self.get_by_ids(index_name, [doc.uid for doc in texts])
        existing_ids = {doc.uid for doc in existing_docs}

        # Filter out documents that already exist
        new_docs = [doc for doc in texts if doc.uid not in existing_ids]

        # Add new documents to the storage
        if new_docs:
            logger.debug(f"add_documents_filter_exists:Adding {new_docs} new documents to the vector storage.")
            await self.add_documents(index_name, new_docs)

        return new_docs

    def knn(
        self,
        query_docs: List[Document],
        target_docs: List[Document],
        k=2047,
        query_batch_size=1000,
        key_batch_size=10000,
    ) -> Dict[str, Tuple[List[str], List[float]]]:
        """
        Retrieve the top-k nearest neighbors for each query id from the key ids.
        Args:
            query_ids:
            key_ids:
            k: top-k
            query_batch_size:
            key_batch_size:

        Returns:

        """
        import torch
        from tqdm import tqdm

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # np.array(float_list, dtype=np.float32)
        query_vecs = np.array(
            [doc.embedding for doc in query_docs], dtype=np.float32
        )  # Assuming Document has an embedding attribute
        target_vecs = np.array(
            [doc.embedding for doc in target_docs], dtype=np.float32
        )  # Assuming Document has an embedding attribute
        query_ids = [doc.uid for doc in query_docs]
        target_ids = [doc.uid for doc in target_docs]
        query_tensor_vecs = torch.tensor(query_vecs, dtype=torch.float32)
        query_tensor_vecs = torch.nn.functional.normalize(query_tensor_vecs, dim=1)

        key_vecs = torch.tensor(target_vecs, dtype=torch.float32)
        key_vecs = torch.nn.functional.normalize(key_vecs, dim=1)

        results = {}
        key_ids = target_ids

        def get_batches(vecs, batch_size):
            for i in range(0, len(vecs), batch_size):
                yield vecs[i : i + batch_size], i

        for query_batch, query_batch_start_idx in tqdm(
            get_batches(vecs=query_tensor_vecs, batch_size=query_batch_size),
            total=(len(query_vecs) + query_batch_size - 1) // query_batch_size,  # Calculate total batches
            desc="KNN for Queries",
        ):
            query_batch = query_batch.clone().detach()
            query_batch = query_batch.to(device)

            batch_topk_sim_scores = []
            batch_topk_indices = []

            offset_keys = 0

            for key_batch, key_batch_start_idx in get_batches(vecs=key_vecs, batch_size=key_batch_size):
                key_batch = key_batch.to(device)
                actual_key_batch_size = key_batch.size(0)

                similarity = torch.mm(query_batch, key_batch.T)

                topk_sim_scores, topk_indices = torch.topk(
                    similarity, min(k, actual_key_batch_size), dim=1, largest=True, sorted=True
                )

                topk_indices += offset_keys

                batch_topk_sim_scores.append(topk_sim_scores)
                batch_topk_indices.append(topk_indices)

                del similarity
                key_batch = key_batch.cpu()
                torch.cuda.empty_cache()

                offset_keys += actual_key_batch_size
            # end for each kb batch

            concatenated_scores = torch.cat(batch_topk_sim_scores, dim=1)
            concatenated_topk_indices = torch.cat(batch_topk_indices, dim=1)

            final_topk_sim_scores, final_topk_indices = torch.topk(
                concatenated_scores, min(k, concatenated_scores.size(1)), dim=1, largest=True, sorted=True
            )
            final_topk_indices = final_topk_indices.cpu()
            final_topk_sim_scores = final_topk_sim_scores.cpu()

            for i in range(final_topk_indices.size(0)):
                query_relative_idx = query_batch_start_idx + i
                query_idx = query_ids[query_relative_idx]

                final_topk_indices_i = final_topk_indices[i]
                final_topk_sim_scores_i = final_topk_sim_scores[i]

                query_to_topk_key_relative_ids = concatenated_topk_indices[i][final_topk_indices_i]
                query_to_topk_key_ids = [key_ids[idx] for idx in query_to_topk_key_relative_ids.cpu().numpy()]
                results[query_idx] = (query_to_topk_key_ids, final_topk_sim_scores_i.numpy().tolist())

            query_batch = query_batch.cpu()
            torch.cuda.empty_cache()
        # end for each query batch

        return results
