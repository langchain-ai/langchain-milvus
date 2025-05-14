import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from langchain_core.runnables.config import run_in_executor


class BaseSparseEmbedding(ABC):
    """Interface for Sparse embedding models.

    You can inherit from it and implement your custom sparse embedding model.

    By default, the asynchronous methods are implemented using the synchronous methods;
    however, implementations may choose to override the asynchronous methods with
    an async native implementation for performance reasons.
    """

    @abstractmethod
    def embed_query(self, query: str) -> Dict[int, float]:
        """Embed query text."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[Dict[int, float]]:
        """Embed search docs."""

    async def aembed_query(self, query: str) -> Dict[int, float]:
        """Asynchronous Embed query text.

        Args:
            query: Text to embed.

        Returns:
            Embedding.
        """
        return await run_in_executor(None, self.embed_query, query)

    async def aembed_documents(self, texts: list[str]) -> List[Dict[int, float]]:
        """Asynchronous Embed search docs.

        Args:
            texts: List of text to embed.

        Returns:
            List of embeddings.
        """
        return await run_in_executor(None, self.embed_documents, texts)


class BM25SparseEmbedding(BaseSparseEmbedding):
    """Sparse embedding model based on BM25.

    **Note: We recommend using the Milvus built-in BM25 function to implement sparse
    embedding in your application.
    This class is more of a reference because it requires the user to manage the corpus,
     which is not practical. The Milvus built-in function solves this problem and makes
     the BM25 sparse process easier and less frustrating for users.
    For more information, please refer to:
    https://milvus.io/docs/full-text-search.md#Full-Text-Search
    and
    https://github.com/milvus-io/bootcamp/blob/master/bootcamp/tutorials/integration/langchain/full_text_search_with_langchain.ipynb
    **

    This class uses the BM25 model in Milvus model to implement sparse vector embedding.
    This model requires pymilvus[model] to be installed.
    `pip install pymilvus[model]`
    For more information please refer to:
    https://milvus.io/docs/embed-with-bm25.md
    """

    def __init__(self, corpus: List[str], language: str = "en"):
        warnings.warn(
            "BM25SparseEmbedding class will be deprecated in the future. "
            "We recommend using the Milvus built-in BM25 function instead, "
            "which is easier to use "
            "and doesn't require manual corpus management. "
            "For more information, please refer to: "
            "https://milvus.io/docs/full-text-search.md#Full-Text-Search",
            DeprecationWarning,
            stacklevel=2,
        )
        from pymilvus.model.sparse import BM25EmbeddingFunction  # type: ignore
        from pymilvus.model.sparse.bm25.tokenizers import (  # type: ignore
            build_default_analyzer,
        )

        self.analyzer = build_default_analyzer(language=language)
        self.bm25_ef = BM25EmbeddingFunction(self.analyzer, num_workers=1)
        self.bm25_ef.fit(corpus)

    def embed_query(self, text: str) -> Dict[int, float]:
        return self._sparse_to_dict(self.bm25_ef.encode_queries([text]))

    def embed_documents(self, texts: List[str]) -> List[Dict[int, float]]:
        sparse_arrays = self.bm25_ef.encode_documents(texts)
        return [self._sparse_to_dict(sparse_array) for sparse_array in sparse_arrays]

    def _sparse_to_dict(self, sparse_array: Any) -> Dict[int, float]:
        if sparse_array.ndim == 1:  # for scipy>=1.15.0 , the ndim is 1
            # `i` is a tuple with one element
            return {i[0]: sparse_array[i] for i in zip(*sparse_array.nonzero())}
        else:  # for scipy<1.15.0, the ndim is 2
            return {j: sparse_array[i, j] for i, j in zip(*sparse_array.nonzero())}
