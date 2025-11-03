from typing import List

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

fake_texts = ["foo", "bar", "baz"]


class FakeEmbeddings(Embeddings):
    """Fake embeddings functionality for testing."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return simple embeddings.
        Embeddings encode each text as its index."""
        return [[float(1.0)] * 9 + [float(i)] for i in range(len(texts))]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """Return constant query embeddings.
        Embeddings are identical to embed_documents(texts)[0].
        Distance to each text will be that text's index,
        as it was passed to embed_documents."""
        return [float(1.0)] * 9 + [float(0.0)]

    async def aembed_query(self, text: str) -> List[float]:
        return self.embed_query(text)


class FakeFp16Embeddings(Embeddings):
    """Fake fp16 precision embeddings functionality for testing."""

    def embed_documents(self, texts: List[str]) -> List:  # type: ignore[no-untyped-def]
        """Return simple embeddings with fp16 precision.
        Embeddings encode each text as its index."""
        fp16_vectors = []
        for i in range(len(texts)):
            raw_vector = [(1 / 9) * d for d in range(9)] + [float(i)]
            fp16_vector = np.array(raw_vector, dtype=np.float16)
            fp16_vectors.append(fp16_vector)
        return fp16_vectors

    async def aembed_documents(self, texts: List[str]) -> List:  # type: ignore[no-untyped-def]
        return self.embed_documents(texts)

    def embed_query(self, text: str):  # type: ignore[no-untyped-def]
        """Return constant query embeddings.
        Embeddings are identical to embed_documents(texts)[0].
        Distance to each text will be that text's index,
        as it was passed to embed_documents."""
        return np.array([(1 / 9) * d for d in range(9)] + [float(0.0)], dtype=np.float16)

    async def aembed_query(self, text: str):  # type: ignore[no-untyped-def]
        return self.embed_query(text)


class FixedValuesEmbeddings(Embeddings):
    """Fake embeddings class with fixed values for document and query embeddings"""

    def __init__(self, *, documents_base_val: float = 1.0, query_val: float = 1.0):
        self.documents_val = documents_base_val
        self.query_val = query_val

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[self.documents_val + i] * 10 for i in range(len(texts))]

    def embed_query(self, text: str) -> List[float]:
        return [float(self.query_val)] * 10


class DirectionEmbeddings(Embeddings):
    """Fake embeddings class with 2 dimention orthogonal basis vectors for testing."""

    def _get_embedding(self, text: str) -> list[float]:
        if text.lower() == "left":
            return [-1, 0]
        elif text.lower() == "right":
            return [1, 0]
        elif text.lower() == "up":
            return [0, 1]
        else:
            return [0, -1]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._get_embedding(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._get_embedding(text)


def assert_docs_equal_without_pk(docs1: List[Document], docs2: List[Document], pk_field: str = "pk") -> None:
    """Assert two lists of Documents are equal, ignoring the primary key field."""
    assert len(docs1) == len(docs2)
    for doc1, doc2 in zip(docs1, docs2):
        assert doc1.page_content == doc2.page_content
        doc1.metadata.pop(pk_field, None)
        doc2.metadata.pop(pk_field, None)
        assert doc1.metadata == doc2.metadata
