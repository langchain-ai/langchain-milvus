"""Sync and async hybrid search must signal deprecated ranker args alike.

The warning fires before any I/O, so no Milvus instance is needed.
"""

from unittest.mock import MagicMock

import pytest
from pymilvus import Function, FunctionType

from langchain_milvus.vectorstores import Milvus


def _store() -> Milvus:
    """A store whose `col` property resolves from cache, so nothing connects."""
    store = Milvus.__new__(Milvus)
    store.collection_name = "c"
    store.alias = "default"
    store._col_cache = MagicMock(name="collection")
    store._cache_key = "c:default"
    return store


def _reranker() -> Function:
    reranker = MagicMock(spec=Function)
    reranker.type = FunctionType.RERANK
    return reranker


def test_sync_warns_when_reranker_and_deprecated_args_are_both_given() -> None:
    """Pins the behaviour the async path is expected to match."""
    with pytest.warns(DeprecationWarning, match="Will use 'reranker' parameter"):
        with pytest.raises(Exception):  # noqa: B017 - stops at the first I/O
            _store()._collection_hybrid_search(
                query="q",
                embeddings=[[0.1]],
                k=4,
                param=None,
                expr=None,
                reranker=_reranker(),
                ranker_type="rrf",
            )


async def test_async_warns_when_reranker_and_deprecated_args_are_both_given() -> None:
    with pytest.warns(DeprecationWarning, match="Will use 'reranker' parameter"):
        with pytest.raises(Exception):  # noqa: B017 - stops at the first I/O
            await _store()._acollection_hybrid_search(
                query="q",
                embeddings=[[0.1]],
                k=4,
                param=None,
                expr=None,
                reranker=_reranker(),
                ranker_params={"k": 60},
            )
