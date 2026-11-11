"""`add_texts` and `aadd_texts` must validate ids identically.

The checks run before any I/O, so the store is built with `__new__` and only
the attribute the guard reads is set.
"""

import pytest

from langchain_milvus.vectorstores import Milvus


def _store() -> Milvus:
    store = Milvus.__new__(Milvus)
    store.auto_id = False
    return store


TEXTS = ["a", "b"]


@pytest.mark.asyncio
async def test_aadd_texts_rejects_non_string_ids() -> None:
    with pytest.raises(AssertionError, match="All ids should be strings"):
        await _store().aadd_texts(TEXTS, ids=[1, 2])  # type: ignore[list-item]


@pytest.mark.asyncio
async def test_aadd_texts_rejects_oversized_ids() -> None:
    with pytest.raises(AssertionError, match="less than 65535 bytes"):
        await _store().aadd_texts(TEXTS, ids=["ok", "x" * 65_536])


@pytest.mark.asyncio
async def test_aadd_texts_rejects_duplicate_ids() -> None:
    with pytest.raises(AssertionError, match="unique ids"):
        await _store().aadd_texts(TEXTS, ids=["same", "same"])


def test_add_texts_rejects_non_string_ids() -> None:
    """The sync path already did this; pinned so the two stay in step."""
    with pytest.raises(AssertionError, match="All ids should be strings"):
        _store().add_texts(TEXTS, ids=[1, 2])  # type: ignore[list-item]


def test_add_texts_rejects_oversized_ids() -> None:
    with pytest.raises(AssertionError, match="less than 65535 bytes"):
        _store().add_texts(TEXTS, ids=["ok", "x" * 65_536])
