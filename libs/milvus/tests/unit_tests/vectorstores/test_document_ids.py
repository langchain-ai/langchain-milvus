"""Regression tests for primary keys on returned documents (issue #96)."""

from copy import deepcopy
from typing import Any, Optional
from unittest.mock import AsyncMock, Mock

import pytest

from langchain_milvus import Milvus


def _store(primary_field: str, metadata_field: Optional[str] = None) -> Milvus:
    # Result conversion should be testable without opening a database connection.
    store = Milvus.__new__(Milvus)
    store._primary_field = primary_field
    store._text_field = "text"
    store._vector_field = "vector"
    store._metadata_field = metadata_field
    store.collection_name = "documents"
    store.timeout = None
    store.alias = "test"
    store._cache_key = "documents:test"
    store._col_cache = Mock()
    return store


@pytest.mark.parametrize("primary_key", [0, 42, "document-1"])
@pytest.mark.parametrize("primary_field", ["pk", "custom_id"])
@pytest.mark.parametrize("metadata_field", [None, "metadata"])
def test_parse_document_id(
    primary_key: Any, primary_field: str, metadata_field: Optional[str]
) -> None:
    store = _store(primary_field, metadata_field)
    metadata = {"source": "test", "id": "metadata-id"}
    data = {primary_field: primary_key, "text": "hello", "vector": [1.0, 0.0]}
    if metadata_field:
        data[metadata_field] = metadata
        expected_metadata = metadata
    else:
        data.update(metadata)
        expected_metadata = {primary_field: primary_key, **metadata}

    document = store._parse_document(data)

    assert document.id == str(primary_key)
    assert document.page_content == "hello"
    assert document.metadata == expected_metadata


@pytest.mark.parametrize("include_null_key", [False, True])
def test_parse_document_without_id(include_null_key: bool) -> None:
    store = _store("pk")
    data: dict = {"text": "hello", "source": "test"}
    if include_null_key:
        data["pk"] = None

    assert store._parse_document(data).id is None


@pytest.mark.parametrize("operation", ["similarity", "hybrid", "mmr", "metadata"])
@pytest.mark.parametrize("use_async", [False, True])
async def test_search_document_ids(
    operation: str, use_async: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store("custom_id", "metadata")
    entity = {"custom_id": 0, "text": "hello", "metadata": {"source": "test"}}
    results = [[{"custom_id": 0, "id": 0, "distance": 0.25, "entity": entity}]]
    store._milvus_client = Mock()
    store._async_milvus_client = AsyncMock()
    monkeypatch.setattr(store, "_collection_search", Mock(return_value=results))
    monkeypatch.setattr(store, "_acollection_search", AsyncMock(return_value=results))
    monkeypatch.setattr(store, "_collection_hybrid_search", Mock(return_value=results))
    monkeypatch.setattr(
        store, "_acollection_hybrid_search", AsyncMock(return_value=results)
    )
    store.client.query.return_value = [{"custom_id": 0, "vector": [1.0, 0.0]}]  # type: ignore[attr-defined]
    store.aclient.query.return_value = [{"custom_id": 0, "vector": [1.0, 0.0]}]  # type: ignore[attr-defined]

    if operation == "similarity":
        if use_async:
            documents = await store.asimilarity_search_by_vector([1.0, 0.0])
        else:
            documents = store.similarity_search_by_vector([1.0, 0.0])
    elif operation == "hybrid":
        store._vector_field = ["vector", "sparse"]
        if use_async:
            documents = await store.asimilarity_search("hello")
        else:
            documents = store.similarity_search("hello")
    elif operation == "mmr":
        if use_async:
            documents = await store.amax_marginal_relevance_search_by_vector(
                [1.0, 0.0], k=1
            )
        else:
            documents = store.max_marginal_relevance_search_by_vector([1.0, 0.0], k=1)
    else:
        store.fields = ["custom_id", "text", "metadata"]
        store.client.query.return_value = [deepcopy(entity)]  # type: ignore[attr-defined]
        store.aclient.query.return_value = [deepcopy(entity)]  # type: ignore[attr-defined]
        if use_async:
            documents = await store.asearch_by_metadata("custom_id == 0")
        else:
            documents = store.search_by_metadata("custom_id == 0")

    assert len(documents) == 1
    assert documents[0].id == "0"
    assert documents[0].page_content == "hello"
    assert documents[0].metadata == (
        entity if operation == "metadata" else {"source": "test"}
    )
