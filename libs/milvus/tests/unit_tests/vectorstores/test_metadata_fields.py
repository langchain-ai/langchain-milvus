"""Metadata queries must not modify field-selection inputs."""

from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from langchain_milvus import Milvus


@pytest.mark.parametrize("use_default", [False, True])
@pytest.mark.parametrize("async_query", [False, True])
async def test_metadata_query_preserves_fields(
    use_default: bool, async_query: bool
) -> None:
    store = object.__new__(Milvus)
    store.collection_name = "documents"
    store._text_field = "text"
    store.fields = ["source"]
    fields = ["source"]
    selected = None if use_default else fields
    client = MagicMock()
    result = [{"text": "hello", "source": "guide"}]
    client.query = (
        AsyncMock(return_value=result)
        if async_query
        else MagicMock(return_value=result)
    )
    client_property = "aclient" if async_query else "client"
    with patch.object(Milvus, "col", new_callable=PropertyMock, return_value=object()):
        with patch.object(
            Milvus, client_property, new_callable=PropertyMock, return_value=client
        ):
            if async_query:
                docs = await store.asearch_by_metadata(
                    "source == 'guide'", fields=selected
                )
            else:
                docs = store.search_by_metadata("source == 'guide'", fields=selected)

    assert docs[0].page_content == "hello"
    assert client.query.call_args.kwargs["output_fields"] == ["source", "text"]
    assert fields == ["source"]
    assert store.fields == ["source"]
