"""Tests for Milvus upsert error handling."""

from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from langchain_milvus import Milvus


def test_upsert_propagates_add_documents_error() -> None:
    store = Milvus.__new__(Milvus)
    document = Document(page_content="example")
    with patch.object(
        Milvus, "add_documents", side_effect=RuntimeError("insert failed")
    ):
        with pytest.raises(RuntimeError, match="insert failed"):
            store.upsert(documents=[document])
