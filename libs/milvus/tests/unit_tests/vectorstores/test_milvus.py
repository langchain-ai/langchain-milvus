"""Test Milvus functionality."""

import tempfile
from typing import ClassVar, Optional

import pytest

from tests.test_milvus_base import TestMilvusBase


class TestMilvusLite(TestMilvusBase):
    """Test class for Milvus vectorstore functionality."""

    # Override the __test__ attribute to make this class testable
    __test__ = True

    # Define TEST_URI as a class variable with proper type annotation
    TEST_URI: ClassVar[Optional[str]] = None

    @pytest.fixture(scope="class")
    def milvus_uri(self) -> str:
        """Creates a temporary database file path to use for tests."""
        with tempfile.NamedTemporaryFile(suffix=".db") as temp_file:
            return temp_file.name

    @pytest.fixture(autouse=True)
    def setup_test_uri(self, milvus_uri: str) -> None:
        """Automatically setup TEST_URI for each test method."""
        self.TEST_URI = milvus_uri  # type: ignore[misc]

    def get_test_uri(self) -> str:
        """Return the URI for the Milvus Lite instance."""
        if self.TEST_URI is None:
            raise ValueError("TEST_URI is not set")
        return self.TEST_URI

    @pytest.mark.skip(
        reason="There are some nuance difference between Milvus-Lite and Milvus "
        "server, which would lead to failure of this test."
    )
    def test_milvus_enable_dynamic_field(self) -> None:
        return super().test_milvus_enable_dynamic_field()

    @pytest.mark.skip(reason="Milvus-Lite doesn't support built-in full-text search yet")
    def test_milvus_builtin_bm25_function(self, enable_dynamic_field: bool) -> None:
        return super().test_milvus_builtin_bm25_function(enable_dynamic_field)
