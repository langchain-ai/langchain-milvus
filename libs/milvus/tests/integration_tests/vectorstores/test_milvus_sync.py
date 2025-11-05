"""Test Milvus synchronous functionality."""

#
# To run this test properly, please start a Milvus server with the following command:
#
# ```shell
# wget https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh
# bash standalone_embed.sh start
# ```
#
# Here is the reference:
# https://milvus.io/docs/install_standalone-docker.md
#
from typing import ClassVar, Optional

from tests.test_milvus_base import TestMilvusBase


class TestMilvusStandalone(TestMilvusBase):
    """Test class for Milvus vectorstore synchronous functionality."""

    # Override the __test__ attribute to make this class testable
    __test__ = True

    # Add proper type annotation to match base class
    TEST_URI: ClassVar[Optional[str]] = "http://localhost:19530"

    def get_test_uri(self) -> str:
        """Return the URI for the Milvus standalone instance."""
        if self.TEST_URI is None:
            raise ValueError("TEST_URI is not set")
        return self.TEST_URI
