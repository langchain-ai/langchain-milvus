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

import pytest
from langchain_core.documents import Document

from langchain_milvus.function import (
    BM25BuiltInFunction,
    TextEmbeddingBuiltInFunction,
)
from langchain_milvus.vectorstores import Milvus
from tests.test_milvus_base import TestMilvusBase
from tests.utils import (
    FakeEmbeddings,
    assert_docs_equal_without_pk,
    fake_texts,
)


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

    @pytest.mark.requires_apikey
    @pytest.mark.parametrize("enable_dynamic_field", [True, False])
    def test_milvus_builtin_text_embedding_function(
        self, enable_dynamic_field: bool
    ) -> None:
        """
        Test builtin TextEmbedding function (Data In Data Out).
        Requires OpenAI API key configured in milvus.yaml.
        """

        def _add_and_assert(docsearch: Milvus) -> None:
            if enable_dynamic_field:
                metadatas = [{"page": i} for i in range(len(fake_texts))]
            else:
                metadatas = None
            docsearch.add_texts(fake_texts, metadatas=metadatas)
            output = docsearch.similarity_search("foo", k=1)
            if enable_dynamic_field:
                assert_docs_equal_without_pk(
                    output, [Document(page_content=fake_texts[0], metadata={"page": 0})]
                )
            else:
                assert_docs_equal_without_pk(
                    output, [Document(page_content=fake_texts[0])]
                )

        # TextEmbedding only
        text_embedding_func = TextEmbeddingBuiltInFunction(
            input_field_names="text",
            output_field_names="vector",
            dim=1536,
            params={
                "provider": "openai",
                "model_name": "text-embedding-3-small",
            },
        )
        docsearch1 = Milvus(
            embedding_function=None,
            builtin_function=text_embedding_func,
            connection_args={"uri": self.TEST_URI},
            auto_id=True,
            drop_old=True,
            consistency_level="Strong",
            enable_dynamic_field=enable_dynamic_field,
        )
        _add_and_assert(docsearch1)

    @pytest.mark.requires_apikey
    def test_milvus_builtin_text_embedding_with_dense(self) -> None:
        """
        Test combining regular dense embedding with TextEmbedding builtin function.
        """
        text_embedding_func = TextEmbeddingBuiltInFunction(
            input_field_names="text",
            output_field_names="vector_builtin",
            dim=1536,
            params={
                "provider": "openai",
                "model_name": "text-embedding-3-small",
            },
        )
        docsearch = Milvus(
            embedding_function=FakeEmbeddings(),
            builtin_function=text_embedding_func,
            connection_args={"uri": self.TEST_URI},
            auto_id=True,
            drop_old=True,
            consistency_level="Strong",
            vector_field=["vector", "vector_builtin"],
        )
        docsearch.add_texts(fake_texts)
        output = docsearch.similarity_search("foo", k=1)
        assert_docs_equal_without_pk(output, [Document(page_content=fake_texts[0])])

    @pytest.mark.requires_apikey
    def test_milvus_builtin_text_embedding_with_custom_index(self) -> None:
        """
        Test TextEmbedding function with custom index parameters.
        """
        text_embedding_func = TextEmbeddingBuiltInFunction(
            input_field_names="text",
            output_field_names="vector_openai",
            dim=1536,
            params={
                "provider": "openai",
                "model_name": "text-embedding-3-small",
            },
        )

        index_param_1 = {
            "metric_type": "L2",
            "index_type": "AUTOINDEX",
        }
        index_param_2 = {
            "metric_type": "COSINE",
            "index_type": "AUTOINDEX",
        }

        docsearch = Milvus(
            embedding_function=FakeEmbeddings(),
            builtin_function=text_embedding_func,
            connection_args={"uri": self.TEST_URI},
            auto_id=True,
            drop_old=True,
            consistency_level="Strong",
            vector_field=["vector_dense", "vector_openai"],
            index_params=[index_param_1, index_param_2],
        )

        docsearch.add_texts(fake_texts)
        output = docsearch.similarity_search("foo", k=1)
        assert_docs_equal_without_pk(output, [Document(page_content=fake_texts[0])])

        # Verify indexes are created correctly
        assert docsearch.col is not None
        index_list = docsearch.col.indexes
        field_names = [ind.field_name for ind in index_list]
        assert set(field_names) == {"vector_dense", "vector_openai"}

    @pytest.mark.requires_apikey
    def test_milvus_builtin_text_embedding_with_shorten_dim(self) -> None:
        """
        Test TextEmbedding function with shortened dimension output.
        OpenAI supports shortening output dimension to reduce cost.
        """
        text_embedding_func = TextEmbeddingBuiltInFunction(
            input_field_names="text",
            output_field_names="vector",
            dim=512,  # Shortened from 1536
            params={
                "provider": "openai",
                "model_name": "text-embedding-3-small",
                "dim": 512,  # Pass dim to OpenAI API
            },
        )
        docsearch = Milvus(
            embedding_function=None,
            builtin_function=text_embedding_func,
            connection_args={"uri": self.TEST_URI},
            auto_id=True,
            drop_old=True,
            consistency_level="Strong",
        )
        docsearch.add_texts(fake_texts)
        output = docsearch.similarity_search("foo", k=1)
        assert_docs_equal_without_pk(output, [Document(page_content=fake_texts[0])])

    @pytest.mark.requires_apikey
    def test_milvus_builtin_text_embedding_with_bm25(self) -> None:
        """
        Test combining TextEmbedding builtin function with BM25 builtin function.
        This tests the case where both builtin functions are used together.
        """
        text_embedding_func = TextEmbeddingBuiltInFunction(
            input_field_names="text",
            output_field_names="vector_openai",
            dim=1536,
            params={
                "provider": "openai",
                "model_name": "text-embedding-3-small",
            },
        )
        bm25_func = BM25BuiltInFunction(
            input_field_names="text",
            output_field_names="vector_bm25",
        )

        index_param_1 = {
            "metric_type": "COSINE",
            "index_type": "AUTOINDEX",
        }
        index_param_2 = {
            "metric_type": "BM25",
            "index_type": "AUTOINDEX",
        }

        docsearch = Milvus(
            embedding_function=None,
            builtin_function=[text_embedding_func, bm25_func],
            connection_args={"uri": self.TEST_URI},
            auto_id=True,
            drop_old=True,
            consistency_level="Strong",
            vector_field=["vector_openai", "vector_bm25"],
            index_params=[index_param_1, index_param_2],
        )
        docsearch.add_texts(fake_texts)
        output = docsearch.similarity_search("foo", k=1)
        assert_docs_equal_without_pk(output, [Document(page_content=fake_texts[0])])

        # Verify both indexes are created
        assert docsearch.col is not None
        index_list = docsearch.col.indexes
        field_names = [ind.field_name for ind in index_list]
        assert set(field_names) == {"vector_openai", "vector_bm25"}

    @pytest.mark.requires_apikey
    def test_milvus_builtin_text_embedding_with_dense_and_bm25(self) -> None:
        """
        Test combining regular dense embedding, TextEmbedding, and BM25 together.
        This is the most complex combination with all three types.
        """
        text_embedding_func = TextEmbeddingBuiltInFunction(
            input_field_names="text",
            output_field_names="vector_openai",
            dim=1536,
            params={
                "provider": "openai",
                "model_name": "text-embedding-3-small",
            },
        )
        bm25_func = BM25BuiltInFunction(
            input_field_names="text",
            output_field_names="vector_bm25",
        )

        index_param_1 = {
            "metric_type": "L2",
            "index_type": "AUTOINDEX",
        }
        index_param_2 = {
            "metric_type": "COSINE",
            "index_type": "AUTOINDEX",
        }
        index_param_3 = {
            "metric_type": "BM25",
            "index_type": "AUTOINDEX",
        }

        docsearch = Milvus(
            embedding_function=FakeEmbeddings(),
            builtin_function=[text_embedding_func, bm25_func],
            connection_args={"uri": self.TEST_URI},
            auto_id=True,
            drop_old=True,
            consistency_level="Strong",
            vector_field=["vector_dense", "vector_openai", "vector_bm25"],
            index_params=[index_param_1, index_param_2, index_param_3],
        )
        docsearch.add_texts(fake_texts)
        output = docsearch.similarity_search("foo", k=1)
        assert_docs_equal_without_pk(output, [Document(page_content=fake_texts[0])])

        # Verify all three indexes are created
        assert docsearch.col is not None
        index_list = docsearch.col.indexes
        field_names = [ind.field_name for ind in index_list]
        assert set(field_names) == {"vector_dense", "vector_openai", "vector_bm25"}

    def test_milvus_multiple_dense_embeddings(self) -> None:
        """
        Test multiple regular dense embedding functions.
        This ensures client-side multi-vector embedding works correctly.
        """
        dense_embedding_func_1 = FakeEmbeddings()
        dense_embedding_func_2 = FakeEmbeddings()

        index_param_1 = {
            "metric_type": "L2",
            "index_type": "AUTOINDEX",
        }
        index_param_2 = {
            "metric_type": "COSINE",
            "index_type": "AUTOINDEX",
        }

        docsearch = Milvus.from_texts(
            embedding=[dense_embedding_func_1, dense_embedding_func_2],
            texts=fake_texts,
            connection_args={"uri": self.TEST_URI},
            drop_old=True,
            consistency_level="Strong",
            vector_field=["vector_1", "vector_2"],
            index_params=[index_param_1, index_param_2],
        )
        output = docsearch.similarity_search(query=fake_texts[0], k=1)
        assert_docs_equal_without_pk(output, [Document(page_content=fake_texts[0])])

        # Verify both indexes are created
        assert docsearch.col is not None
        index_list = docsearch.col.indexes
        field_names = [ind.field_name for ind in index_list]
        assert set(field_names) == {"vector_1", "vector_2"}

    @pytest.mark.requires_apikey
    def test_milvus_text_embedding_only_no_client_embedding(self) -> None:
        """
        Test TextEmbedding builtin function without client-side embedding.
        This ensures embedding_function=None is properly handled.
        """
        text_embedding_func = TextEmbeddingBuiltInFunction(
            input_field_names="text",
            output_field_names="vector",
            dim=1536,
            params={
                "provider": "openai",
                "model_name": "text-embedding-3-small",
            },
        )
        docsearch = Milvus(
            embedding_function=None,
            builtin_function=text_embedding_func,
            connection_args={"uri": self.TEST_URI},
            auto_id=True,
            drop_old=True,
            consistency_level="Strong",
        )

        # Test add_texts
        docsearch.add_texts(fake_texts)
        output = docsearch.similarity_search("foo", k=1)
        assert_docs_equal_without_pk(output, [Document(page_content=fake_texts[0])])

        # Test add_documents
        documents = [
            Document(page_content="new document 1"),
            Document(page_content="new document 2"),
        ]
        docsearch.add_documents(documents)
        output = docsearch.similarity_search("new document", k=2)
        assert len(output) >= 2
