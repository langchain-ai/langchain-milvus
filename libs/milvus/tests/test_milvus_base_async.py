from abc import ABC, abstractmethod
from typing import Any, ClassVar, List, Optional

import pytest
from langchain_core.documents import Document

from langchain_milvus.function import BM25BuiltInFunction
from langchain_milvus.utils.sparse import BM25SparseEmbedding
from langchain_milvus.vectorstores import Milvus
from tests.utils import (
    DirectionEmbeddings,
    FakeEmbeddings,
    FakeFp16Embeddings,
    FixedValuesEmbeddings,
    assert_docs_equal_without_pk,
    fake_texts,
)


class TestMilvusBaseAsync(ABC):
    __test__ = False
    TEST_URI: ClassVar[Optional[str]] = None

    @abstractmethod
    def get_test_uri(self) -> str:
        pass

    async def _milvus_from_texts(
        self,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        drop: bool = True,
        **kwargs: Any,
    ) -> Milvus:
        if self.TEST_URI is None:
            raise ValueError("TEST_URI must be set in child classes")
        return await Milvus.afrom_texts(
            fake_texts,
            FakeEmbeddings(),
            metadatas=metadatas,
            ids=ids,
            connection_args={"uri": self.TEST_URI},
            drop_old=drop,
            consistency_level="Strong",
            **kwargs,
        )

    async def _get_pks(self, expr: str, docsearch: Milvus) -> List[Any]:
        return await docsearch.aget_pks(expr)  # type: ignore[return-value]

    @pytest.mark.asyncio
    async def test_milvus(self) -> None:
        docsearch = await self._milvus_from_texts()
        output = await docsearch.asimilarity_search("foo", k=1)
        assert_docs_equal_without_pk(output, [Document(page_content="foo")])

    @pytest.mark.asyncio
    async def test_milvus_add_embeddings_search(self) -> None:
        embed_func = FakeEmbeddings()
        docsearch = Milvus(
            embed_func,
            connection_args={"uri": self.TEST_URI},
            drop_old=True,
            consistency_level="Strong",
            auto_id=True,
        )
        await docsearch.aadd_embeddings(texts=fake_texts, embeddings=embed_func.embed_documents(fake_texts))
        output = await docsearch.asimilarity_search("foo", k=1)
        assert_docs_equal_without_pk(output, [Document(page_content="foo")])

    @pytest.mark.asyncio
    async def test_milvus_vector_search(self) -> None:
        docsearch = await self._milvus_from_texts()
        output = await docsearch.asimilarity_search_by_vector(FakeEmbeddings().embed_query("foo"), k=1)
        assert_docs_equal_without_pk(output, [Document(page_content="foo")])

    @pytest.mark.asyncio
    async def test_milvus_with_metadata(self) -> None:
        docsearch = await self._milvus_from_texts(metadatas=[{"label": "test"}] * len(fake_texts))
        output = await docsearch.asimilarity_search("foo", k=1)
        assert_docs_equal_without_pk(output, [Document(page_content="foo", metadata={"label": "test"})])

    @pytest.mark.asyncio
    async def test_milvus_with_id(self) -> None:
        ids = ["id_" + str(i) for i in range(len(fake_texts))]
        docsearch = await self._milvus_from_texts(ids=ids)
        output = await docsearch.asimilarity_search("foo", k=1)
        assert_docs_equal_without_pk(output, [Document(page_content="foo")])
        result = await docsearch.adelete(ids=ids)
        assert result is True
        try:
            ids = ["dup_id" for _ in fake_texts]
            await self._milvus_from_texts(ids=ids)
        except Exception as e:
            assert isinstance(e, AssertionError)

    @pytest.mark.asyncio
    async def test_milvus_with_score(self) -> None:
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": i} for i in range(len(texts))]
        docsearch = await self._milvus_from_texts(metadatas=metadatas)
        output = await docsearch.asimilarity_search_with_score("foo", k=3)
        docs = [o[0] for o in output]
        scores = [o[1] for o in output]
        assert_docs_equal_without_pk(
            docs,
            [
                Document(page_content="foo", metadata={"page": 0}),
                Document(page_content="bar", metadata={"page": 1}),
                Document(page_content="baz", metadata={"page": 2}),
            ],
        )
        assert scores[0] < scores[1] < scores[2]

    @pytest.mark.asyncio
    async def test_milvus_max_marginal_relevance_search(self) -> None:
        """Test end to end construction and MRR search."""
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": i} for i in range(len(texts))]
        docsearch = await self._milvus_from_texts(metadatas=metadatas)
        output = await docsearch.amax_marginal_relevance_search("foo", k=2, fetch_k=3)
        assert_docs_equal_without_pk(
            output,
            [
                Document(page_content="foo", metadata={"page": 0}),
                Document(page_content="bar", metadata={"page": 1}),
            ],
        )

    @pytest.mark.asyncio
    async def test_milvus_max_marginal_relevance_search_with_dynamic_field(
        self,
    ) -> None:
        """Test end to end construction and MRR search with enabling dynamic field."""
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": i} for i in range(len(texts))]
        docsearch = await self._milvus_from_texts(metadatas=metadatas, enable_dynamic_field=True)
        output = await docsearch.amax_marginal_relevance_search("foo", k=2, fetch_k=3)
        assert_docs_equal_without_pk(
            output,
            [
                Document(page_content="foo", metadata={"page": 0}),
                Document(page_content="bar", metadata={"page": 1}),
            ],
        )

    @pytest.mark.asyncio
    async def test_milvus_add_extra(self) -> None:
        """Test end to end construction and MRR search."""
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": i} for i in range(len(texts))]
        docsearch = await self._milvus_from_texts(metadatas=metadatas)

        await docsearch.aadd_texts(texts, metadatas)

        output = await docsearch.asimilarity_search("foo", k=10)
        assert len(output) == 6

    @pytest.mark.asyncio
    async def test_milvus_no_drop(self) -> None:
        """Test construction without dropping old data."""
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": i} for i in range(len(texts))]
        docsearch = await self._milvus_from_texts(metadatas=metadatas)
        del docsearch

        docsearch = await self._milvus_from_texts(metadatas=metadatas, drop=False)

        output = await docsearch.asimilarity_search("foo", k=10)
        assert len(output) == 6

    @pytest.mark.asyncio
    async def test_milvus_get_pks(self) -> None:
        """Test end to end construction and get pks with expr"""
        texts = ["foo", "bar", "baz"]
        metadatas = [{"id": i} for i in range(len(texts))]
        docsearch = await self._milvus_from_texts(metadatas=metadatas)
        expr = "id in [1,2]"
        output = await self._get_pks(expr, docsearch)
        assert len(output) == 2

    @pytest.mark.asyncio
    async def test_search_by_metadata(self) -> None:
        """
        Test metadata-based search in Milvus.

        This function verifies that `search_by_metadata` correctly retrieves
        documents based on a metadata filtering expression.
        """
        # Sample texts and metadata for Milvus collection
        texts = ["Song A", "Song B", "Song C"]
        metadatas = [
            {"id": 1, "SingerName": "IU", "Genre": "Ballad"},
            {"id": 2, "SingerName": "BTS", "Genre": "Pop"},
            {"id": 3, "SingerName": "IU", "Genre": "K-Pop"},
        ]

        # Create a Milvus collection with sample data
        docsearch = await Milvus.afrom_texts(
            connection_args={"uri": self.TEST_URI},
            texts=texts,
            embedding=FakeEmbeddings(),
            metadatas=metadatas,
            auto_id=True,
            consistency_level="Strong",
            drop_old=True,
        )

        # Search for all songs by IU
        output = await docsearch.asearch_by_metadata(expr="SingerName == 'IU'", limit=10)
        assert len(output) == 2  # Expecting 2 results for IU
        assert all(doc.metadata["SingerName"] == "IU" for doc in output)

        # Search for Ballad genre songs
        output = await docsearch.asearch_by_metadata(expr="Genre == 'Ballad'", limit=10)
        assert len(output) == 1  # Expecting 1 result
        assert output[0].metadata["Genre"] == "Ballad"

        # Search with a condition that should return no results
        output = await docsearch.asearch_by_metadata(expr="Genre == 'Rock'", limit=10)
        assert len(output) == 0  # Expecting 0 results

    @pytest.mark.asyncio
    async def test_milvus_delete_entities(self) -> None:
        """Test end to end construction and delete entities"""
        texts = ["foo", "bar", "baz"]
        metadatas = [{"id": i} for i in range(len(texts))]
        docsearch = await self._milvus_from_texts(metadatas=metadatas)
        expr = "id in [1,2]"
        pks = await self._get_pks(expr, docsearch)
        result = await docsearch.adelete(pks)
        assert result is True

    @pytest.mark.asyncio
    async def test_milvus_upsert_entities(self) -> None:
        """Test end to end construction and upsert entities"""
        texts = ["foo", "bar", "baz"]
        metadatas = [{"id": i} for i in range(len(texts))]
        docsearch = await self._milvus_from_texts(metadatas=metadatas)
        expr = "id in [1,2]"
        pks = await self._get_pks(expr, docsearch)
        documents = [
            Document(page_content="test_1", metadata={"id": 1}),
            Document(page_content="test_2", metadata={"id": 3}),
        ]
        await docsearch.aupsert(pks, documents)
        expr = "id in [1,3]"
        res = docsearch.client.query(collection_name=docsearch.collection_name, filter=expr)
        assert len(res) == 2
        assert res[0]["id"] == 1
        assert res[1]["id"] == 3

    @pytest.mark.asyncio
    async def test_milvus_enable_dynamic_field(self) -> None:
        """Test end to end construction and enable dynamic field"""
        texts = ["foo", "bar", "baz"]
        metadatas = [{"id": i} for i in range(len(texts))]
        docsearch = await self._milvus_from_texts(metadatas=metadatas, enable_dynamic_field=True)
        output = await docsearch.asimilarity_search("foo", k=10)
        assert len(output) == 3

        # When enable dynamic field, any new field data will be added to the collection.
        new_metadatas = [{"id_new": i} for i in range(len(texts))]
        await docsearch.aadd_texts(texts, new_metadatas)

        output = await docsearch.asimilarity_search("foo", k=10)
        assert len(output) == 6

        assert set(docsearch.fields) == {
            docsearch._primary_field,
            docsearch._text_field,
            docsearch._vector_field,
        }

    @pytest.mark.asyncio
    async def test_milvus_disable_dynamic_field(self) -> None:
        """Test end to end construction and disable dynamic field"""
        texts = ["foo", "bar", "baz"]
        metadatas = [{"id": i} for i in range(len(texts))]
        docsearch = await self._milvus_from_texts(metadatas=metadatas, enable_dynamic_field=False)
        output = await docsearch.asimilarity_search("foo", k=10)
        assert len(output) == 3
        # ["pk", "text", "vector", "id"]
        assert set(docsearch.fields) == {
            docsearch._primary_field,
            docsearch._text_field,
            docsearch._vector_field,
            "id",
        }

        # Try to add new fields "id_new", but since dynamic field is disabled,
        # all fields in the collection is specified as ["pk", "text", "vector", "id"],
        # new field information "id_new" will not be added.
        new_metadatas = [{"id": i, "id_new": i} for i in range(len(texts))]
        await docsearch.aadd_texts(texts, new_metadatas)
        output = await docsearch.asimilarity_search("foo", k=10)
        assert len(output) == 6
        for doc in output:
            assert set(doc.metadata.keys()) == {"id", "pk"}  # `id_new` is not added.

        # When disable dynamic field,
        # missing data of the created fields "id", will raise an exception.
        with pytest.raises(Exception):
            new_metadatas = [{"id_new": i} for i in range(len(texts))]
            await docsearch.aadd_texts(texts, new_metadatas)

    @pytest.mark.asyncio
    async def test_milvus_metadata_field(self) -> None:
        """Test end to end construction and use metadata field"""
        texts = ["foo", "bar", "baz"]
        metadatas = [{"id": i} for i in range(len(texts))]
        docsearch = await self._milvus_from_texts(metadatas=metadatas, metadata_field="metadata")
        output = await docsearch.asimilarity_search("foo", k=10)
        assert len(output) == 3

        new_metadatas = [{"id_new": i} for i in range(len(texts))]
        await docsearch.aadd_texts(texts, new_metadatas)

        output = await docsearch.asimilarity_search("foo", k=10)
        assert len(output) == 6

        assert set(docsearch.fields) == {
            docsearch._primary_field,
            docsearch._text_field,
            docsearch._vector_field,
            docsearch._metadata_field,
        }

    @pytest.mark.asyncio
    async def test_milvus_enable_dynamic_field_with_partition_key(self) -> None:
        """
        Test end to end construction and enable dynamic field
        with partition_key_field
        """
        texts = ["foo", "bar", "baz"]
        metadatas = [{"id": i, "namespace": f"name_{i}"} for i in range(len(texts))]

        docsearch = await self._milvus_from_texts(
            metadatas=metadatas,
            enable_dynamic_field=True,
            partition_key_field="namespace",
        )

        # without namespace filter
        output = await docsearch.asimilarity_search("foo", k=10)
        assert len(output) == 3

        assert set(docsearch.fields) == {
            docsearch._primary_field,
            docsearch._text_field,
            docsearch._vector_field,
            docsearch._partition_key_field,
        }

    @pytest.mark.skip(reason="BM25SparseEmbedding will be deprecated in the future.")
    @pytest.mark.asyncio
    async def test_milvus_sparse_embeddings(self) -> None:
        texts = [
            "In 'The Clockwork Kingdom' by Augusta Wynter, a brilliant inventor "
            "discovers a hidden world of clockwork machines and ancient magic, where"
            " a rebellion is brewing against the tyrannical ruler of the land. In 'The"
            " Phantom Pilgrim' by Rowan Welles, a charismatic smuggler is hired by a "
            "mysterious organization to transport a valuable artifact across a war-torn"
            " continent, but soon finds themselves pursued by assassins and rival "
            "factions. In 'The Dreamwalker's Journey' by Lyra Snow, a young "
            "dreamwalker discovers she has the ability to enter people's dreams, but "
            "soon finds herself trapped in a surreal world of nightmares and "
            "illusions, where the boundaries between reality and fantasy blur.",
        ]
        try:
            sparse_embedding_func = BM25SparseEmbedding(corpus=texts)
        except LookupError:
            import nltk  # type: ignore[import]

            nltk.download("punkt_tab")
            sparse_embedding_func = BM25SparseEmbedding(corpus=texts)

        docsearch = await Milvus.afrom_texts(
            embedding=sparse_embedding_func,
            texts=texts,
            connection_args={"uri": self.TEST_URI},
            drop_old=True,
            consistency_level="Strong",
        )

        output = await docsearch.asimilarity_search("Pilgrim", k=1)
        assert "Pilgrim" in output[0].page_content

    @pytest.mark.asyncio
    async def test_milvus_array_field(self) -> None:
        """Manually specify metadata schema, including an array_field.
        For more information about array data type and filtering, please refer to
        https://milvus.io/docs/array_data_type.md
        """
        from pymilvus import DataType

        texts = ["foo", "bar", "baz"]
        metadatas = [{"id": i, "array_field": [i, i + 1, i + 2]} for i in range(len(texts))]

        # Manually specify metadata schema, including an array_field.
        # If some fields are not specified, it will automatically infer their schemas.
        docsearch = await self._milvus_from_texts(
            metadatas=metadatas,
            metadata_schema={
                "array_field": dict(
                    dtype=DataType.ARRAY,
                    element_type=DataType.INT64,
                    max_capacity=50,
                ),
            },
        )
        output = await docsearch.asimilarity_search("foo", k=10, expr="array_field[0] < 2")
        assert len(output) == 2
        output = await docsearch.asimilarity_search("foo", k=10, expr="ARRAY_CONTAINS(array_field, 3)")
        assert len(output) == 2

        # If we use enable_dynamic_field,
        # there is no need to manually specify metadata schema.
        docsearch = await self._milvus_from_texts(
            enable_dynamic_field=True,
            metadatas=metadatas,
        )
        output = await docsearch.asimilarity_search("foo", k=10, expr="array_field[0] < 2")
        assert len(output) == 2
        output = await docsearch.asimilarity_search("foo", k=10, expr="ARRAY_CONTAINS(array_field, 3)")
        assert len(output) == 2

    @pytest.mark.asyncio
    async def test_milvus_vector_field(self) -> None:
        # Support custom vector field schema, e.g. supporting Float16 and BFloat
        # https://milvus.io/docs/release_notes.md#Float16-and-BFloat--Vector-DataType
        from pymilvus import DataType

        texts = ["foo", "bar", "baz"]

        docsearch = await Milvus.afrom_texts(
            embedding=FakeFp16Embeddings(),
            texts=texts,
            connection_args={"uri": self.TEST_URI},
            vector_schema=dict(
                dtype=DataType.FLOAT16_VECTOR,
                dim=10,
                # or kwargs={"dim": 10},
            ),
            index_params={
                "metric_type": "L2",
                "index_type": "FLAT",  # For milvus lite, only support FLAT for fp16
            },
            drop_old=True,
            consistency_level="Strong",
        )

        output = await docsearch.asimilarity_search("foo", k=1)
        assert_docs_equal_without_pk(output, [Document(page_content="foo")])

    @pytest.mark.asyncio
    async def test_milvus_multi_vector_embeddings(self) -> None:
        sparse_embedding_func = BM25SparseEmbedding(corpus=fake_texts)
        dense_embedding_func_1 = FakeEmbeddings()
        dense_embeddings_func_2 = FakeEmbeddings()
        docsearch = await Milvus.afrom_texts(
            embedding=[
                sparse_embedding_func,
                dense_embedding_func_1,
                dense_embeddings_func_2,
            ],
            texts=fake_texts,
            connection_args={"uri": self.TEST_URI},
            drop_old=True,
            consistency_level="Strong",
        )
        output = await docsearch.asimilarity_search(query=fake_texts[0], k=1)
        assert_docs_equal_without_pk(output, [Document(page_content=fake_texts[0])])

    @pytest.mark.asyncio
    async def test_milvus_multi_vector_with_index_params(self) -> None:
        """Test setting index params which are different from the defaults."""
        index_param_1 = {
            "metric_type": "COSINE",
            "index_type": "AUTOINDEX",
        }
        index_param_2 = {
            "metric_type": "IP",
            "index_type": "AUTOINDEX",
        }

        docsearch = await Milvus.afrom_texts(
            texts=fake_texts,
            embedding=[FakeEmbeddings(), FakeEmbeddings()],
            index_params=[index_param_1, index_param_2],
            vector_field=["vec_field_1", "vec_field_2"],
            connection_args={"uri": self.TEST_URI},
            drop_old=True,
            consistency_level="Strong",
        )

        assert docsearch.col is not None
        assert isinstance(docsearch.index_params, list) and len(docsearch.index_params) == 2
        assert isinstance(docsearch.search_params, list) and len(docsearch.search_params) == 2

        # The order of the indexes is not guaranteed, so we need to check
        index_list = docsearch.col.indexes
        if index_list[0].field_name == "vec_field_1":
            index_1 = index_list[0]
            index_2 = index_list[1]
        else:
            index_1 = index_list[1]
            index_2 = index_list[0]

        field_names = [ind.field_name for ind in index_list]
        assert set(field_names) == {"vec_field_1", "vec_field_2"}

        assert index_1.field_name == "vec_field_1"
        assert index_1.params["metric_type"] == "COSINE"
        assert docsearch.search_params[0]["metric_type"] == "COSINE"

        assert index_2.field_name == "vec_field_2"
        assert index_2.params["metric_type"] == "IP"
        assert docsearch.search_params[1]["metric_type"] == "IP"

    @pytest.mark.asyncio
    async def test_milvus_multi_vector_search_with_ranker(self) -> None:
        """Test hybrid search with specified ranker"""

        index_param_1 = {
            "metric_type": "L2",
            "index_type": "AUTOINDEX",
        }
        index_param_2 = {
            "metric_type": "L2",
            "index_type": "AUTOINDEX",
        }

        # Force the query vector to always be identical
        # to the embeddings of the *first* document
        embedding_1 = FixedValuesEmbeddings(documents_base_val=0.0, query_val=float(0))
        # Force the query to always be identical to the *last* document's embeddings
        embedding_2 = FixedValuesEmbeddings(documents_base_val=0.0, query_val=float(len(fake_texts)))
        docsearch = await Milvus.afrom_texts(
            embedding=[embedding_1, embedding_2],
            texts=fake_texts,
            index_params=[index_param_1, index_param_2],
            connection_args={"uri": self.TEST_URI},
            drop_old=True,
            consistency_level="Strong",
        )

        query = fake_texts[0]
        output = await docsearch.asimilarity_search(
            query=query,
            ranker_type="weighted",
            ranker_params={"weights": [1.0, 0.0]},  # Count for first embeddings only
            k=1,
        )
        assert_docs_equal_without_pk(output, [Document(page_content=fake_texts[0])])

        output = await docsearch.asimilarity_search(
            query=query,
            ranker_type="weighted",
            ranker_params={"weights": [0.0, 1.0]},  # Count for second embeddings only
            k=1,
        )
        assert_docs_equal_without_pk(output, [Document(page_content=fake_texts[-1])])

    @pytest.mark.parametrize("metric_type", ["L2", "IP", "COSINE"])
    @pytest.mark.parametrize("score_threshold", [0.5001, 0.4999])
    @pytest.mark.asyncio
    async def test_milvus_similarity_search_with_relevance_scores(
        self, metric_type: str, score_threshold: float
    ) -> None:
        """Test similarity search with relevance scores"""
        docsearch = Milvus(
            embedding_function=DirectionEmbeddings(),
            connection_args={
                "uri": self.TEST_URI,
            },
            auto_id=True,
            drop_old=True,
            consistency_level="Strong",
            index_params={
                "metric_type": metric_type,
                "index_type": "FLAT",
                "params": {},
            },
        )
        await docsearch.aadd_texts(["left", "right", "up", "down"])
        output = await docsearch.asimilarity_search_with_score("down", k=4)
        assert output[0][0].page_content == "down"
        assert output[-1][0].page_content == "up"
        retriever = docsearch.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "score_threshold": score_threshold,
                "k": 4,
            },
        )
        retrival_output = await retriever.ainvoke("down")
        if score_threshold == 0.5001:
            assert len(retrival_output) == 1
            assert retrival_output[0].page_content == "down"
        elif score_threshold == 0.4999:
            assert len(retrival_output) == 3
            assert retrival_output[0].page_content == "down"

    @pytest.mark.parametrize("enable_dynamic_field", [True, False])
    @pytest.mark.asyncio
    async def test_milvus_builtin_bm25_function(self, enable_dynamic_field: bool) -> None:
        """
        Test builtin BM25 function
        """

        async def _add_and_assert(docsearch: Milvus) -> None:
            if enable_dynamic_field:
                metadatas = [{"page": i} for i in range(len(fake_texts))]
            else:
                metadatas = None
            await docsearch.aadd_texts(fake_texts, metadatas=metadatas)
            output = await docsearch.asimilarity_search("foo", k=1)
            if enable_dynamic_field:
                assert_docs_equal_without_pk(output, [Document(page_content=fake_texts[0], metadata={"page": 0})])
            else:
                assert_docs_equal_without_pk(output, [Document(page_content=fake_texts[0])])

        # BM25 only
        docsearch1 = Milvus(
            embedding_function=[],
            builtin_function=[BM25BuiltInFunction()],
            connection_args={"uri": self.TEST_URI},
            auto_id=True,
            drop_old=True,
            consistency_level="Strong",
            vector_field="sparse",
            enable_dynamic_field=enable_dynamic_field,
        )
        await _add_and_assert(docsearch1)

        # Dense embedding + BM25
        docsearch2 = Milvus(
            embedding_function=FakeEmbeddings(),
            builtin_function=[BM25BuiltInFunction()],
            connection_args={"uri": self.TEST_URI},
            auto_id=True,
            drop_old=True,
            consistency_level="Strong",
            vector_field="sparse",
            enable_dynamic_field=enable_dynamic_field,
        )
        await _add_and_assert(docsearch2)

        # Dense embedding + BM25 + custom index params
        index_param_1 = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
        }
        index_param_2 = {
            "metric_type": "BM25",
            "index_type": "AUTOINDEX",
        }
        docsearch3 = Milvus(
            embedding_function=[
                FakeEmbeddings(),
            ],
            builtin_function=[
                BM25BuiltInFunction(
                    input_field_names="text00",
                    output_field_names="sparse00",
                )
            ],
            index_params=[index_param_1, index_param_2],
            connection_args={"uri": self.TEST_URI},
            auto_id=True,
            drop_old=True,
            consistency_level="Strong",
            text_field="text00",
            vector_field=["dense00", "sparse00"],
            enable_dynamic_field=enable_dynamic_field,
        )
        await _add_and_assert(docsearch3)
