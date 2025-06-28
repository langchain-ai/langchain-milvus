from __future__ import annotations

import logging
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from pymilvus import (
    AnnSearchRequest,
    AsyncMilvusClient,
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    FunctionType,
    MilvusClient,
    MilvusException,
    RRFRanker,
    WeightedRanker,
    utility,
)
from pymilvus.client.types import LoadState  # type: ignore
from pymilvus.orm.types import infer_dtype_bydata  # type: ignore

from langchain_milvus.function import BaseMilvusBuiltInFunction, BM25BuiltInFunction
from langchain_milvus.utils.constant import PRIMARY_FIELD, TEXT_FIELD, VECTOR_FIELD
from langchain_milvus.utils.sparse import BaseSparseEmbedding

logger = logging.getLogger(__name__)

# - If you only need a local vector database for small scale data or prototyping,
# setting the uri as a local file, e.g.`./milvus.db`, is the most convenient method,
# as it automatically utilizes [Milvus Lite](https://milvus.io/docs/milvus_lite.md)
# to store all data in this file.
#
# - If you have large scale of data, say more than a million vectors, you can set up a
# more performant Milvus server on [Docker or Kubernetes](https://milvus.io/docs/quickstart.md).
# In this setup, please use the server address and port as your uri, e.g.`http://localhost:19530`.
# If you enable the authentication feature on Milvus, use
# "<your_username>:<your_password>" as the token, otherwise don't set the token.
#
# - If you use [Zilliz Cloud](https://zilliz.com/cloud), the fully managed cloud service
# for Milvus, adjust the `uri` and `token`, which correspond to the
# [Public Endpoint and API key](https://docs.zilliz.com/docs/on-zilliz-cloud-console#cluster-details)
# in Zilliz Cloud.

DEFAULT_MILVUS_CONNECTION = {
    "uri": "http://localhost:19530",
    # "token": "",
}

Matrix = Union[List[List[float]], List[np.ndarray], np.ndarray]


def cosine_similarity(X: Matrix, Y: Matrix) -> np.ndarray:
    """Row-wise cosine similarity between two equal-width matrices."""
    if len(X) == 0 or len(Y) == 0:
        return np.array([])

    X = np.array(X)
    Y = np.array(Y)
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"Number of columns in X and Y must be the same. X has shape {X.shape} "
            f"and Y has shape {Y.shape}."
        )
    try:
        import simsimd as simd

        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32)
        Z = 1 - np.array(simd.cdist(X, Y, metric="cosine"))
        return Z
    except ImportError:
        logger.debug(
            "Unable to import simsimd, defaulting to NumPy implementation. If you want "
            "to use simsimd please install with `pip install simsimd`."
        )
        X_norm = np.linalg.norm(X, axis=1)
        Y_norm = np.linalg.norm(Y, axis=1)
        # Ignore divide by zero errors run time warnings as those are handled below.
        with np.errstate(divide="ignore", invalid="ignore"):
            similarity = np.dot(X, Y.T) / np.outer(X_norm, Y_norm)
        similarity[np.isnan(similarity) | np.isinf(similarity)] = 0.0
        return similarity


def maximal_marginal_relevance(
    query_embedding: np.ndarray,
    embedding_list: list,
    lambda_mult: float = 0.5,
    k: int = 4,
) -> List[int]:
    """Calculate maximal marginal relevance.

    Args:
        query_embedding: The query embedding.
        embedding_list: The list of embeddings.
        lambda_mult: The lambda multiplier. Defaults to 0.5.
        k: The number of results to return. Defaults to 4.

    Returns:
        List[int]: The list of indices.
    """
    if min(k, len(embedding_list)) <= 0:
        return []
    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)
    similarity_to_query = cosine_similarity(query_embedding, embedding_list)[0]
    most_similar = int(np.argmax(similarity_to_query))
    idxs = [most_similar]
    selected = np.array([embedding_list[most_similar]])
    while len(idxs) < min(k, len(embedding_list)):
        best_score = -np.inf
        idx_to_add = -1
        similarity_to_selected = cosine_similarity(embedding_list, selected)
        for i, query_score in enumerate(similarity_to_query):
            if i in idxs:
                continue
            redundant_score = max(similarity_to_selected[i])
            equation_score = (
                lambda_mult * query_score - (1 - lambda_mult) * redundant_score
            )
            if equation_score > best_score:
                best_score = equation_score
                idx_to_add = i
        idxs.append(idx_to_add)
        selected = np.append(selected, [embedding_list[idx_to_add]], axis=0)
    return idxs


EmbeddingType = Union[Embeddings, BaseSparseEmbedding]
T = TypeVar("T")


class Milvus(VectorStore):
    """Milvus vector store integration.

    Setup:
        Install ``langchain_milvus`` package:

        .. code-block:: bash

            pip install -qU  langchain_milvus

    Key init args — indexing params:
        collection_name: str
            Name of the collection.
        collection_description: str
            Description of the collection.
        embedding_function: Union[Embeddings, BaseSparseEmbedding]
            Embedding function to use.

    Key init args — client params:
        connection_args: Optional[dict]
            Connection arguments.

    Instantiate:
        .. code-block:: python

            from langchain_milvus import Milvus
            from langchain_openai import OpenAIEmbeddings

            URI = "./milvus_example.db"

            vector_store = Milvus(
                embedding_function=OpenAIEmbeddings(),
                connection_args={"uri": URI},
            )

    Add Documents:
        .. code-block:: python

            from langchain_core.documents import Document

            document_1 = Document(page_content="foo", metadata={"baz": "bar"})
            document_2 = Document(page_content="thud", metadata={"baz": "baz"})
            document_3 = Document(page_content="i will be deleted :(", metadata={"baz": "qux"})

            documents = [document_1, document_2, document_3]
            ids = ["1", "2", "3"]
            vector_store.add_documents(documents=documents, ids=ids)

    Delete Documents:
        .. code-block:: python

            vector_store.delete(ids=["3"])

    Search:
        .. code-block:: python

            results = vector_store.similarity_search(query="thud",k=1)
            for doc in results:
                print(f"* {doc.page_content} [{doc.metadata}]")

        .. code-block:: python

            * thud [{'baz': 'baz', 'pk': '2'}]

    Search with score:
        .. code-block:: python

            results = vector_store.similarity_search_with_score(query="qux",k=1)
            for doc, score in results:
                print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")

        .. code-block:: python

            * [SIM=0.335463] foo [{'baz': 'bar', 'pk': '1'}]

    Async:
        .. code-block:: python

            # add documents
            # await vector_store.aadd_documents(documents=documents, ids=ids)

            # delete documents
            # await vector_store.adelete(ids=["3"])

            # search
            # results = vector_store.asimilarity_search(query="thud",k=1)

            # search with score
            results = await vector_store.asimilarity_search_with_score(query="qux",k=1)
            for doc,score in results:
                print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")

        .. code-block:: python

            * [SIM=0.335463] foo [{'baz': 'bar', 'pk': '1'}]

    Use as Retriever:
        .. code-block:: python

            retriever = vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 1, "fetch_k": 2, "lambda_mult": 0.5},
            )
            retriever.invoke("thud")

        .. code-block:: python

            [Document(metadata={'baz': 'baz', 'pk': '2'}, page_content='thud')]

    """  # noqa: E501

    def __init__(
        self,
        embedding_function: Optional[Union[EmbeddingType, List[EmbeddingType]]],
        collection_name: str = "LangChainCollection",
        collection_description: str = "",
        collection_properties: Optional[dict[str, Any]] = None,
        connection_args: Optional[dict[str, Any]] = None,
        consistency_level: str = "Session",
        index_params: Optional[Union[dict, List[dict]]] = None,
        search_params: Optional[Union[dict, List[dict]]] = None,
        drop_old: Optional[bool] = False,
        auto_id: bool = False,
        *,
        primary_field: str = PRIMARY_FIELD,
        text_field: str = TEXT_FIELD,
        vector_field: Union[str, List[str]] = VECTOR_FIELD,
        enable_dynamic_field: bool = False,
        metadata_field: Optional[str] = None,
        partition_key_field: Optional[str] = None,
        num_partitions: Optional[int] = None,
        partition_names: Optional[list] = None,
        replica_number: int = 1,
        timeout: Optional[float] = None,
        num_shards: Optional[int] = None,
        vector_schema: Optional[Union[dict[str, Any], List[dict[str, Any]]]] = None,
        metadata_schema: Optional[dict[str, Any]] = None,
        builtin_function: Optional[
            Union[BaseMilvusBuiltInFunction, List[BaseMilvusBuiltInFunction]]
        ] = None,
    ):
        """Initialize the Milvus vector store."""
        # Default search params when one is not provided.
        self.default_search_params = {
            "FLAT": {"metric_type": "L2", "params": {}},
            "IVF_FLAT": {"metric_type": "L2", "params": {"nprobe": 10}},
            "IVF_SQ8": {"metric_type": "L2", "params": {"nprobe": 10}},
            "IVF_PQ": {"metric_type": "L2", "params": {"nprobe": 10}},
            "HNSW": {"metric_type": "L2", "params": {"ef": 10}},
            "RHNSW_FLAT": {"metric_type": "L2", "params": {"ef": 10}},
            "RHNSW_SQ": {"metric_type": "L2", "params": {"ef": 10}},
            "RHNSW_PQ": {"metric_type": "L2", "params": {"ef": 10}},
            "IVF_HNSW": {"metric_type": "L2", "params": {"nprobe": 10, "ef": 10}},
            "ANNOY": {"metric_type": "L2", "params": {"search_k": 10}},
            "SCANN": {"metric_type": "L2", "params": {"search_k": 10}},
            "AUTOINDEX": {"metric_type": "L2", "params": {}},
            "GPU_CAGRA": {
                "metric_type": "L2",
                "params": {
                    "itopk_size": 128,
                    "search_width": 4,
                    "min_iterations": 0,
                    "max_iterations": 0,
                    "team_size": 0,
                },
            },
            "GPU_IVF_FLAT": {"metric_type": "L2", "params": {"nprobe": 10}},
            "GPU_IVF_PQ": {"metric_type": "L2", "params": {"nprobe": 10}},
            "GPU_BRUTE_FORCE": {"metric_type": "L2", "params": {"nprobe": 10}},
            "SPARSE_INVERTED_INDEX": {
                "metric_type": "IP",
                "params": {"drop_ratio_build": 0.2},
            },
            "SPARSE_WAND": {"metric_type": "IP", "params": {"drop_ratio_build": 0.2}},
        }

        if not embedding_function and not builtin_function:
            raise ValueError(
                "Either `embedding_function` or `builtin_function` should be provided."
            )

        self.embedding_func: Optional[
            Union[EmbeddingType, List[EmbeddingType]]
        ] = self._from_list(embedding_function)
        self.builtin_func: Optional[
            Union[BaseMilvusBuiltInFunction, List[BaseMilvusBuiltInFunction]]
        ] = self._from_list(builtin_function)
        self.collection_name = collection_name
        self.collection_description = collection_description
        self.collection_properties = collection_properties
        self.index_params = index_params
        self.search_params = search_params
        self.consistency_level = consistency_level
        self.auto_id = auto_id
        self.num_partitions = num_partitions

        # In order for a collection to be compatible, pk needs to be varchar
        self._primary_field = primary_field
        # In order for compatibility, the text field will need to be called "text"
        self._text_field = text_field

        self._check_vector_field(vector_field, vector_schema)
        if metadata_field:
            logger.warning(
                "DeprecationWarning: `metadata_field` is about to be deprecated, "
                "please set `enable_dynamic_field`=True instead."
            )
        if enable_dynamic_field and metadata_field:
            metadata_field = None
            logger.warning(
                "When `enable_dynamic_field` is True, `metadata_field` is ignored."
            )
        self.enable_dynamic_field = enable_dynamic_field
        self._metadata_field = metadata_field
        self._partition_key_field = partition_key_field
        self.fields: list[str] = []
        self.partition_names = partition_names
        self.replica_number = replica_number
        self.timeout = timeout
        self.num_shards = num_shards
        self.metadata_schema = metadata_schema

        # Create the connection to the server
        if connection_args is None:
            connection_args = DEFAULT_MILVUS_CONNECTION

        # Store connection args for potential async client creation
        self._connection_args = connection_args

        self._milvus_client = MilvusClient(
            **connection_args,
        )

        # Safely create AsyncMilvusClient to avoid failures in multithreading
        # environments
        try:
            self._async_milvus_client = AsyncMilvusClient(
                **connection_args,
            )
        except Exception as e:
            # If creation fails (e.g., no event loop in multithreading
            # environment), set to None. This won't affect Milvus instance
            # creation, async operations will only fail when actually needed
            logger.warning(
                f"Failed to initialize AsyncMilvusClient during Milvus "
                f"initialization: {e}. Async operations will be unavailable "
                f"until AsyncMilvusClient is successfully created."
            )
            self._async_milvus_client = None

        self.alias = self.client._using

        self.col: Optional[Collection] = None

        # Grab the existing collection if it exists
        if utility.has_collection(self.collection_name, using=self.alias):
            self.col = Collection(
                self.collection_name,
                using=self.alias,
            )
            if self.collection_properties is not None:
                self.col.set_properties(self.collection_properties)
        # If need to drop old, drop it
        if drop_old and isinstance(self.col, Collection):
            self.col.drop()
            self.col = None

        # Initialize the vector store
        self._init(
            partition_names=partition_names,
            replica_number=replica_number,
            timeout=timeout,
        )

    def _check_vector_field(
        self,
        vector_field: Union[str, List[str]],
        vector_schema: Optional[Union[dict[str, Any], List[dict[str, Any]]]] = None,
    ) -> None:
        """
        Check the validity of vector_field and vector_schema,
        as well as the relationships with embedding_func and builtin_func.
        """
        assert len(self._as_list(vector_field)) == len(
            set(self._as_list(vector_field))
        ), "Vector field names should be unique."

        vector_fields_from_function = []
        for builtin_function in self._as_list(self.builtin_func):
            vector_fields_from_function.extend(
                self._as_list(builtin_function.output_field_names)
            )
        # Check there are not overlapping fields
        assert len(vector_fields_from_function) == len(
            set(vector_fields_from_function)
        ), "When using builtin functions, there should be no overlapping fields."

        embedding_fields_expected = []
        for field in self._as_list(vector_field):
            if field not in vector_fields_from_function:
                embedding_fields_expected.append(field)

        # Number of customized fields <= number of embedding functions
        if len(embedding_fields_expected) <= len(self._as_list(self.embedding_func)):  # type: ignore[arg-type]
            vector_fields_from_embedding = embedding_fields_expected
            appending_fields = []
            for i in range(
                len(embedding_fields_expected),
                len(self._as_list(self.embedding_func)),  # type: ignore[arg-type]
            ):
                appending_fields.append(f"vector_{i + 1}")
            vector_fields_from_embedding.extend(appending_fields)
            if len(appending_fields) > 0:
                logger.warning(
                    "When multiple embeddings function are used, one should provide "
                    "matching `vector_field` names. "
                    "Using generated vector names %s",
                    appending_fields,
                )
        # Number of customized fields > number of embedding functions
        else:
            raise ValueError(
                f"Too many custom fields: {embedding_fields_expected}."
                f" They cannot be mapped to a limited number of embedding functions,"
                f" nor do they belong to any build-in function."
            )

        assert (
            len(set(vector_fields_from_function) & set(vector_fields_from_embedding))
            == 0
        ), (
            "Vector fields from embeddings and vector fields from builtin functions "
            "should not overlap."
        )
        all_vector_fields = vector_fields_from_embedding + vector_fields_from_function
        # For backward compatibility, the vector field needs to be called "vector",
        # and it can be either a list or a string.
        self._vector_field: Union[str, List[str]] = cast(
            Union[str, List[str]], self._from_list(all_vector_fields)
        )
        self._vector_fields_from_embedding: List[str] = vector_fields_from_embedding
        self._vector_fields_from_function: List[str] = vector_fields_from_function

        # Check vector schema and prepare vector schema map
        self.vector_schema = vector_schema
        self._vector_schema_map: Dict[str, dict] = {}
        if self.vector_schema:
            if len(self._as_list(self.vector_schema)) == 1:
                assert len(self._as_list(self._vector_field)) == 1, (
                    "When only one custom vector_schema is provided, "
                    "it should keep the vector store has only one vector field."
                )
                vector_field_ = cast(str, self._from_list(self._vector_field))
                vector_schema_ = cast(dict, self._from_list(self.vector_schema))
                self._vector_schema_map[vector_field_] = vector_schema_
            else:
                if self._is_embedding_only or self._is_function_only:
                    assert len(self._as_list(self._vector_field)) == len(
                        self._as_list(self.vector_schema)
                    ), (
                        "You should provide the same number of custom `vector_schema`s "
                        "as the number of corresponding `vector_field`s."
                    )
                else:
                    # If both embedding and builtin functions are provided,
                    # it must specify vector_schema for each vector field.
                    assert len(self._as_list(vector_field)) == len(
                        self._as_list(self.vector_schema)
                    ), (
                        "When multiple custom `vector_schema`s are provided, "
                        "you should provide the same number of corresponding "
                        "`vector_field`s."
                    )
                for field, vector_schema in zip(
                    self._as_list(vector_field), self._as_list(self.vector_schema)
                ):
                    self._vector_schema_map[field] = vector_schema
        else:
            self._vector_schema_map = {
                field: {} for field in self._as_list(self._vector_field)
            }

        # Check index param and prepare index param map
        self._index_param_map: Dict[str, dict] = {}
        if self.index_params:
            if len(self._as_list(self.index_params)) == 1:
                assert len(self._as_list(self._vector_field)) == 1, (
                    "When only one custom index_params is provided, "
                    "it should keep the vector store has only one vector field."
                )
                vector_field_ = cast(str, self._from_list(self._vector_field))
                index_params_ = cast(dict, self._from_list(self.index_params))
                self._index_param_map[vector_field_] = index_params_
            else:
                if self._is_embedding_only or self._is_function_only:
                    assert len(self._as_list(self._vector_field)) == len(
                        self._as_list(self.index_params)
                    ), (
                        "You should provide the same number of custom `index_params`s "
                        "as the number of corresponding `vector_field`s."
                    )
                else:
                    # If both embedding and builtin functions are provided,
                    # it must specify index_params for each vector field.
                    assert len(self._as_list(vector_field)) == len(
                        self._as_list(self.index_params)
                    ), (
                        "When multiple custom `index_params`s are provided, "
                        "you should provide the same number of corresponding "
                        "`vector_field`s."
                    )
                for field, index_params in zip(
                    self._as_list(vector_field), self._as_list(self.index_params)
                ):
                    self._index_param_map[field] = index_params
        else:
            self._index_param_map = {
                field: {} for field in self._as_list(self._vector_field)
            }

    @property
    def embeddings(self) -> Optional[Union[EmbeddingType, List[EmbeddingType]]]:  # type: ignore
        """Get embedding function(s)."""
        return self.embedding_func

    @property
    def client(self) -> MilvusClient:
        """Get client."""
        return self._milvus_client

    @property
    def aclient(self) -> AsyncMilvusClient:
        """Get async client."""
        if self._async_milvus_client is None:
            # Try to create AsyncMilvusClient in current environment
            try:
                import asyncio
                import threading

                # Check current thread environment
                current_thread = threading.current_thread()
                is_main_thread = current_thread is threading.main_thread()

                try:
                    # Try to get current event loop
                    loop = asyncio.get_event_loop()
                    if loop.is_running() and not is_main_thread:
                        # In non-main thread with running loop, create directly
                        self._async_milvus_client = AsyncMilvusClient(
                            **self._connection_args
                        )
                    elif not loop.is_running():
                        # Loop exists but not running, set it as current thread's loop
                        asyncio.set_event_loop(loop)
                        self._async_milvus_client = AsyncMilvusClient(
                            **self._connection_args
                        )
                    else:
                        # Other cases, create directly
                        self._async_milvus_client = AsyncMilvusClient(
                            **self._connection_args
                        )
                except RuntimeError:
                    # No event loop, create a new one
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    self._async_milvus_client = AsyncMilvusClient(
                        **self._connection_args
                    )

            except Exception as e:
                # If still fails, raise clear error message
                raise RuntimeError(
                    f"Failed to create AsyncMilvusClient: {e}. "
                    f"This usually happens in multithreading environments "
                    f"(like Streamlit) where event loops are not properly "
                    f"configured. To fix this issue, you can: "
                    f"1. Use only synchronous methods (avoid methods "
                    f"starting with 'a') "
                    f"2. Create the Milvus instance in the main thread "
                    f"3. Or ensure proper asyncio event loop setup in your "
                    f"environment."
                ) from e

        return self._async_milvus_client

    @property
    def vector_fields(self) -> List[str]:
        """Get vector field(s)."""
        return self._as_list(self._vector_field)

    @property
    def _is_multi_vector(self) -> bool:
        """Whether the sum of embedding functions and builtin functions is multi."""
        return isinstance(self._vector_field, list) and len(self._vector_field) > 1

    @property
    def _is_multi_embedding(self) -> bool:
        """Whether there are multi embedding functions in this instance."""
        return isinstance(self.embedding_func, list) and len(self.embedding_func) > 1

    @property
    def _is_multi_function(self) -> bool:
        """Whether there are multi builtin functions in this instance."""
        return isinstance(self.builtin_func, list) and len(self.builtin_func) > 1

    @property
    def _is_embedding_only(self) -> bool:
        """Whether there are only embedding function(s) but no builtin function(s)."""
        return (
            len(self._as_list(self.embedding_func)) > 0  # type: ignore[arg-type]
            and len(self._as_list(self.builtin_func)) == 0
        )

    @property
    def _is_function_only(self) -> bool:
        """Whether there are only builtin function(s) but no embedding function(s)."""
        return (
            len(self._as_list(self.embedding_func)) == 0  # type: ignore[arg-type]
            and len(self._as_list(self.builtin_func)) > 0
        )

    @property
    def _is_sparse(self) -> bool:
        """Detect whether there is only one sparse embedding/builtin function"""
        if self._is_embedding_only:
            embedding_func = self._as_list(self.embedding_func)  # type: ignore[arg-type]
            if len(embedding_func) == 1 and self._is_sparse_embedding(
                embedding_func[0]  # type: ignore[arg-type]
            ):
                return True
        if self._is_function_only:
            builtin_func = self._as_list(self.builtin_func)
            if len(builtin_func) == 1 and isinstance(
                builtin_func[0], BM25BuiltInFunction
            ):
                return True
        return False

    @staticmethod
    def _is_sparse_embedding(embeddings_function: EmbeddingType) -> bool:
        return isinstance(embeddings_function, BaseSparseEmbedding)

    def _init(
        self,
        embeddings: Optional[List[list]] = None,
        metadatas: Optional[list[dict]] = None,
        partition_names: Optional[list] = None,
        replica_number: int = 1,
        timeout: Optional[float] = None,
    ) -> None:
        if embeddings is not None:
            self._create_collection(embeddings, metadatas)
        self._extract_fields()
        self._create_index()
        self._create_search_params()
        self._load(
            partition_names=partition_names,
            replica_number=replica_number,
            timeout=timeout,
        )

    def _create_collection(
        self, embeddings: List[list], metadatas: Optional[list[dict]] = None
    ) -> None:
        metadata_fields = self._prepare_metadata_fields(metadatas)
        text_fields = self._prepare_text_fields()
        primary_key_fields = self._prepare_primary_key_fields()
        vector_fields = self._prepare_vector_fields(embeddings)

        fields = text_fields + primary_key_fields + vector_fields + metadata_fields

        # Create the schema for the collection
        schema = CollectionSchema(
            fields,
            description=self.collection_description,
            partition_key_field=self._partition_key_field,
            enable_dynamic_field=self.enable_dynamic_field,
            functions=[func.function for func in self._as_list(self.builtin_func)],
        )

        # Create the collection
        try:
            if self.num_shards is not None:
                # Issue with defaults:
                # https://github.com/milvus-io/pymilvus/blob/59bf5e811ad56e20946559317fed855330758d9c/pymilvus/client/prepare.py#L82-L85
                self.col = Collection(
                    name=self.collection_name,
                    schema=schema,
                    consistency_level=self.consistency_level,
                    using=self.alias,
                    num_shards=self.num_shards,
                    num_partitions=self.num_partitions,
                )
            else:
                self.col = Collection(
                    name=self.collection_name,
                    schema=schema,
                    consistency_level=self.consistency_level,
                    using=self.alias,
                    num_partitions=self.num_partitions,
                )
            # Set the collection properties if they exist
            if self.collection_properties is not None:
                self.col.set_properties(self.collection_properties)
        except MilvusException as e:
            logger.error(
                "Failed to create collection: %s error: %s", self.collection_name, e
            )
            raise e

    def _prepare_metadata_fields(
        self, metadatas: Optional[list[dict]] = None
    ) -> List[FieldSchema]:
        fields = []
        # If enable_dynamic_field, we don't need to create fields, and just pass it.
        if self.enable_dynamic_field:
            # If both dynamic fields and partition key field are enabled
            if self._partition_key_field is not None:
                # create the partition field
                fields.append(
                    FieldSchema(
                        self._partition_key_field, DataType.VARCHAR, max_length=65_535
                    )
                )
        elif self._metadata_field is not None:
            fields.append(FieldSchema(self._metadata_field, DataType.JSON))
        else:
            # Determine metadata schema
            if metadatas:
                # Create FieldSchema for each entry in metadata.
                vector_fields: List[str] = self._as_list(self._vector_field)
                for key, value in metadatas[0].items():
                    # Check if the key is reserved
                    if (
                        key
                        in [
                            self._primary_field,
                            self._text_field,
                        ]
                        + vector_fields
                    ):
                        logger.error(
                            (
                                "Failure to create collection, "
                                "metadata key: %s is reserved."
                            ),
                            key,
                        )
                        raise ValueError(f"Metadata key {key} is reserved.")
                    # Infer the corresponding datatype of the metadata
                    if (
                        self.metadata_schema
                        and key in self.metadata_schema  # type: ignore
                        and "dtype" in self.metadata_schema[key]  # type: ignore
                    ):
                        fields.append(
                            self._get_field_schema_from_dict(
                                key, self.metadata_schema[key]
                            )
                        )
                    else:
                        dtype = infer_dtype_bydata(value)
                        # Datatype isn't compatible
                        if dtype == DataType.UNKNOWN or dtype == DataType.NONE:
                            logger.error(
                                (
                                    "Failure to create collection, "
                                    "unrecognized dtype for key: %s"
                                ),
                                key,
                            )
                            raise ValueError(f"Unrecognized datatype for {key}.")
                        # Datatype is a string/varchar equivalent
                        elif dtype == DataType.VARCHAR:
                            kwargs = {}
                            for function in self._as_list(self.builtin_func):
                                if isinstance(function, BM25BuiltInFunction):
                                    if function.input_field_names == self._text_field:
                                        kwargs = (
                                            function.get_input_field_schema_kwargs()
                                        )
                                        break

                            fields.append(
                                FieldSchema(
                                    key, DataType.VARCHAR, max_length=65_535, **kwargs
                                )
                            )
                        # infer_dtype_bydata currently can't recognize array type,
                        # so this line can not be accessed.
                        # This line may need to be modified in the future when
                        # infer_dtype_bydata can recognize array type.
                        # https://github.com/milvus-io/pymilvus/issues/2165
                        elif dtype == DataType.ARRAY:
                            kwargs = self.metadata_schema[key]["kwargs"]  # type: ignore
                            fields.append(
                                FieldSchema(name=key, dtype=DataType.ARRAY, **kwargs)
                            )
                        else:
                            fields.append(FieldSchema(key, dtype))
        return fields

    def _prepare_text_fields(self) -> List[FieldSchema]:
        fields = []
        kwargs = {}
        for function in self._as_list(self.builtin_func):
            if isinstance(function, BM25BuiltInFunction):
                if self._from_list(function.input_field_names) == self._text_field:
                    kwargs = function.get_input_field_schema_kwargs()
                    break

        fields.append(
            FieldSchema(self._text_field, DataType.VARCHAR, max_length=65_535, **kwargs)
        )
        return fields

    def _prepare_primary_key_fields(self) -> List[FieldSchema]:
        fields = []
        if self.auto_id:
            fields.append(
                FieldSchema(
                    self._primary_field, DataType.INT64, is_primary=True, auto_id=True
                )
            )
        else:
            fields.append(
                FieldSchema(
                    self._primary_field,
                    DataType.VARCHAR,
                    is_primary=True,
                    auto_id=False,
                    max_length=65_535,
                )
            )
        return fields

    def _prepare_vector_fields(self, embeddings: List[list]) -> List[FieldSchema]:
        fields = []
        embeddings_functions: List[EmbeddingType] = self._as_list(self.embedding_func)

        assert (
            len(self._vector_fields_from_embedding)
            == len(embeddings_functions)
            == len(embeddings)
        ), (
            "The number of `self._vector_fields_from_embedding`, "
            "`embeddings_functions`, and `embeddings` should be the same."
            f"Got {len(self._vector_fields_from_embedding)}, "
            f"{len(embeddings_functions)}, and {len(embeddings)}."
        )
        # Loop through the embedding functions
        for vector_field, embedding_func, embedding in zip(
            self._vector_fields_from_embedding, embeddings_functions, embeddings
        ):
            vector_schema = self._vector_schema_map.get(vector_field, None)
            if vector_schema and "dtype" in vector_schema:
                fields.append(
                    self._get_field_schema_from_dict(vector_field, vector_schema)
                )
            else:
                if self._is_sparse_embedding(embedding_func):
                    fields.append(
                        FieldSchema(vector_field, DataType.SPARSE_FLOAT_VECTOR)
                    )
                else:
                    # Supports binary or float vectors
                    fields.append(
                        FieldSchema(
                            vector_field,
                            infer_dtype_bydata(embedding[0]),
                            dim=len(embedding[0]),
                        )
                    )
        # Loop through the built-in functions
        for vector_field, builtin_function in zip(
            self._vector_fields_from_function, self._as_list(self.builtin_func)
        ):
            vector_schema = self._vector_schema_map.get(vector_field, None)
            if vector_schema and "dtype" in vector_schema:
                field = self._get_field_schema_from_dict(vector_field, vector_schema)
            elif isinstance(builtin_function, BM25BuiltInFunction):
                field = FieldSchema(vector_field, DataType.SPARSE_FLOAT_VECTOR)
            else:
                raise ValueError(
                    "Unsupported embedding function type: "
                    f"{type(builtin_function)} for field: {vector_field}."
                )
            field.is_function_output = True
            fields.append(field)
        return fields

    def _get_field_schema_from_dict(
        self, field_name: str, schema_dict: dict
    ) -> FieldSchema:
        assert "dtype" in schema_dict, (
            f"Please provide `dtype` in the schema dict. "
            f"Existing keys are: {schema_dict.keys()}"
        )
        dtype = schema_dict.pop("dtype")
        kwargs = schema_dict.pop("kwargs", {})
        kwargs.update(schema_dict)
        return FieldSchema(name=field_name, dtype=dtype, **kwargs)

    def _extract_fields(self) -> None:
        """Grab the existing fields from the Collection"""
        if isinstance(self.col, Collection):
            schema = self.col.schema
            for x in schema.fields:
                self.fields.append(x.name)

    def _get_index(self, field_name: Optional[str] = None) -> Optional[dict[str, Any]]:
        """Return the vector index information if it exists"""
        if not self._is_multi_vector:
            field_name: str = field_name or self._vector_field  # type: ignore

        if isinstance(self.col, Collection):
            for x in self.col.indexes:
                if x.field_name == field_name:
                    return x.to_dict()
        return None

    def _get_indexes(
        self, field_names: Optional[List[str]] = None
    ) -> List[dict[str, Any]]:
        """Return the list of vector index information"""
        index_list = []
        if not field_names:
            field_names = self._as_list(self._vector_field)
        for field_name in field_names:
            index = self._get_index(field_name)
            if index is not None:
                index_list.append(index)
        return index_list

    def _create_index(self) -> None:
        """Create an index on the collection"""
        if isinstance(self.col, Collection):
            embeddings_functions: List[EmbeddingType] = self._as_list(
                self.embedding_func
            )

            default_index_params = {
                "metric_type": "L2",
                "index_type": "AUTOINDEX",
                "params": {},
            }
            for vector_field, embeddings_func in zip(
                self._vector_fields_from_embedding, embeddings_functions
            ):
                if not self._get_index(vector_field):
                    try:
                        if not self._index_param_map.get(vector_field, None):
                            if self._is_sparse_embedding(embeddings_func):
                                index_params = {
                                    "metric_type": "IP",
                                    "index_type": "SPARSE_INVERTED_INDEX",
                                    "params": {"drop_ratio_build": 0.2},
                                }
                            else:
                                index_params = default_index_params
                            self._index_param_map[vector_field] = index_params
                        else:
                            index_params = self._index_param_map[vector_field]
                        self.col.create_index(
                            vector_field,
                            index_params=index_params,
                            using=self.alias,
                        )
                        logger.debug(
                            "Successfully created an index"
                            "on %s field on collection: %s",
                            vector_field,
                            self.collection_name,
                        )
                    except MilvusException as e:
                        logger.error(
                            "Failed to create an index on collection: %s",
                            self.collection_name,
                        )
                        raise e
            for vector_field, builtin_function in zip(
                self._vector_fields_from_function, self._as_list(self.builtin_func)
            ):
                if not self._get_index(vector_field):
                    try:
                        if not self._index_param_map.get(vector_field, None):
                            if builtin_function.type == FunctionType.BM25:
                                index_params = {
                                    "metric_type": "BM25",
                                    "index_type": "AUTOINDEX",
                                    "params": {},
                                }
                            else:
                                raise ValueError(
                                    "Unsupported built-in function type: "
                                    f"{builtin_function.type} for field: "
                                    f"{vector_field}."
                                )
                            self._index_param_map[vector_field] = index_params
                        else:
                            index_params = self._index_param_map[vector_field]
                        self.col.create_index(
                            vector_field,
                            index_params=index_params,
                            using=self.alias,
                        )
                        logger.debug(
                            "Successfully created an index"
                            "on %s field on collection: %s",
                            vector_field,
                            self.collection_name,
                        )
                    except MilvusException as e:
                        logger.error(
                            "Failed to create an index on collection: %s",
                            self.collection_name,
                        )
                        raise e
            index_params_list: List[dict] = []
            for field in self._as_list(self._vector_field):
                index_params_list.append(self._index_param_map.get(field, {}))
            self.index_params = self._from_list(index_params_list)

    def _create_search_params(self) -> None:
        """Generate search params based on the current index type"""
        import copy

        if isinstance(self.col, Collection) and self.search_params is None:
            vector_fields: List[str] = self._as_list(self._vector_field)
            search_params_list: List[dict] = []

            for vector_field in vector_fields:
                index = self._get_index(vector_field)
                if index is not None:
                    index_type: str = index["index_param"]["index_type"]
                    metric_type: str = index["index_param"]["metric_type"]
                    search_params = copy.deepcopy(
                        self.default_search_params[index_type]
                    )
                    search_params["metric_type"] = metric_type
                    search_params_list.append(search_params)
            self.search_params = self._from_list(search_params_list)

    def _load(
        self,
        partition_names: Optional[list] = None,
        replica_number: int = 1,
        timeout: Optional[float] = None,
    ) -> None:
        """Load the collection if available."""
        timeout = self.timeout or timeout
        if (
            isinstance(self.col, Collection)
            and self._get_indexes()
            and utility.load_state(self.collection_name, using=self.alias)
            == LoadState.NotLoad
        ):
            self.col.load(
                partition_names=partition_names,
                replica_number=replica_number,
                timeout=timeout,
            )

    def _prepare_insert_list(
        self,
        texts: List[str],
        embeddings: List[List[float]] | List[List[List[float]]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        force_ids: bool = False,
    ) -> list[dict]:
        """Prepare insert list for batch insertion.

        Args:
            texts: List of texts to insert
            embeddings: List of embeddings corresponding to the texts
            metadatas: Optional metadata for each text
            ids: Optional IDs for each text
            force_ids: If force_ids, when auto_id is True and ids is not None,
             it will return a list containing the customized ids, otherwise,
             it will not contain the customized ids.

        Returns:
            List of dictionaries ready for insertion
        """
        insert_list: list[dict] = []

        for vector_field_embeddings in embeddings:
            assert len(texts) == len(
                vector_field_embeddings
            ), "Mismatched lengths of texts and embeddings."

        if metadatas is not None:
            assert len(texts) == len(
                metadatas
            ), "Mismatched lengths of texts and metadatas."

        for i, text in zip(range(len(texts)), texts):
            entity_dict = {}
            metadata = metadatas[i] if metadatas else {}
            if not self.auto_id or force_ids:
                entity_dict[self._primary_field] = ids[i]  # type: ignore[index]

            entity_dict[self._text_field] = text

            for vector_field, vector_field_embeddings in zip(  # type: ignore
                self._vector_fields_from_embedding, embeddings
            ):
                entity_dict[vector_field] = vector_field_embeddings[i]

            if self._metadata_field and not self.enable_dynamic_field:
                entity_dict[self._metadata_field] = metadata
            else:
                for key, value in metadata.items():
                    # if not enable_dynamic_field, skip fields not in the collection.
                    if not self.enable_dynamic_field and key not in self.fields:
                        continue
                    # If enable_dynamic_field, all fields are allowed.
                    entity_dict[key] = value

            insert_list.append(entity_dict)

        return insert_list

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        timeout: Optional[float] = None,
        batch_size: int = 1000,
        *,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Insert text data into Milvus.

        Inserting data when the collection has not be made yet will result
        in creating a new Collection. The data of the first entity decides
        the schema of the new collection, the dim is extracted from the first
        embedding and the columns are decided by the first metadata dict.
        Metadata keys will need to be present for all inserted values. At
        the moment there is no None equivalent in Milvus.

        Args:
            texts (Iterable[str]): The texts to embed, it is assumed
                that they all fit in memory.
            metadatas (Optional[List[dict]]): Metadata dicts attached to each of
                the texts. Defaults to None.
            should be less than 65535 bytes. Required and work when auto_id is False.
            timeout (Optional[float]): Timeout for each batch insert. Defaults
                to None.
            batch_size (int, optional): Batch size to use for insertion.
                Defaults to 1000.
            ids (Optional[List[str]]): List of text ids. The length of each item

        Raises:
            MilvusException: Failure to add texts

        Returns:
            List[str]: The resulting keys for each inserted element.
        """
        texts = list(texts)
        if not self.auto_id and ids is None:
            warnings.warn(
                "No ids provided and auto_id is False. "
                "Setting auto_id to True automatically.",
                UserWarning,
            )
            self.auto_id = True
        elif not self.auto_id and ids:  # Check ids
            assert len(set(ids)) == len(
                texts
            ), "Different lengths of texts and unique ids are provided."
            assert all(isinstance(x, str) for x in ids), "All ids should be strings."
            assert all(
                len(x.encode()) <= 65_535 for x in ids
            ), "Each id should be a string less than 65535 bytes."

        elif self.auto_id and ids:
            logger.warning(
                "The ids parameter is ignored when auto_id is True. "
                "The ids will be generated automatically."
            )

        embeddings_functions: List[EmbeddingType] = self._as_list(self.embedding_func)
        embeddings: List = []

        for embedding_func in embeddings_functions:
            try:
                embeddings.append(embedding_func.embed_documents(texts))
            except NotImplementedError:
                embeddings.append([embedding_func.embed_query(x) for x in texts])
        # Currently, it is field-wise
        # assuming [f1, f2] embeddings functions and [a, b, c] as texts:
        # embeddings = [
        #     [f1(a), f1(b), f1(c)],
        #     [f2(a), f2(b), f2(c)]
        # ]
        # or
        # embeddings = [
        #     [f1(a), f1(b), f1(c)]
        # ]

        if len(texts) == 0:
            logger.debug("Nothing to insert, skipping.")
            return []

        # Transpose it into row-wise
        if self._is_multi_embedding:
            # transposed_embeddings = [
            #     [f1(a), f2(a)],
            #     [f1(b), f2(b)],
            #     [f1(c), f2(c)]
            # ]
            transposed_embeddings = [
                [embeddings[j][i] for j in range(len(embeddings))]
                for i in range(len(embeddings[0]))
            ]
        else:
            # transposed_embeddings = [
            #     f1(a),
            #     f1(b),
            #     f1(c)
            # ]
            transposed_embeddings = embeddings[0] if len(embeddings) > 0 else []

        return self.add_embeddings(
            texts=texts,
            embeddings=transposed_embeddings,
            metadatas=metadatas,
            timeout=timeout,
            batch_size=batch_size,
            ids=ids,
            **kwargs,
        )

    def add_embeddings(
        self,
        texts: List[str],
        embeddings: List[List[float]] | List[List[List[float]]],
        metadatas: Optional[List[dict]] = None,
        timeout: Optional[float] = None,
        batch_size: int = 1000,
        *,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Insert text data with embeddings vectors into Milvus.

        This method inserts a batch of text embeddings into a Milvus collection.
        If the collection is not initialized, it will automatically initialize
        the collection based on the embeddings,metadatas, and other parameters.
        The embeddings are expected to be pre-generated using compatible embedding
        functions, and the metadata associated with each text is optional but
        must match the number of texts.

        Args:
            texts (List[str]): the texts to insert
            embeddings (List[List[float]] | List[List[List[float]]]):
                A vector embeddings for each text (in case of a single vector)
                or list of vectors for each text (in case of multi-vector)
            metadatas (Optional[List[dict]]): Metadata dicts attached to each of
                the texts. Defaults to None.
            should be less than 65535 bytes. Required and work when auto_id is False.
            timeout (Optional[float]): Timeout for each batch insert. Defaults
                to None.
            batch_size (int, optional): Batch size to use for insertion.
                Defaults to 1000.
            ids (Optional[List[str]]): List of text ids. The length of each item

        Raises:
            MilvusException: Failure to add texts and embeddings

        Returns:
            List[str]: The resulting keys for each inserted element.
        """

        if embeddings:
            # row-wise -> field-wise
            if not self._is_multi_embedding:
                embeddings = [[embedding] for embedding in embeddings]  # type: ignore
            # transposed_embeddings = [
            #     [f1(a), f2(a)],
            #     [f1(b), f2(b)],
            #     [f1(c), f2(c)]
            # ]
            # Transpose embeddings to make it a list of embeddings of each type.
            embeddings = [  # type: ignore
                [embeddings[j][i] for j in range(len(embeddings))]
                for i in range(len(embeddings[0]))
            ]
            # embeddings = [
            #     [f1(a), f1(b), f1(c)],
            #     [f2(a), f2(b), f2(c)]
            # ]

        # If the collection hasn't been initialized yet, perform all steps to do so
        if not isinstance(self.col, Collection):
            kwargs = {"embeddings": embeddings, "metadatas": metadatas}
            if self.partition_names:
                kwargs["partition_names"] = self.partition_names
            if self.replica_number:
                kwargs["replica_number"] = self.replica_number
            if self.timeout:
                kwargs["timeout"] = self.timeout
            self._init(**kwargs)

        insert_list = self._prepare_insert_list(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

        # Total insert count
        total_count = len(insert_list)

        pks: list[str] = []

        assert isinstance(self.col, Collection)
        for i in range(0, total_count, batch_size):
            # Grab end index
            end = min(i + batch_size, total_count)
            batch_insert_list = insert_list[i:end]
            # Insert into the collection.
            try:
                timeout = self.timeout or timeout
                res = self.client.insert(
                    self.collection_name,
                    batch_insert_list,
                    timeout=timeout,
                    **kwargs,
                )
                pks.extend(res["ids"])
            except MilvusException as e:
                self._handle_batch_operation_exception(
                    e, batch_insert_list, i, total_count, "insert"
                )
        return pks

    def _handle_batch_operation_exception(
        self,
        e: MilvusException,
        batch_list: list[dict],
        batch_index: int,
        total_count: int,
        operation_name: str,
    ) -> None:
        """Handle batch operation exceptions with detailed logging.

        Args:
            e: The MilvusException that occurred
            batch_list: The batch list that caused the exception
            batch_index: Current batch index (0-based)
            total_count: Total number of entities
            operation_name: Name of the operation (e.g., "insert", "upsert")

        Raises:
            MilvusException: Re-raises the original exception after logging
        """
        first_entity = {}
        if batch_list:
            first_entity = batch_list[0]
        log_entity = {}
        for k, v in first_entity.items():
            if isinstance(v, list) and len(v) > 10:
                log_entity[k] = f"{v[:10]}... (truncated, total len: {len(v)})"
            else:
                log_entity[k] = v
        logger.error(
            "Failed to %s batch starting at entity: %s/%s. " "First entity data: %s",
            operation_name,
            batch_index + 1,
            total_count,
            log_entity,
        )
        raise e

    def _collection_search(
        self,
        embedding_or_text: List[float] | Dict[int, float] | str,
        k: int = 4,
        param: Optional[dict] = None,
        expr: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> Optional[List[List[dict]]]:
        """Perform a search on an embedding or a query text and return milvus search
        results.

        For more information about the search parameters, take a look at the pymilvus
        documentation found here:
        https://milvus.io/api-reference/pymilvus/v2.5.x/ORM/Collection/search.md

        Args:
            embedding_or_text (List[float] | Dict[int, float] | str): The embedding
                vector or query text being searched.
            k (int, optional): The amount of results to return. Defaults to 4.
            param (dict): The search params for the specified index.
                Defaults to None.
            expr (str, optional): Filtering expression. Defaults to None.
            timeout (float, optional): How long to wait before timeout error.
                Defaults to None.
            kwargs: Collection.search() keyword arguments.

        Returns:
            List[List[dict]]: Milvus search result.
        """
        if self.col is None:
            logger.debug("No existing collection to search.")
            return None

        assert not self._is_multi_vector, (
            "_collection_search does not support multi-vector search. "
            "You can use _collection_hybrid_search instead."
        )

        if param is None:
            assert len(self._as_list(self.search_params)) == 1, (
                "The number of search params is larger than 1, "
                "please check the search_params in this Milvus instance."
            )
            param = self._as_list(self.search_params)[0]

        if self.enable_dynamic_field:
            output_fields = ["*"]
        else:
            output_fields = self._remove_forbidden_fields(self.fields[:])
        col_search_res = self.client.search(
            self.collection_name,
            data=[embedding_or_text],
            anns_field=self._vector_field,
            search_params=param,
            limit=k,
            filter=expr,
            output_fields=output_fields,
            timeout=self.timeout or timeout,
            **kwargs,
        )
        return col_search_res

    def _collection_hybrid_search(
        self,
        query: str,
        k: int = 4,
        param: Optional[dict | list[dict]] = None,
        expr: Optional[str] = None,
        fetch_k: Optional[int] = 4,
        ranker_type: Optional[Literal["rrf", "weighted"]] = None,
        ranker_params: Optional[dict] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> Optional[List[List[dict]]]:
        """
        Perform a hybrid search on a query string and return milvus search results.

        For more information about the search parameters, take a look at the pymilvus
        documentation found here:
        https://milvus.io/api-reference/pymilvus/v2.5.x/ORM/Collection/hybrid_search.md

        Args:
            query (str): The text being searched.
            k (int, optional): The amount of results to return. Defaults to 4.
            param (dict | list[dict], optional): The search params for the specified
                index. Defaults to None.
            expr (str, optional): Filtering expression. Defaults to None.
            fetch_k (int, optional): The amount of pre-fetching results for each query.
                Defaults to 4.
            ranker_type (str, optional): The type of ranker to use. Defaults to None.
            ranker_params (dict, optional): The parameters for the ranker.
                Defaults to None.
            timeout (float, optional): How long to wait before timeout error.
                Defaults to None.
            kwargs: Collection.hybrid_search() keyword arguments.

        Returns:
            List[List[dict]]: Milvus search result.
        """
        if self.col is None:
            logger.debug("No existing collection to search.")
            return None

        search_requests = []
        reranker = self._create_ranker(
            ranker_type=ranker_type,
            ranker_params=ranker_params or {},
        )
        if not param:
            param_list = self._as_list(self.search_params)
        else:
            assert len(self._as_list(param)) == len(
                self._as_list(self.search_params)
            ), (
                f"The number of search params ({len(self._as_list(param))})"
                f" does not match the number of vector fields "
                f"({len(self._as_list(self._vector_field))})."
                f" All vector fields are: {(self._as_list(self._vector_field))},"
                " please provide a list of search params for each vector field."
            )
            param_list = self._as_list(param)
        for field, param_dict in zip(self._vector_field, param_list):
            search_data: List[float] | Dict[int, float] | str
            if field in self._vector_fields_from_embedding:
                embedding_func: EmbeddingType = self._as_list(self.embedding_func)[  # type: ignore
                    self._vector_fields_from_embedding.index(field)
                ]
                search_data = embedding_func.embed_query(query)
            else:
                search_data = query
            request = AnnSearchRequest(
                data=[search_data],
                anns_field=field,
                param=param_dict,
                limit=fetch_k,
                expr=expr,
            )
            search_requests.append(request)
        if self.enable_dynamic_field:
            output_fields = ["*"]
        else:
            output_fields = self._remove_forbidden_fields(self.fields[:])
        col_search_res = self.client.hybrid_search(
            self.collection_name,
            reqs=search_requests,
            ranker=reranker,
            limit=k,
            output_fields=output_fields,
            timeout=self.timeout or timeout,
            **kwargs,
        )
        return col_search_res

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        param: Optional[dict | list[dict]] = None,
        expr: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Perform a similarity search against the query string.

        Args:
            query (str): The text to search.
            k (int, optional): How many results to return. Defaults to 4.
            param (dict | list[dict], optional): The search params for the index type.
                Defaults to None.
            expr (str, optional): Filtering expression. Defaults to None.
            timeout (int, optional): How long to wait before timeout error.
                Defaults to None.
            kwargs: Collection.search() keyword arguments.

        Returns:
            List[Document]: Document results for search.
        """
        if self.col is None:
            logger.debug("No existing collection to search.")
            return []
        timeout = self.timeout or timeout
        res = self.similarity_search_with_score(
            query=query, k=k, param=param, expr=expr, timeout=timeout, **kwargs
        )
        return [doc for doc, _ in res]

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        param: Optional[dict] = None,
        expr: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Perform a similarity search against the query string.

        Args:
            embedding (List[float]): The embedding vector to search.
            k (int, optional): How many results to return. Defaults to 4.
            param (dict, optional): The search params for the index type.
                Defaults to None.
            expr (str, optional): Filtering expression. Defaults to None.
            timeout (int, optional): How long to wait before timeout error.
                Defaults to None.
            kwargs: Collection.search() keyword arguments.

        Returns:
            List[Document]: Document results for search.
        """
        if self.col is None:
            logger.debug("No existing collection to search.")
            return []
        timeout = self.timeout or timeout
        res = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, param=param, expr=expr, timeout=timeout, **kwargs
        )
        return [doc for doc, _ in res]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        param: Optional[dict | list[dict]] = None,
        expr: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Perform a search on a query string and return results with score.

        For more information about the search parameters, take a look at the pymilvus
        documentation found here:
        https://milvus.io/api-reference/pymilvus/v2.5.x/ORM/Collection/search.md

        Args:
            query (str): The text being searched.
            k (int, optional): The amount of results to return. Defaults to 4.
            param (dict | list[dict], optional): The search params for the specified
            index. Defaults to None.
            expr (str, optional): Filtering expression. Defaults to None.
            timeout (float, optional): How long to wait before timeout error.
                Defaults to None.
            kwargs: Collection.search() or hybrid_search() keyword arguments.

        Returns:
            List[Tuple[Document, float]]: List of result doc and score.
        """
        if self.col is None:
            logger.debug("No existing collection to search.")
            return []

        if self._is_multi_vector:
            col_search_res = self._collection_hybrid_search(
                query=query, k=k, param=param, expr=expr, timeout=timeout, **kwargs
            )

        else:
            assert len(self._as_list(param)) <= 1, (
                "When there is only one vector field, you can not provide multiple "
                "search param dicts."
            )
            param = cast(Optional[dict], self._from_list(param))
            if (
                len(self._as_list(self.embedding_func)) == 1  # type: ignore[arg-type]
                and len(self._as_list(self.builtin_func)) == 0
            ):
                embedding = self._as_list(self.embedding_func)[0].embed_query(query)  # type: ignore
                col_search_res = self._collection_search(
                    embedding_or_text=embedding,
                    k=k,
                    param=param,
                    expr=expr,
                    timeout=timeout,
                    **kwargs,
                )
            elif (
                len(self._as_list(self.embedding_func)) == 0  # type: ignore[arg-type]
                and len(self._as_list(self.builtin_func)) == 1
            ):
                col_search_res = self._collection_search(
                    embedding_or_text=query,
                    k=k,
                    param=param,
                    expr=expr,
                    timeout=timeout,
                    **kwargs,
                )
            else:
                raise RuntimeError(
                    "Check either it's multi vectors or single vector with "
                    "only one embedding/builtin function."
                )

        return self._parse_documents_from_search_results(col_search_res)

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float] | Dict[int, float],
        k: int = 4,
        param: Optional[dict] = None,
        expr: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Perform a search on an embedding and return results with score.

        For more information about the search parameters, take a look at the pymilvus
        documentation found here:
        https://milvus.io/api-reference/pymilvus/v2.5.x/ORM/Collection/search.md

        Args:
            embedding (List[float] | Dict[int, float]): The embedding vector being
                searched.
            k (int, optional): The amount of results to return. Defaults to 4.
            param (dict): The search params for the specified index.
                Defaults to None.
            expr (str, optional): Filtering expression. Defaults to None.
            timeout (float, optional): How long to wait before timeout error.
                Defaults to None.
            kwargs: Collection.search() keyword arguments.

        Returns:
            List[Tuple[Document, float]]: Result doc and score.
        """
        col_search_res = self._collection_search(
            embedding_or_text=embedding,
            k=k,
            param=param,
            expr=expr,
            timeout=timeout,
            **kwargs,
        )
        return self._parse_documents_from_search_results(col_search_res)

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        param: Optional[dict] = None,
        expr: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Perform a search and return results that are reordered by MMR.

        Args:
            query (str): The text being searched.
            k (int, optional): How many results to give. Defaults to 4.
            fetch_k (int, optional): Total results to select k from.
                Defaults to 20.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5
            param (dict, optional): The search params for the specified index.
                Defaults to None.
            expr (str, optional): Filtering expression. Defaults to None.
            timeout (float, optional): How long to wait before timeout error.
                Defaults to None.
            kwargs: Collection.search() keyword arguments.


        Returns:
            List[Document]: Document results for search.
        """
        if self.col is None:
            logger.debug("No existing collection to search.")
            return []

        assert (
            len(self._as_list(self.embedding_func)) == 1  # type: ignore[arg-type]
        ), "You must set only one embedding function for MMR search."
        if len(self._vector_fields_from_function) > 0:
            logger.warning(
                "MMR search will only use the embedding function, "
                "without the built-in functions."
            )

        embedding = self._as_list(self.embedding_func)[0].embed_query(query)  # type: ignore
        timeout = self.timeout or timeout
        return self.max_marginal_relevance_search_by_vector(
            embedding=embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            param=param,
            expr=expr,
            timeout=timeout,
            **kwargs,
        )

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: list[float] | dict[int, float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        param: Optional[dict] = None,
        expr: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Perform a search and return results that are reordered by MMR.

        Args:
            embedding (list[float] | dict[int, float]): The embedding vector being
                searched.
            k (int, optional): How many results to give. Defaults to 4.
            fetch_k (int, optional): Total results to select k from.
                Defaults to 20.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5
            param (dict, optional): The search params for the specified index.
                Defaults to None.
            expr (str, optional): Filtering expression. Defaults to None.
            timeout (float, optional): How long to wait before timeout error.
                Defaults to None.
            kwargs: Collection.search() keyword arguments.

        Returns:
            List[Document]: Document results for search.
        """
        col_search_res = self._collection_search(
            embedding_or_text=embedding,
            k=fetch_k,
            param=param,
            expr=expr,
            timeout=timeout,
            **kwargs,
        )
        if col_search_res is None:
            return []
        ids = []
        documents = []
        scores = []
        for result in col_search_res[0]:
            doc = self._parse_document(result["entity"])
            documents.append(doc)
            scores.append(result["distance"])
            ids.append(result.get(self._primary_field, "id"))

        vectors = self.client.query(
            self.collection_name,
            filter=f"{self._primary_field} in {ids}",
            output_fields=[self._primary_field, self._vector_field],
            timeout=timeout,
        )
        # Reorganize the results from query to match search order.
        vectors = {x[self._primary_field]: x[self._vector_field] for x in vectors}

        ordered_result_embeddings = [vectors[x] for x in ids]

        # Get the new order of results.
        new_ordering = maximal_marginal_relevance(
            np.array(embedding), ordered_result_embeddings, k=k, lambda_mult=lambda_mult
        )

        # Reorder the values and return.
        ret = []
        for x in new_ordering:
            # Function can return -1 index
            if x == -1:
                break
            else:
                ret.append(documents[x])
        return ret

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """
        The 'correct' relevance function
        may differ depending on a few things, including:
        - the distance / similarity metric used by the VectorStore
        - the scale of your embeddings (OpenAI's are unit normed. Many others are not!)
        - embedding dimensionality
        - etc.

        """
        if not self.col or not self.col.indexes:
            raise ValueError(
                "No index params provided. Could not determine relevance function."
            )
        if self._is_multi_embedding or self._is_multi_function:
            raise ValueError(
                "No supported normalization function for multi vectors. "
                "Could not determine relevance function."
            )
        if self._is_sparse:
            raise ValueError(
                "No supported normalization function for sparse indexes. "
                "Could not determine relevance function."
            )

        def _map_l2_to_similarity(l2_distance: float) -> float:
            """Return a similarity score on a scale [0, 1].
            It is recommended that the original vector is normalized,
            Milvus only calculates the value before applying square root.
            l2_distance range: (0 is most similar, 4 most dissimilar)
            See
            https://milvus.io/docs/metric.md?tab=floating#Euclidean-distance-L2
            """
            return 1 - l2_distance / 4.0

        def _map_ip_to_similarity(ip_score: float) -> float:
            """Return a similarity score on a scale [0, 1].
            It is recommended that the original vector is normalized,
            ip_score range: (1 is most similar, -1 most dissimilar)
            See
            https://milvus.io/docs/metric.md?tab=floating#Inner-product-IP
            https://milvus.io/docs/metric.md?tab=floating#Cosine-Similarity
            """
            return (ip_score + 1) / 2.0

        if not self.index_params:
            logger.warning(
                "No index params provided. Could not determine relevance function. "
                "Use L2 distance as default."
            )
            return _map_l2_to_similarity
        indexes_params = self._as_list(self.index_params)
        if len(indexes_params) > 1:
            raise ValueError(
                "No supported normalization function for multi vectors. "
                "Could not determine relevance function."
            )
        # In the left case, the len of indexes_params is 1.
        metric_type = indexes_params[0]["metric_type"]
        if metric_type == "L2":
            return _map_l2_to_similarity
        elif metric_type in ["IP", "COSINE"]:
            return _map_ip_to_similarity
        else:
            raise ValueError(
                "No supported normalization function"
                f" for metric type: {metric_type}."
            )

    def delete(
        self, ids: Optional[List[str]] = None, expr: Optional[str] = None, **kwargs: str
    ) -> Optional[bool]:
        """Delete by vector ID or boolean expression.
        Refer to [Milvus documentation](https://milvus.io/docs/delete_data.md)
        for notes and examples of expressions.

        Args:
            ids: List of ids to delete.
            expr: Boolean expression that specifies the entities to delete.
            kwargs: Other parameters in Milvus delete api.

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise.
        """
        if isinstance(ids, list) and len(ids) > 0:
            if expr is not None:
                logger.warning(
                    "Both ids and expr are provided. " "Ignore expr and delete by ids."
                )
            expr = f"{self._primary_field} in {ids}"
        else:
            assert isinstance(
                expr, str
            ), "Either ids list or expr string must be provided."
        try:
            self.client.delete(self.collection_name, filter=expr, **kwargs)
            return True
        except MilvusException as e:
            logger.error(
                "Failed to delete entities: %s error: %s", self.collection_name, e
            )
            return False

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Optional[Union[EmbeddingType, List[EmbeddingType]]],
        metadatas: Optional[List[dict]] = None,
        collection_name: str = "LangChainCollection",
        connection_args: Optional[Dict[str, Any]] = None,
        consistency_level: str = "Session",
        index_params: Optional[Union[dict, List[dict]]] = None,
        search_params: Optional[Union[dict, List[dict]]] = None,
        drop_old: bool = False,
        *,
        ids: Optional[List[str]] = None,
        auto_id: bool = False,
        builtin_function: Optional[
            Union[BaseMilvusBuiltInFunction, List[BaseMilvusBuiltInFunction]]
        ] = None,
        **kwargs: Any,
    ) -> Milvus:
        """Create a Milvus collection, indexes it with HNSW, and insert data.

        Args:
            texts (List[str]): Text data.
            embedding (Optional[Union[Embeddings, BaseSparseEmbedding]]): Embedding
                function.
            metadatas (Optional[List[dict]]): Metadata for each text if it exists.
                Defaults to None.
            collection_name (str, optional): Collection name to use. Defaults to
                "LangChainCollection".
            connection_args (dict[str, Any], optional): Connection args to use. Defaults
                to DEFAULT_MILVUS_CONNECTION.
            consistency_level (str, optional): Which consistency level to use. Defaults
                to "Session".
            index_params (Optional[dict], optional): Which index_params to use. Defaults
                to None.
            search_params (Optional[dict], optional): Which search params to use.
                Defaults to None.
            drop_old (Optional[bool], optional): Whether to drop the collection with
                that name if it exists. Defaults to False.
            ids (Optional[List[str]]): List of text ids. Defaults to None.
            auto_id (bool): Whether to enable auto id for primary key. Defaults to
                False. If False, you need to provide text ids (string less than 65535
                bytes). If True, Milvus will generate unique integers as primary keys.
            builtin_function (Optional[Union[BaseMilvusBuiltInFunction,
                List[BaseMilvusBuiltInFunction]]]):
                Built-in function to use. Defaults to None.
            **kwargs: Other parameters in Milvus Collection.
        Returns:
            Milvus: Milvus Vector Store
        """
        if isinstance(ids, list) and len(ids) > 0:
            if auto_id:
                logger.warning(
                    "Both ids and auto_id are provided. " "Ignore auto_id and use ids."
                )
            auto_id = False
        else:
            auto_id = True

        vector_db = cls(
            embedding_function=embedding,
            collection_name=collection_name,
            connection_args=connection_args,
            consistency_level=consistency_level,
            index_params=index_params,
            search_params=search_params,
            drop_old=drop_old,
            auto_id=auto_id,
            builtin_function=builtin_function,
            **kwargs,
        )
        vector_db.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        return vector_db

    def _parse_document(self, data: dict) -> Document:
        vector_fields: List[str] = self._as_list(self._vector_field)
        for vector_field in vector_fields:
            if vector_field in data:
                data.pop(vector_field)
        return Document(
            page_content=data.pop(self._text_field),
            metadata=data.pop(self._metadata_field) if self._metadata_field else data,
        )

    def _parse_documents_from_search_results(
        self,
        col_search_res: Optional[List[List[dict]]],
    ) -> List[Tuple[Document, float]]:
        if not col_search_res:
            return []
        ret = []
        for result in col_search_res[0]:
            doc = self._parse_document(result["entity"])
            pair = (doc, result["distance"])
            ret.append(pair)
        return ret

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """Run more documents through the embeddings and add to the vectorstore.

        Args:
            documents: Documents to add to the vectorstore.

        Returns:
            List of IDs of the added texts.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return self.add_texts(texts, metadatas, **kwargs)

    def get_pks(self, expr: str, **kwargs: Any) -> List[int] | None:
        """Get primary keys with expression

        Args:
            expr: Expression - E.g: "id in [1, 2]", or "title LIKE 'Abc%'"

        Returns:
            List[int]: List of IDs (Primary Keys)
        """

        if self.col is None:
            logger.debug("No existing collection to get pk.")
            return None

        try:
            query_result = self.client.query(
                self.collection_name,
                filter=expr,
                output_fields=[self._primary_field],
            )
        except MilvusException as exc:
            logger.error("Failed to get ids: %s error: %s", self.collection_name, exc)
            raise exc
        pks = [item.get(self._primary_field) for item in query_result]
        return pks

    def upsert(  # type: ignore
        self,
        ids: Optional[List[str]] = None,
        documents: List[Document] | None = None,
        batch_size: int = 1000,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Update/Insert documents to the vectorstore.

        Args:
            ids: IDs to update - Let's call get_pks to get ids with expression \n
            documents (List[Document]): Documents to add to the vectorstore.

        """

        if documents is None or len(documents) == 0:
            logger.debug("No documents to upsert.")
            return

        if not ids:
            self.add_documents(documents=documents, **kwargs)
        else:
            assert len(set(ids)) == len(
                documents
            ), "Different lengths of documents and unique ids are provided."

        embeddings_functions: List[EmbeddingType] = self._as_list(self.embedding_func)
        embeddings: List = []
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        for embedding_func in embeddings_functions:
            try:
                embeddings.append(embedding_func.embed_documents(texts))
            except NotImplementedError:
                embeddings.append([embedding_func.embed_query(x) for x in texts])

        upsert_list = self._prepare_insert_list(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
            force_ids=True,
        )

        # Total upsert count
        total_count = len(upsert_list)

        assert isinstance(self.col, Collection)
        for i in range(0, total_count, batch_size):
            # Grab end index
            end = min(i + batch_size, total_count)
            batch_upsert_list = upsert_list[i:end]
            # Upsert into the collection.
            try:
                timeout = self.timeout or timeout
                self.client.upsert(
                    self.collection_name,
                    batch_upsert_list,
                    timeout=timeout,
                    **kwargs,
                )
            except MilvusException as e:
                self._handle_batch_operation_exception(
                    e, batch_upsert_list, i, total_count, "upsert"
                )
        return

    @staticmethod
    def _as_list(value: Optional[Union[T, List[T]]]) -> List[T]:
        """Try to cast a value to a list"""
        if not value:
            return []
        return [value] if not isinstance(value, list) else value

    @staticmethod
    def _from_list(value: Optional[Union[T, List[T]]]) -> Optional[Union[T, List[T]]]:
        """Try to cast a list to a single value"""
        if isinstance(value, list) and len(value) == 1:
            return value[0]
        return value

    def _create_ranker(
        self,
        ranker_type: Optional[Literal["rrf", "weighted"]],
        ranker_params: dict,
    ) -> Union[WeightedRanker, RRFRanker]:
        """A Ranker factory method"""
        default_weights = [1.0] * len(self._as_list(self._vector_field))
        if not ranker_type:
            return WeightedRanker(*default_weights)

        if ranker_type == "weighted":
            weights = ranker_params.get("weights", default_weights)
            return WeightedRanker(*weights)
        elif ranker_type == "rrf":
            k = ranker_params.get("k", None)
            if k:
                return RRFRanker(k)
            return RRFRanker()
        else:
            logger.error(
                "Ranker %s does not exist. "
                "Please use on of the following rankers: %s, %s",
                ranker_type,
                "weighted",
                "rrf",
            )
            raise ValueError("Unrecognized ranker of type %s", ranker_type)

    def _remove_forbidden_fields(self, fields: List[str]) -> List[str]:
        """Bm25 function fields are not allowed as output fields in Milvus."""
        forbidden_fields = []
        for builtin_function in self._as_list(self.builtin_func):
            if builtin_function.type == FunctionType.BM25:
                forbidden_fields.extend(
                    self._as_list(builtin_function.output_field_names)
                )
        return [field for field in fields if field not in forbidden_fields]

    def search_by_metadata(
        self, expr: str, fields: Optional[List[str]] = None, limit: int = 10
    ) -> List[Document]:
        """
        Searches the Milvus vector store based on metadata conditions.

        This function performs a metadata-based query using an expression
        that filters stored documents without vector similarity.

        Args:
            expr (str): A filtering expression (e.g., `"city == 'Seoul'"`).
            fields (Optional[List[str]]): List of fields to retrieve.
                                          If None, retrieves all available fields.
            limit (int): Maximum number of results to return.

        Returns:
            List[Document]: List of documents matching the metadata filter.
        """
        from pymilvus import MilvusException

        if self.col is None:
            logger.debug("No existing collection to search.")
            return []

        # Default to retrieving all fields if none are provided
        if fields is None:
            fields = self.fields

        try:
            results = self.client.query(
                self.collection_name,
                filter=expr,
                output_fields=fields,
                limit=limit,
            )
            return [
                Document(page_content=result[self._text_field], metadata=result)
                for result in results
            ]
        except MilvusException as e:
            logger.error(f"Metadata search failed: {e}")
            return []

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        timeout: Optional[float] = None,
        batch_size: int = 1000,
        *,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Insert text data into Milvus asynchronously.

        Inserting data when the collection has not be made yet will result
        in creating a new Collection. The data of the first entity decides
        the schema of the new collection, the dim is extracted from the first
        embedding and the columns are decided by the first metadata dict.
        Metadata keys will need to be present for all inserted values. At
        the moment there is no None equivalent in Milvus.

        Args:
            texts (Iterable[str]): The texts to embed, it is assumed
                that they all fit in memory.
            metadatas (Optional[List[dict]]): Metadata dicts attached to each of
                the texts. Defaults to None.
            should be less than 65535 bytes. Required and work when auto_id is False.
            timeout (Optional[float]): Timeout for each batch insert. Defaults
                to None.
            batch_size (int, optional): Batch size to use for insertion.
                Defaults to 1000.
            ids (Optional[List[str]]): List of text ids. The length of each item

        Raises:
            MilvusException: Failure to add texts

        Returns:
            List[str]: The resulting keys for each inserted element.
        """
        texts = list(texts)
        if not self.auto_id and ids is None:
            warnings.warn(
                "No ids provided and auto_id is False. "
                "Setting auto_id to True automatically.",
                UserWarning,
            )
            self.auto_id = True
        elif not self.auto_id and ids:  # Check ids
            assert len(set(ids)) == len(
                texts
            ), "Different lengths of texts and unique ids are provided."

        elif self.auto_id and ids:
            logger.warning(
                "The ids parameter is ignored when auto_id is True. "
                "The ids will be generated automatically."
            )

        embeddings_functions: List[EmbeddingType] = self._as_list(self.embedding_func)
        embeddings: List = []

        for embedding_func in embeddings_functions:
            try:
                embeddings.append(await embedding_func.aembed_documents(texts))
            except NotImplementedError:
                embeddings.append([await embedding_func.aembed_query(x) for x in texts])

        if len(texts) == 0:
            logger.debug("Nothing to insert, skipping.")
            return []

        # Transpose it into row-wise
        if self._is_multi_embedding:
            transposed_embeddings = [
                [embeddings[j][i] for j in range(len(embeddings))]
                for i in range(len(embeddings[0]))
            ]
        else:
            transposed_embeddings = embeddings[0] if len(embeddings) > 0 else []

        return await self.aadd_embeddings(
            texts=texts,
            embeddings=transposed_embeddings,
            metadatas=metadatas,
            timeout=timeout,
            batch_size=batch_size,
            ids=ids,
            **kwargs,
        )

    async def aadd_embeddings(
        self,
        texts: List[str],
        embeddings: List[List[float]] | List[List[List[float]]],
        metadatas: Optional[List[dict]] = None,
        timeout: Optional[float] = None,
        batch_size: int = 1000,
        *,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Insert text data with embeddings vectors into Milvus asynchronously.

        This method inserts a batch of text embeddings into a Milvus collection.
        If the collection is not initialized, it will automatically initialize
        the collection based on the embeddings,metadatas, and other parameters.
        The embeddings are expected to be pre-generated using compatible embedding
        functions, and the metadata associated with each text is optional but
        must match the number of texts.

        Args:
            texts (List[str]): the texts to insert
            embeddings (List[List[float]] | List[List[List[float]]]):
                A vector embeddings for each text (in case of a single vector)
                or list of vectors for each text (in case of multi-vector)
            metadatas (Optional[List[dict]]): Metadata dicts attached to each of
                the texts. Defaults to None.
            should be less than 65535 bytes. Required and work when auto_id is False.
            timeout (Optional[float]): Timeout for each batch insert. Defaults
                to None.
            batch_size (int, optional): Batch size to use for insertion.
                Defaults to 1000.
            ids (Optional[List[str]]): List of text ids. The length of each item

        Raises:
            MilvusException: Failure to add texts and embeddings

        Returns:
            List[str]: The resulting keys for each inserted element.
        """

        if embeddings:
            # row-wise -> field-wise
            if not self._is_multi_embedding:
                embeddings = [[embedding] for embedding in embeddings]  # type: ignore
            # Transpose embeddings to make it a list of embeddings of each type.
            embeddings = [  # type: ignore
                [embeddings[j][i] for j in range(len(embeddings))]
                for i in range(len(embeddings[0]))
            ]

        # If the collection hasn't been initialized yet, perform all steps to do so
        if not isinstance(self.col, Collection):
            kwargs = {"embeddings": embeddings, "metadatas": metadatas}
            if self.partition_names:
                kwargs["partition_names"] = self.partition_names
            if self.replica_number:
                kwargs["replica_number"] = self.replica_number
            if self.timeout:
                kwargs["timeout"] = self.timeout
            self._init(**kwargs)

        insert_list = self._prepare_insert_list(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

        # Total insert count
        total_count = len(insert_list)

        pks: list[str] = []

        assert isinstance(self.col, Collection)
        for i in range(0, total_count, batch_size):
            # Grab end index
            end = min(i + batch_size, total_count)
            batch_insert_list = insert_list[i:end]
            # Insert into the collection.
            try:
                timeout = self.timeout or timeout
                res = await self.aclient.insert(
                    self.collection_name,
                    batch_insert_list,
                    timeout=timeout,
                    **kwargs,
                )
                pks.extend(res["ids"])
            except MilvusException as e:
                self._handle_batch_operation_exception(
                    e, batch_insert_list, i, total_count, "insert"
                )
        return pks

    async def _acollection_search(
        self,
        embedding_or_text: List[float] | Dict[int, float] | str,
        k: int = 4,
        param: Optional[dict] = None,
        expr: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> Optional[List[List[dict]]]:
        """Perform an async search on an embedding or a query text and return milvus
        search results.

        For more information about the search parameters, take a look at the pymilvus
        documentation found here:
        https://milvus.io/api-reference/pymilvus/v2.5.x/ORM/Collection/search.md

        Args:
            embedding_or_text (List[float] | Dict[int, float] | str): The embedding
                vector or query text being searched.
            k (int, optional): The amount of results to return. Defaults to 4.
            param (dict): The search params for the specified index.
                Defaults to None.
            expr (str, optional): Filtering expression. Defaults to None.
            timeout (float, optional): How long to wait before timeout error.
                Defaults to None.
            kwargs: Collection.search() keyword arguments.

        Returns:
            List[List[dict]]: Milvus search result.
        """
        if self.col is None:
            logger.debug("No existing collection to search.")
            return None

        assert not self._is_multi_vector, (
            "_acollection_search does not support multi-vector search. "
            "You can use _acollection_hybrid_search instead."
        )

        if param is None:
            assert len(self._as_list(self.search_params)) == 1, (
                "The number of search params is larger than 1, "
                "please check the search_params in this Milvus instance."
            )
            param = self._as_list(self.search_params)[0]

        if self.enable_dynamic_field:
            output_fields = ["*"]
        else:
            output_fields = self._remove_forbidden_fields(self.fields[:])
        col_search_res = await self.aclient.search(
            self.collection_name,
            data=[embedding_or_text],
            anns_field=self._vector_field,
            search_params=param,
            limit=k,
            filter=expr,
            output_fields=output_fields,
            timeout=self.timeout or timeout,
            **kwargs,
        )
        return col_search_res

    async def _acollection_hybrid_search(
        self,
        query: str,
        k: int = 4,
        param: Optional[dict | list[dict]] = None,
        expr: Optional[str] = None,
        fetch_k: Optional[int] = 4,
        ranker_type: Optional[Literal["rrf", "weighted"]] = None,
        ranker_params: Optional[dict] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> Optional[List[List[dict]]]:
        """
        Perform an async hybrid search on a query string and return milvus search
        results.

        For more information about the search parameters, take a look at the pymilvus
        documentation found here:
        https://milvus.io/api-reference/pymilvus/v2.5.x/ORM/Collection/hybrid_search.md

        Args:
            query (str): The text being searched.
            k (int, optional): The amount of results to return. Defaults to 4.
            param (dict | list[dict], optional): The search params for the specified
                index. Defaults to None.
            expr (str, optional): Filtering expression. Defaults to None.
            fetch_k (int, optional): The amount of pre-fetching results for each query.
                Defaults to 4.
            ranker_type (str, optional): The type of ranker to use. Defaults to None.
            ranker_params (dict, optional): The parameters for the ranker.
                Defaults to None.
            timeout (float, optional): How long to wait before timeout error.
                Defaults to None.
            kwargs: Collection.hybrid_search() keyword arguments.

        Returns:
            List[List[dict]]: Milvus search result.
        """
        if self.col is None:
            logger.debug("No existing collection to search.")
            return None

        search_requests = []
        reranker = self._create_ranker(
            ranker_type=ranker_type,
            ranker_params=ranker_params or {},
        )
        if not param:
            param_list = self._as_list(self.search_params)
        else:
            assert len(self._as_list(param)) == len(
                self._as_list(self.search_params)
            ), (
                f"The number of search params ({len(self._as_list(param))})"
                f" does not match the number of vector fields "
                f"({len(self._as_list(self._vector_field))})."
                f" All vector fields are: {(self._as_list(self._vector_field))},"
                " please provide a list of search params for each vector field."
            )
            param_list = self._as_list(param)
        for field, param_dict in zip(self._vector_field, param_list):
            search_data: List[float] | Dict[int, float] | str
            if field in self._vector_fields_from_embedding:
                embedding_func: EmbeddingType = self._as_list(self.embedding_func)[  # type: ignore
                    self._vector_fields_from_embedding.index(field)
                ]
                search_data = await embedding_func.aembed_query(query)
            else:
                search_data = query
            request = AnnSearchRequest(
                data=[search_data],
                anns_field=field,
                param=param_dict,
                limit=fetch_k,
                expr=expr,
            )
            search_requests.append(request)
        if self.enable_dynamic_field:
            output_fields = ["*"]
        else:
            output_fields = self._remove_forbidden_fields(self.fields[:])
        col_search_res = await self.aclient.hybrid_search(
            self.collection_name,
            reqs=search_requests,
            ranker=reranker,
            limit=k,
            output_fields=output_fields,
            timeout=self.timeout or timeout,
            **kwargs,
        )
        return col_search_res

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        param: Optional[dict | list[dict]] = None,
        expr: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Perform an async similarity search against the query string.

        Args:
            query (str): The text to search.
            k (int, optional): How many results to return. Defaults to 4.
            param (dict | list[dict], optional): The search params for the index type.
                Defaults to None.
            expr (str, optional): Filtering expression. Defaults to None.
            timeout (int, optional): How long to wait before timeout error.
                Defaults to None.
            kwargs: Collection.search() keyword arguments.

        Returns:
            List[Document]: Document results for search.
        """
        if self.col is None:
            logger.debug("No existing collection to search.")
            return []
        timeout = self.timeout or timeout
        res = await self.asimilarity_search_with_score(
            query=query, k=k, param=param, expr=expr, timeout=timeout, **kwargs
        )
        return [doc for doc, _ in res]

    async def asimilarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        param: Optional[dict] = None,
        expr: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Perform an async similarity search against the query vector.

        Args:
            embedding (List[float]): The embedding vector to search.
            k (int, optional): How many results to return. Defaults to 4.
            param (dict, optional): The search params for the index type.
                Defaults to None.
            expr (str, optional): Filtering expression. Defaults to None.
            timeout (int, optional): How long to wait before timeout error.
                Defaults to None.
            kwargs: Collection.search() keyword arguments.

        Returns:
            List[Document]: Document results for search.
        """
        if self.col is None:
            logger.debug("No existing collection to search.")
            return []
        timeout = self.timeout or timeout
        res = await self.asimilarity_search_with_score_by_vector(
            embedding=embedding, k=k, param=param, expr=expr, timeout=timeout, **kwargs
        )
        return [doc for doc, _ in res]

    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        param: Optional[dict | list[dict]] = None,
        expr: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Perform an async search on a query string and return results with score.

        For more information about the search parameters, take a look at the pymilvus
        documentation found here:
        https://milvus.io/api-reference/pymilvus/v2.5.x/ORM/Collection/search.md

        Args:
            query (str): The text being searched.
            k (int, optional): The amount of results to return. Defaults to 4.
            param (dict | list[dict], optional): The search params for the specified
            index. Defaults to None.
            expr (str, optional): Filtering expression. Defaults to None.
            timeout (float, optional): How long to wait before timeout error.
                Defaults to None.
            kwargs: Collection.search() or hybrid_search() keyword arguments.

        Returns:
            List[Tuple[Document, float]]: List of result doc and score.
        """
        if self.col is None:
            logger.debug("No existing collection to search.")
            return []

        if self._is_multi_vector:
            col_search_res = await self._acollection_hybrid_search(
                query=query, k=k, param=param, expr=expr, timeout=timeout, **kwargs
            )

        else:
            assert len(self._as_list(param)) <= 1, (
                "When there is only one vector field, you can not provide multiple "
                "search param dicts."
            )
            param = cast(Optional[dict], self._from_list(param))
            if (
                len(self._as_list(self.embedding_func)) == 1  # type: ignore[arg-type]
                and len(self._as_list(self.builtin_func)) == 0
            ):
                embedding = await self._as_list(self.embedding_func)[0].aembed_query(  # type: ignore
                    query
                )
                col_search_res = await self._acollection_search(
                    embedding_or_text=embedding,
                    k=k,
                    param=param,
                    expr=expr,
                    timeout=timeout,
                    **kwargs,
                )
            elif (
                len(self._as_list(self.embedding_func)) == 0  # type: ignore[arg-type]
                and len(self._as_list(self.builtin_func)) == 1
            ):
                col_search_res = await self._acollection_search(
                    embedding_or_text=query,
                    k=k,
                    param=param,
                    expr=expr,
                    timeout=timeout,
                    **kwargs,
                )
            else:
                raise RuntimeError(
                    "Check either it's multi vectors or single vector with "
                    "only one embedding/builtin function."
                )

        return self._parse_documents_from_search_results(col_search_res)

    async def asimilarity_search_with_score_by_vector(
        self,
        embedding: List[float] | Dict[int, float],
        k: int = 4,
        param: Optional[dict] = None,
        expr: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Perform an async search on an embedding and return results with score.

        For more information about the search parameters, take a look at the pymilvus
        documentation found here:
        https://milvus.io/api-reference/pymilvus/v2.5.x/ORM/Collection/search.md

        Args:
            embedding (List[float] | Dict[int, float]): The embedding vector being
                searched.
            k (int, optional): The amount of results to return. Defaults to 4.
            param (dict): The search params for the specified index.
                Defaults to None.
            expr (str, optional): Filtering expression. Defaults to None.
            timeout (float, optional): How long to wait before timeout error.
                Defaults to None.
            kwargs: Collection.search() keyword arguments.

        Returns:
            List[Tuple[Document, float]]: Result doc and score.
        """
        col_search_res = await self._acollection_search(
            embedding_or_text=embedding,
            k=k,
            param=param,
            expr=expr,
            timeout=timeout,
            **kwargs,
        )
        return self._parse_documents_from_search_results(col_search_res)

    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        param: Optional[dict] = None,
        expr: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Perform an async search and return results that are reordered by MMR.

        Args:
            query (str): The text being searched.
            k (int, optional): How many results to give. Defaults to 4.
            fetch_k (int, optional): Total results to select k from.
                Defaults to 20.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5
            param (dict, optional): The search params for the specified index.
                Defaults to None.
            expr (str, optional): Filtering expression. Defaults to None.
            timeout (float, optional): How long to wait before timeout error.
                Defaults to None.
            kwargs: Collection.search() keyword arguments.


        Returns:
            List[Document]: Document results for search.
        """
        if self.col is None:
            logger.debug("No existing collection to search.")
            return []

        assert (
            len(self._as_list(self.embedding_func)) == 1  # type: ignore[arg-type]
        ), "You must set only one embedding function for MMR search."
        if len(self._vector_fields_from_function) > 0:
            logger.warning(
                "MMR search will only use the embedding function, "
                "without the built-in functions."
            )

        embedding = await self._as_list(self.embedding_func)[0].aembed_query(query)  # type: ignore
        timeout = self.timeout or timeout
        return await self.amax_marginal_relevance_search_by_vector(
            embedding=embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            param=param,
            expr=expr,
            timeout=timeout,
            **kwargs,
        )

    async def amax_marginal_relevance_search_by_vector(
        self,
        embedding: list[float] | dict[int, float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        param: Optional[dict] = None,
        expr: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Perform an async search and return results that are reordered by MMR.

        Args:
            embedding (list[float] | dict[int, float]): The embedding vector being
                searched.
            k (int, optional): How many results to give. Defaults to 4.
            fetch_k (int, optional): Total results to select k from.
                Defaults to 20.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5
            param (dict, optional): The search params for the specified index.
                Defaults to None.
            expr (str, optional): Filtering expression. Defaults to None.
            timeout (float, optional): How long to wait before timeout error.
                Defaults to None.
            kwargs: Collection.search() keyword arguments.

        Returns:
            List[Document]: Document results for search.
        """
        col_search_res = await self._acollection_search(
            embedding_or_text=embedding,
            k=fetch_k,
            param=param,
            expr=expr,
            timeout=timeout,
            **kwargs,
        )
        if col_search_res is None:
            return []
        ids = []
        documents = []
        scores = []
        for result in col_search_res[0]:
            doc = self._parse_document(result["entity"])
            documents.append(doc)
            scores.append(result["distance"])
            ids.append(result.get(self._primary_field, "id"))

        vectors = await self.aclient.query(
            self.collection_name,
            filter=f"{self._primary_field} in {ids}",
            output_fields=[self._primary_field, self._vector_field],
            timeout=timeout,
        )
        # Reorganize the results from query to match search order.
        vectors = {x[self._primary_field]: x[self._vector_field] for x in vectors}

        ordered_result_embeddings = [vectors[x] for x in ids]

        # Get the new order of results.
        new_ordering = maximal_marginal_relevance(
            np.array(embedding), ordered_result_embeddings, k=k, lambda_mult=lambda_mult
        )

        # Reorder the values and return.
        ret = []
        for x in new_ordering:
            # Function can return -1 index
            if x == -1:
                break
            else:
                ret.append(documents[x])
        return ret

    async def adelete(
        self, ids: Optional[List[str]] = None, expr: Optional[str] = None, **kwargs: Any
    ) -> Optional[bool]:
        """Async delete by vector ID or boolean expression.
        Refer to [Milvus documentation](https://milvus.io/docs/delete_data.md)
        for notes and examples of expressions.

        Args:
            ids: List of ids to delete.
            expr: Boolean expression that specifies the entities to delete.
            kwargs: Other parameters in Milvus delete api.

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise.
        """
        if isinstance(ids, list) and len(ids) > 0:
            if expr is not None:
                logger.warning(
                    "Both ids and expr are provided. " "Ignore expr and delete by ids."
                )
            expr = f"{self._primary_field} in {ids}"
        else:
            assert isinstance(
                expr, str
            ), "Either ids list or expr string must be provided."
        try:
            await self.aclient.delete(self.collection_name, filter=expr, **kwargs)
            return True
        except MilvusException as e:
            logger.error(
                "Failed to delete entities: %s error: %s", self.collection_name, e
            )
            return False

    @classmethod
    async def afrom_texts(
        cls,
        texts: List[str],
        embedding: Optional[Union[EmbeddingType, List[EmbeddingType]]],
        metadatas: Optional[List[dict]] = None,
        collection_name: str = "LangChainCollection",
        connection_args: Optional[Dict[str, Any]] = None,
        consistency_level: str = "Session",
        index_params: Optional[Union[dict, List[dict]]] = None,
        search_params: Optional[Union[dict, List[dict]]] = None,
        drop_old: bool = False,
        *,
        ids: Optional[List[str]] = None,
        auto_id: bool = False,
        builtin_function: Optional[
            Union[BaseMilvusBuiltInFunction, List[BaseMilvusBuiltInFunction]]
        ] = None,
        **kwargs: Any,
    ) -> Milvus:
        """Create a Milvus collection, indexes it with HNSW, and insert data
        asynchronously.

        Args:
            texts (List[str]): Text data.
            embedding (Optional[Union[Embeddings, BaseSparseEmbedding]]): Embedding
                function.
            metadatas (Optional[List[dict]]): Metadata for each text if it exists.
                Defaults to None.
            collection_name (str, optional): Collection name to use. Defaults to
                "LangChainCollection".
            connection_args (dict[str, Any], optional): Connection args to use. Defaults
                to DEFAULT_MILVUS_CONNECTION.
            consistency_level (str, optional): Which consistency level to use. Defaults
                to "Session".
            index_params (Optional[dict], optional): Which index_params to use. Defaults
                to None.
            search_params (Optional[dict], optional): Which search params to use.
                Defaults to None.
            drop_old (Optional[bool], optional): Whether to drop the collection with
                that name if it exists. Defaults to False.
            ids (Optional[List[str]]): List of text ids. Defaults to None.
            auto_id (bool): Whether to enable auto id for primary key. Defaults to
                False. If False, you need to provide text ids (string less than 65535
                bytes). If True, Milvus will generate unique integers as primary keys.
            builtin_function (Optional[Union[BaseMilvusBuiltInFunction,
                List[BaseMilvusBuiltInFunction]]]):
                Built-in function to use. Defaults to None.
            **kwargs: Other parameters in Milvus Collection.
        Returns:
            Milvus: Milvus Vector Store
        """
        if isinstance(ids, list) and len(ids) > 0:
            if auto_id:
                logger.warning(
                    "Both ids and auto_id are provided. " "Ignore auto_id and use ids."
                )
            auto_id = False
        else:
            auto_id = True

        vector_db = cls(
            embedding_function=embedding,
            collection_name=collection_name,
            connection_args=connection_args,
            consistency_level=consistency_level,
            index_params=index_params,
            search_params=search_params,
            drop_old=drop_old,
            auto_id=auto_id,
            builtin_function=builtin_function,
            **kwargs,
        )
        await vector_db.aadd_texts(texts=texts, metadatas=metadatas, ids=ids)
        return vector_db

    async def aadd_documents(
        self, documents: List[Document], **kwargs: Any
    ) -> List[str]:
        """Run more documents through the embeddings and add to the vectorstore
        asynchronously.

        Args:
            documents: Documents to add to the vectorstore.

        Returns:
            List of IDs of the added texts.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return await self.aadd_texts(texts, metadatas, **kwargs)

    async def aget_pks(self, expr: str, **kwargs: Any) -> List[int] | None:
        """Async get primary keys with expression

        Args:
            expr: Expression - E.g: "id in [1, 2]", or "title LIKE 'Abc%'"

        Returns:
            List[int]: List of IDs (Primary Keys)
        """

        if self.col is None:
            logger.debug("No existing collection to get pk.")
            return None

        try:
            query_result = await self.aclient.query(
                self.collection_name,
                filter=expr,
                output_fields=[self._primary_field],
            )
        except MilvusException as exc:
            logger.error("Failed to get ids: %s error: %s", self.collection_name, exc)
            raise exc
        pks = [item.get(self._primary_field) for item in query_result]
        return pks

    async def aupsert(
        self,
        ids: Optional[List[str]] = None,
        documents: List[Document] | None = None,
        batch_size: int = 1000,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Update/Insert documents to the vectorstore asynchronously.

        Args:
            ids: IDs to update - Let's call aget_pks to get ids with expression
            documents (List[Document]): Documents to add to the vectorstore.
            batch_size (int, optional): Batch size to use for upsert.
                Defaults to 1000.
            timeout (Optional[float]): Timeout for each batch upsert. Defaults
                to None.
            **kwargs: Other parameters in Milvus upsert api.
        """

        if documents is None or len(documents) == 0:
            logger.debug("No documents to upsert.")
            return

        if not ids:
            await self.aadd_documents(documents=documents, **kwargs)
            return

        assert len(set(ids)) == len(
            documents
        ), "Different lengths of documents and unique ids are provided."

        embeddings_functions: List[EmbeddingType] = self._as_list(self.embedding_func)
        embeddings: List = []
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        for embedding_func in embeddings_functions:
            try:
                embeddings.append(await embedding_func.aembed_documents(texts))
            except NotImplementedError:
                embeddings.append([await embedding_func.aembed_query(x) for x in texts])

        upsert_list = self._prepare_insert_list(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
            force_ids=True,
        )

        # Total upsert count
        total_count = len(upsert_list)

        assert isinstance(self.col, Collection)
        for i in range(0, total_count, batch_size):
            # Grab end index
            end = min(i + batch_size, total_count)
            batch_upsert_list = upsert_list[i:end]
            # Upsert into the collection.
            try:
                timeout = self.timeout or timeout
                await self.aclient.upsert(
                    self.collection_name,
                    batch_upsert_list,
                    timeout=timeout,
                    **kwargs,
                )
            except MilvusException as e:
                self._handle_batch_operation_exception(
                    e, batch_upsert_list, i, total_count, "upsert"
                )
        return

    async def asearch_by_metadata(
        self, expr: str, fields: Optional[List[str]] = None, limit: int = 10
    ) -> List[Document]:
        """
        Async searches the Milvus vector store based on metadata conditions.

        This function performs a metadata-based query using an expression
        that filters stored documents without vector similarity.

        Args:
            expr (str): A filtering expression (e.g., `"city == 'Seoul'"`).
            fields (Optional[List[str]]): List of fields to retrieve.
                                          If None, retrieves all available fields.
            limit (int): Maximum number of results to return.

        Returns:
            List[Document]: List of documents matching the metadata filter.
        """
        from pymilvus import MilvusException

        if self.col is None:
            logger.debug("No existing collection to search.")
            return []

        # Default to retrieving all fields if none are provided
        if fields is None:
            fields = self.fields

        try:
            results = await self.aclient.query(
                self.collection_name,
                filter=expr,
                output_fields=fields,
                limit=limit,
            )
            return [
                Document(page_content=result[self._text_field], metadata=result)
                for result in results
            ]
        except MilvusException as e:
            logger.error(f"Metadata search failed: {e}")
            return []
