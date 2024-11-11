from __future__ import annotations

import logging
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
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    MilvusException,
    RRFRanker,
    WeightedRanker,
    utility,
)
from pymilvus.client.types import LoadState  # type: ignore
from pymilvus.orm.types import infer_dtype_bydata  # type: ignore

from langchain_milvus import MilvusCollectionHybridSearchRetriever
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

    Search with filter:
        .. code-block:: python

            results = vector_store.similarity_search(query="thud",k=1,filter={"bar": "baz"})
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
        embedding_function: Union[EmbeddingType, List[EmbeddingType]],  # type: ignore
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
        primary_field: str = "pk",
        text_field: str = "text",
        vector_field: Union[str, List[str]] = "vector",
        enable_dynamic_field: bool = False,
        metadata_field: Optional[str] = None,
        partition_key_field: Optional[str] = None,
        partition_names: Optional[list] = None,
        replica_number: int = 1,
        timeout: Optional[float] = None,
        num_shards: Optional[int] = None,
        vector_schema: Optional[Union[dict[str, Any], List[dict[str, Any]]]] = None,
        metadata_schema: Optional[dict[str, Any]] = None,
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
            "SPARSE_INVERTED_INDEX": {
                "metric_type": "IP",
                "params": {"drop_ratio_build": 0.2},
            },
            "SPARSE_WAND": {"metric_type": "IP", "params": {"drop_ratio_build": 0.2}},
        }

        self.embedding_func = embedding_function
        self.collection_name = collection_name
        self.collection_description = collection_description
        self.collection_properties = collection_properties
        self.index_params = index_params
        self.search_params = search_params
        self.consistency_level = consistency_level
        self.auto_id = auto_id

        # In order for a collection to be compatible, pk needs to be varchar
        self._primary_field = primary_field
        # In order for compatibility, the text field will need to be called "text"
        self._text_field = text_field

        if isinstance(self.embedding_func, list):
            if len(self.embedding_func) == 1:
                self.embedding_func = self.embedding_func[0]
            else:
                self.embedding_func = cast(List[EmbeddingType], self.embedding_func)
                if not isinstance(vector_field, list):
                    vector_field = [
                        f"vector_{i + 1}" for i, e in enumerate(self.embedding_func)
                    ]
                    logger.warning(
                        "When multiple embeddings function are used, one should provide"
                        "matching `vector_field` names. "
                        "Using generated vector names %s",
                        vector_field,
                    )

        # In order for compatibility, the vector field needs to be called "vector"
        self._vector_field = vector_field
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
        self.vector_schema = vector_schema

        # Create the connection to the server
        if connection_args is None:
            connection_args = DEFAULT_MILVUS_CONNECTION
        self._milvus_client = MilvusClient(
            **connection_args,
        )
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

    @property
    def embeddings(self) -> Union[EmbeddingType, List[EmbeddingType]]:  # type: ignore
        return self.embedding_func

    @property
    def client(self) -> MilvusClient:
        """Get client."""
        return self._milvus_client

    @property
    def _is_multi_vector(self) -> bool:
        return isinstance(self.embedding_func, list)

    @property
    def _is_sparse(self) -> bool:
        embedding_func: List[EmbeddingType] = self._as_list(self.embedding_func)
        if self._is_sparse_embedding(embedding_func[0]):
            return True
        else:
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
        fields = []
        vector_fields: List[str] = self._as_list(self._vector_field)
        # If enable_dynamic_field, we don't need to create fields, and just pass it.
        # In the future, when metadata_field is deprecated,
        # This logical structure will be simplified like this:
        # ```
        # if not self.enable_dynamic_field and metadatas:
        #     for key, value in metadatas[0].items():
        #         ...
        # ```
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
                for key, value in metadatas[0].items():
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
                            fields.append(
                                FieldSchema(key, DataType.VARCHAR, max_length=65_535)
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

        # Create the text field
        fields.append(
            FieldSchema(self._text_field, DataType.VARCHAR, max_length=65_535)
        )
        # Create the primary key field
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

        embeddings_functions: List[EmbeddingType] = self._as_list(self.embedding_func)
        vector_schemas: List[dict[str, Any]] = (
            self._as_list(self.vector_schema)
            if self.vector_schema
            else [{} for _ in range(len(embeddings_functions))]
        )
        for vector_field, vector_schema, embedding_func, vector_field_embeddings in zip(
            vector_fields, vector_schemas, embeddings_functions, embeddings
        ):
            dim = len(vector_field_embeddings[0])
            # Create the vector field
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
                            infer_dtype_bydata(vector_field_embeddings[0]),
                            dim=dim,
                        )
                    )

        # Create the schema for the collection
        schema = CollectionSchema(
            fields,
            description=self.collection_description,
            partition_key_field=self._partition_key_field,
            enable_dynamic_field=self.enable_dynamic_field,
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
                )
            else:
                self.col = Collection(
                    name=self.collection_name,
                    schema=schema,
                    consistency_level=self.consistency_level,
                    using=self.alias,
                )
            # Set the collection properties if they exist
            if self.collection_properties is not None:
                self.col.set_properties(self.collection_properties)
        except MilvusException as e:
            logger.error(
                "Failed to create collection: %s error: %s", self.collection_name, e
            )
            raise e

    def _get_field_schema_from_dict(self, field_name: str, schema_dict: dict):  # type: ignore[no-untyped-def]
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

    def _create_index(self) -> None:
        """Create an index on the collection"""
        if isinstance(self.col, Collection) and self._get_index() is None:
            embeddings_functions: List[EmbeddingType] = self._as_list(
                self.embedding_func
            )
            vector_fields: List[str] = self._as_list(self._vector_field)
            if self.index_params is None:
                indexes_params: List[dict] = [
                    {} for _ in range(len(embeddings_functions))
                ]
            else:
                indexes_params = self._as_list(self.index_params)

            for i, embeddings_func in enumerate(embeddings_functions):
                if not self._get_index(vector_fields[i]):
                    try:
                        # If no index params, use a default HNSW based one
                        if not indexes_params[i]:
                            if self._is_sparse_embedding(embeddings_func):
                                indexes_params[i] = {
                                    "metric_type": "IP",
                                    "index_type": "SPARSE_INVERTED_INDEX",
                                    "params": {"drop_ratio_build": 0.2},
                                }
                            else:
                                indexes_params[i] = {
                                    "metric_type": "L2",
                                    "index_type": "HNSW",
                                    "params": {"M": 8, "efConstruction": 64},
                                }

                        try:
                            self.col.create_index(
                                vector_fields[i],
                                index_params=indexes_params[i],
                                using=self.alias,
                            )

                        # If default did not work, most likely on Zilliz Cloud
                        except MilvusException:
                            # Use AUTOINDEX based index
                            index_params = {
                                "metric_type": "L2",
                                "index_type": "AUTOINDEX",
                                "params": {},
                            }
                            self.col.create_index(
                                vector_fields[i],
                                index_params=index_params,
                                using=self.alias,
                            )
                        logger.debug(
                            "Successfully created an index"
                            "on %s field on collection: %s",
                            vector_fields[i],
                            self.collection_name,
                        )
                    except MilvusException as e:
                        logger.error(
                            "Failed to create an index on collection: %s",
                            self.collection_name,
                        )
                        raise e
            if self._is_multi_vector:
                self.index_params = indexes_params
            else:
                self.index_params = indexes_params[0]

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
            if self._is_multi_vector:
                self.search_params = search_params_list
            else:
                self.search_params = search_params_list[0]

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
            and self._get_index() is not None
            and utility.load_state(self.collection_name, using=self.alias)
            == LoadState.NotLoad
        ):
            self.col.load(
                partition_names=partition_names,
                replica_number=replica_number,
                timeout=timeout,
            )

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
        if not self.auto_id:
            assert isinstance(ids, list), (
                "A list of valid ids are required when auto_id is False. "
                "You can set `auto_id` to True in this Milvus instance to generate "
                "ids automatically, or specify string-type ids for each text."
            )
            assert len(set(ids)) == len(
                texts
            ), "Different lengths of texts and unique ids are provided."
            assert all(isinstance(x, str) for x in ids), "All ids should be strings."
            assert all(
                len(x.encode()) <= 65_535 for x in ids
            ), "Each id should be a string less than 65535 bytes."

        else:
            if ids is not None:
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

        # assuming [f1, f2] embeddings functions and [a, b, c] as texts:
        # embeddings = [
        #     [f1(a), f1(b), f1(c)],
        #     [f2(a), f2(b), f2(c)]
        # ]
        # or
        # embeddings = [
        #     [f1(a), f1(b), f1(c)]
        # ]

        if len(embeddings) == 0:
            logger.debug("Nothing to insert, skipping.")
            return []

        if self._is_multi_vector:
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
            transposed_embeddings = embeddings[0]

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
            embeddings (List[List[Union[float, List[float]]]]):
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

        if not self._is_multi_vector:
            embeddings = [[embedding] for embedding in embeddings]  # type: ignore
        # Transpose embeddings to make it a list of embeddings of each type.
        embeddings = [  # type: ignore
            [embeddings[j][i] for j in range(len(embeddings))]
            for i in range(len(embeddings[0]))
        ]

        vector_fields: List[str] = self._as_list(self._vector_field)

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
            if not self.auto_id:
                entity_dict[self._primary_field] = ids[i]  # type: ignore[index]

            entity_dict[self._text_field] = text

            for vector_field, vector_field_embeddings in zip(vector_fields, embeddings):  # type: ignore
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
                res: Collection
                timeout = self.timeout or timeout
                res = self.col.insert(batch_insert_list, timeout=timeout, **kwargs)
                pks.extend(res.primary_keys)
            except MilvusException as e:
                logger.error(
                    "Failed to insert batch starting at entity: %s/%s", i, total_count
                )
                raise e
        return pks

    def _collection_search(
        self,
        embedding: List[float] | Dict[int, float],
        k: int = 4,
        param: Optional[dict] = None,
        expr: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> "pymilvus.client.abstract.SearchResult | None":  # type: ignore[name-defined] # noqa: F821
        """Perform a search on an embedding and return milvus search results.

        For more information about the search parameters, take a look at the pymilvus
        documentation found here:
        https://milvus.io/api-reference/pymilvus/v2.4.x/ORM/Collection/search.md

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
            pymilvus.client.abstract.SearchResult: Milvus search result.
        """
        if self.col is None:
            logger.debug("No existing collection to search.")
            return None

        assert not isinstance(self.search_params, list) and not isinstance(
            self._vector_field, list
        ), "_collection_search does not support multi-vector search."

        if param is None:
            param = self.search_params

        # Determine result metadata fields with PK.
        if self.enable_dynamic_field:
            output_fields = ["*"]
        else:
            output_fields = self.fields[:]
            output_fields.remove(self._vector_field)
        timeout = self.timeout or timeout
        # Perform the search.
        res = self.col.search(
            data=[embedding],
            anns_field=self._vector_field,
            param=param,
            limit=k,
            expr=expr,
            output_fields=output_fields,
            timeout=timeout,
            **kwargs,
        )
        return res

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        param: Optional[dict] = None,
        expr: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Perform a similarity search against the query string.

        Args:
            query (str): The text to search.
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
        param: Optional[dict] = None,
        expr: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Perform a search on a query string and return results with score.

        For more information about the search parameters, take a look at the pymilvus
        documentation found here:
        https://milvus.io/api-reference/pymilvus/v2.4.x/ORM/Collection/search.md

        Args:
            query (str): The text being searched.
            k (int, optional): The amount of results to return. Defaults to 4.
            param (dict): The search params for the specified index.
                Defaults to None.
            expr (str, optional): Filtering expression. Defaults to None.
            timeout (float, optional): How long to wait before timeout error.
                Defaults to None.
            kwargs: Collection.search() keyword arguments.

        Returns:
            List[float], List[Tuple[Document, any, any]]:
        """
        if self.col is None:
            logger.debug("No existing collection to search.")
            return []
        if isinstance(self.embedding_func, list):  # is multi-vector
            ranker = self._create_ranker(
                kwargs.pop("ranker_type", None), kwargs.pop("ranker_params", {})
            )
            hybrid_retriever = MilvusCollectionHybridSearchRetriever(
                collection=self.col,
                rerank=ranker,
                anns_fields=self._vector_field,
                field_embeddings=self.embeddings,
                field_search_params=param or self.search_params,
                field_exprs=expr,
                top_k=k,
                text_field=self._text_field,
                timeout=timeout,
                **kwargs,
            )
            res = []
            col_search_res = hybrid_retriever.hybrid_search(query)
            for result in col_search_res[0]:
                data = {x: result.entity.get(x) for x in result.entity.fields}
                doc = self._parse_document(data)
                res.append((doc, result.score))
        else:
            # Embed the query text.
            embedding = self.embedding_func.embed_query(query)
            timeout = self.timeout or timeout
            res = self.similarity_search_with_score_by_vector(
                embedding=embedding,
                k=k,
                param=param,
                expr=expr,
                timeout=timeout,
                **kwargs,
            )
        return res

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
        https://milvus.io/api-reference/pymilvus/v2.4.x/ORM/Collection/search.md

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
            embedding=embedding, k=k, param=param, expr=expr, timeout=timeout, **kwargs
        )
        if col_search_res is None:
            return []
        ret = []
        for result in col_search_res[0]:
            data = {x: result.entity.get(x) for x in result.entity.fields}
            doc = self._parse_document(data)
            pair = (doc, result.score)
            ret.append(pair)

        return ret

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

        assert not isinstance(
            self.embedding_func, list
        ), "MMR is not-suported in multi-vector settings"
        embedding = self.embedding_func.embed_query(query)
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
            embedding=embedding,
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
            data = {x: result.entity.get(x) for x in result.entity.fields}
            doc = self._parse_document(data)
            documents.append(doc)
            scores.append(result.score)
            ids.append(result.id)

        vectors = self.col.query(  # type: ignore[union-attr]
            expr=f"{self._primary_field} in {ids}",
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
        if self._is_multi_vector:
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

        if self.index_params is None:
            logger.warning(
                "No index params provided. Could not determine relevance function. "
                "Use L2 distance as default."
            )
            return _map_l2_to_similarity
        indexes_params = self._as_list(self.index_params)
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

    def delete(  # type: ignore[no-untyped-def]
        self, ids: Optional[List[str]] = None, expr: Optional[str] = None, **kwargs: str
    ):
        """Delete by vector ID or boolean expression.
        Refer to [Milvus documentation](https://milvus.io/docs/delete_data.md)
        for notes and examples of expressions.

        Args:
            ids: List of ids to delete.
            expr: Boolean expression that specifies the entities to delete.
            kwargs: Other parameters in Milvus delete api.
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
        return self.col.delete(expr=expr, **kwargs)  # type: ignore[union-attr]

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Union[EmbeddingType, List[EmbeddingType]],  # type: ignore
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
        **kwargs: Any,
    ) -> Milvus:
        """Create a Milvus collection, indexes it with HNSW, and insert data.

        Args:
            texts (List[str]): Text data.
            embedding (Union[Embeddings, BaseSparseEmbedding]): Embedding function.
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

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """Run more documents through the embeddings and add to the vectorstore.

        Args:
            documents: Documents to add to the vectorstore.

        Returns:
            List of IDs of the added texts.
        """
        # TODO: Handle the case where the user doesn't provide ids on the Collection
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return self.add_texts(texts, metadatas, **kwargs)

    async def aadd_documents(
        self, documents: List[Document], **kwargs: Any
    ) -> List[str]:
        """Run more documents through the embeddings and add to the vectorstore.

        Args:
            documents: Documents to add to the vectorstore.

        Returns:
            List of IDs of the added texts.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return await self.aadd_texts(texts, metadatas, **kwargs)

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
            query_result = self.col.query(
                expr=expr, output_fields=[self._primary_field]
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
        **kwargs: Any,
    ) -> List[str] | None:
        """Update/Insert documents to the vectorstore.

        Args:
            ids: IDs to update - Let's call get_pks to get ids with expression \n
            documents (List[Document]): Documents to add to the vectorstore.

        Returns:
            List[str]: IDs of the added texts.
        """

        if documents is None or len(documents) == 0:
            logger.debug("No documents to upsert.")
            return None

        if ids is not None and len(ids):
            try:
                self.delete(ids=ids)
            except MilvusException:
                pass
        try:
            return self.add_documents(documents=documents, **kwargs)
        except MilvusException as exc:
            logger.error(
                "Failed to upsert entities: %s error: %s", self.collection_name, exc
            )
            raise exc

    @staticmethod
    def _as_list(value: Union[T, List[T]]) -> List[T]:
        return [value] if not isinstance(value, list) else value

    def _create_ranker(
        self,
        ranker_type: Optional[Literal["rrf", "weighted"]],
        ranker_params: dict,
    ) -> Union[WeightedRanker, RRFRanker]:
        """A Ranker factory method"""
        embeddings_functions: List[EmbeddingType] = self._as_list(self.embedding_func)
        default_weights = [1.0] * len(embeddings_functions)
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
