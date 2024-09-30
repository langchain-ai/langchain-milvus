from __future__ import annotations

import logging

from langchain_milvus.vectorstores.milvus import Milvus

logger = logging.getLogger(__name__)


class Zilliz(Milvus):
    """`Zilliz` vector store.

    You need to have `pymilvus` installed and a
    running Zilliz database.

    See the following documentation for how to run a Zilliz instance:
    https://docs.zilliz.com/docs/create-cluster


    IF USING L2/IP metric IT IS HIGHLY SUGGESTED TO NORMALIZE YOUR DATA.

    Args:
        embedding_function (Embeddings): Function used to embed the text.
        collection_name (str): Which Zilliz collection to use. Defaults to
            "LangChainCollection".
        connection_args (Optional[dict[str, any]]): The connection args used for
            this class comes in the form of a dict.
        consistency_level (str): The consistency level to use for a collection.
            Defaults to "Session".
        index_params (Optional[dict]): Which index params to use. Defaults to
            HNSW/AUTOINDEX depending on service.
        search_params (Optional[dict]): Which search params to use. Defaults to
            default of index.
        drop_old (Optional[bool]): Whether to drop the current collection. Defaults
            to False.
        auto_id (bool): Whether to enable auto id for primary key. Defaults to False.
            If False, you need to provide text ids (string less than 65535 bytes).
            If True, Milvus will generate unique integers as primary keys.

    The connection args used for this class comes in the form of a dict,
    the two major arguments are:
        uri (str): The Public Endpoint of Zilliz instance. Example uri:
            "https://in03-ba4234asae.api.gcp-us-west1.zillizcloud.com",
        token (str): API key, for serverless clusters which can be used as
            replacements for user and password.
    For more information, please refer to:
    https://docs.zilliz.com/docs/on-zilliz-cloud-console#cluster-details
    and
    https://docs.zilliz.com/reference/python/python/Connections-connect

    Example:
        .. code-block:: python

        from langchain_milvus import Zilliz
        from langchain_openai import OpenAIEmbeddings

        embedding = OpenAIEmbeddings()
        # Connect to a Zilliz instance
        milvus_store = Zilliz(
            embedding_function = embedding,
            collection_name = "LangChainCollection",
            connection_args = {
                "uri": "https://in03-ba4234asae.api.gcp-us-west1.zillizcloud.com",
                "token": "temp", # API key
            }
            drop_old: True,
        )

    Raises:
        ValueError: If the pymilvus python package is not installed.
    """

    def _create_index(self) -> None:
        """Create a index on the collection"""
        from pymilvus import Collection, MilvusException

        if isinstance(self.col, Collection) and self._get_index() is None:
            try:
                # If no index params, use a default AutoIndex based one
                if self.index_params is None:
                    if self._is_sparse_embedding:
                        self.index_params = {
                            "metric_type": "IP",
                            "index_type": "SPARSE_INVERTED_INDEX",
                            "params": {"drop_ratio_build": 0.2},
                        }
                    else:
                        self.index_params = {
                            "metric_type": "L2",
                            "index_type": "AUTOINDEX",
                            "params": {},
                        }

                try:
                    self.col.create_index(
                        self._vector_field,
                        index_params=self.index_params,
                        using=self.alias,
                    )

                # If default did not work, most likely Milvus self-hosted
                except MilvusException:
                    # Use HNSW based index
                    self.index_params = {
                        "metric_type": "L2",
                        "index_type": "HNSW",
                        "params": {"M": 8, "efConstruction": 64},
                    }
                    self.col.create_index(
                        self._vector_field,
                        index_params=self.index_params,
                        using=self.alias,
                    )
                logger.debug(
                    "Successfully created an index on collection: %s",
                    self.collection_name,
                )

            except MilvusException as e:
                logger.error(
                    "Failed to create an index on collection: %s", self.collection_name
                )
                raise e
