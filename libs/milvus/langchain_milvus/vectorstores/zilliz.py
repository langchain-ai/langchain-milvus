from __future__ import annotations

import logging
from typing import Any

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
        *args: Variable length argument list passed to the parent Milvus class.
        **kwargs: Arbitrary keyword arguments passed to the parent Milvus class.
            See Milvus class documentation for supported parameters including:
            `embedding_function`, `collection_name`, `connection_args`,
            `consistency_level`, `index_params`, `search_params`, `drop_old`, `auto_id`,
            and others.

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
        ```python
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
        ```

    Raises:
        ValueError: If the pymilvus python package is not installed.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        import warnings

        warnings.warn(
            "The Zilliz class will be deprecated in the future. "
            "Please use the Milvus class instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)

    # For backwards compatibility, this class is preserved.
    # But it is recommended to use Milvus instead.
