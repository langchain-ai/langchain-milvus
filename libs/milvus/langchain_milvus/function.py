import uuid
from abc import ABC
from typing import Any, Dict, List, Optional, Union

from pymilvus import Function, FunctionType

from langchain_milvus.utils.constant import SPARSE_VECTOR_FIELD, TEXT_FIELD


class BaseMilvusBuiltInFunction(ABC):
    """
    Base class for Milvus built-in functions.

    See:
    https://milvus.io/docs/manage-collections.md#Function
    """

    def __init__(self) -> None:
        self._function: Optional[Function] = None

    @property
    def function(self) -> Function:
        return self._function

    @property
    def input_field_names(self) -> Union[str, List[str]]:
        return self.function.input_field_names

    @property
    def output_field_names(self) -> Union[str, List[str]]:
        return self.function.output_field_names

    @property
    def type(self) -> FunctionType:
        return self.function.type


class BM25BuiltInFunction(BaseMilvusBuiltInFunction):
    """
    Milvus BM25 built-in function.

    Supports both single-language and multi-language analyzers.

    See:
    - https://milvus.io/docs/full-text-search.md
    - https://milvus.io/docs/multi-language-analyzers.md

    Example for single-language analyzer:
        BM25BuiltInFunction(
            analyzer_params={"type": "english"}
        )

    Example for multi-language analyzers:
        BM25BuiltInFunction(
            multi_analyzer_params={
                "analyzers": {
                    "english": {"type": "english"},
                    "chinese": {"type": "chinese"},
                    "default": {"tokenizer": "icu"}
                },
                "by_field": "language",
                "alias": {
                    "cn": "chinese",
                    "en": "english"
                }
            }
        )
    """

    def __init__(
        self,
        *,
        input_field_names: str = TEXT_FIELD,
        output_field_names: str = SPARSE_VECTOR_FIELD,
        analyzer_params: Optional[Dict[Any, Any]] = None,
        multi_analyzer_params: Optional[Dict[Any, Any]] = None,
        enable_match: bool = False,
        function_name: Optional[str] = None,
    ):
        """
        Args:
            input_field_names (str): The name of the input field, default is 'text'.
            output_field_names (str): The name of the output field, default is 'sparse'.
            analyzer_params (Optional[Dict[Any, Any]]): The parameters for the analyzer.
                Default is None. See:
                https://milvus.io/docs/analyzer-overview.md#Analyzer-Overview
            multi_analyzer_params (Optional[Dict[Any, Any]]): The parameters for
                multi-language analyzers. Default is None. See:
                https://milvus.io/docs/multi-language-analyzers.md
                This parameter is mutually exclusive with analyzer_params.
                Example:
                    {
                        "analyzers": {
                            "english": {"type": "english"},
                            "chinese": {"type": "chinese"},
                            "default": {"tokenizer": "icu"}
                        },
                        "by_field": "language",
                        "alias": {
                            "cn": "chinese",
                            "en": "english"
                        }
                    }
            enable_match (bool): Whether to enable match.
            function_name (Optional[str]): The name of the function. Default is None,
                which means a random name will be generated.
        """
        super().__init__()
        if analyzer_params is not None and multi_analyzer_params is not None:
            raise ValueError(
                "analyzer_params and multi_analyzer_params cannot be set "
                "at the same time. Please use either analyzer_params for "
                "single-language analyzer or multi_analyzer_params for "
                "multi-language analyzers."
            )
        if not function_name:
            function_name = f"bm25_function_{str(uuid.uuid4())[:8]}"
        self._function = Function(
            name=function_name,
            input_field_names=input_field_names,
            output_field_names=output_field_names,
            function_type=FunctionType.BM25,
        )
        self.analyzer_params: Optional[Dict[Any, Any]] = analyzer_params
        self.multi_analyzer_params: Optional[Dict[Any, Any]] = multi_analyzer_params
        self.enable_match = enable_match

    def get_input_field_schema_kwargs(self) -> dict:
        field_schema_kwargs: Dict[Any, Any] = {
            "enable_analyzer": True,
            "enable_match": self.enable_match,
        }
        if self.multi_analyzer_params is not None:
            field_schema_kwargs["multi_analyzer_params"] = self.multi_analyzer_params
        elif self.analyzer_params is not None:
            field_schema_kwargs["analyzer_params"] = self.analyzer_params
        return field_schema_kwargs


class TextEmbeddingBuiltInFunction(BaseMilvusBuiltInFunction):
    """
    Milvus Text Embedding built-in function (Data In Data Out).

    This function allows Milvus to automatically generate embeddings from text
    by calling external embedding service providers (OpenAI, Bedrock, Vertex AI, etc.).

    See:
    https://milvus.io/docs/embedding-function-overview.md
    """

    def __init__(
        self,
        *,
        input_field_names: Union[str, List[str]],
        output_field_names: Union[str, List[str]],
        dim: int,
        params: Dict[str, Any],
        function_name: Optional[str] = None,
    ):
        """
        Args:
            input_field_names (Union[str, List[str]]): The name(s) of the
                input field(s) containing text data.
            output_field_names (Union[str, List[str]]): The name(s) of the
                output field(s) where embeddings will be stored.
            dim (int): The dimension of the output embeddings. Required
                because langchain needs to know the vector dimension upfront
                when the embedding is generated on the server side.
            params (Dict[str, Any]): Parameters for the embedding function.
                This dict is passed through to Milvus Function. Includes:
                - "provider": embedding service provider
                  (e.g., "openai", "dashscope")
                - "model_name": model name
                  (e.g., "text-embedding-3-small")
                - "credential": optional credential label from milvus.yaml
                - Other provider-specific parameters
                  (e.g., "dim", "user", "region", "url")
                Examples:
                    {"provider": "openai",
                     "model_name": "text-embedding-3-small"}
                    {"provider": "dashscope",
                     "model_name": "text-embedding-v3",
                     "credential": "apikey1"}
            function_name (Optional[str]): The name of the function.
                Default is None, which means a random name will be generated.
        """
        super().__init__()
        if not function_name:
            function_name = f"text_embedding_{str(uuid.uuid4())[:8]}"

        self._function = Function(
            name=function_name,
            input_field_names=input_field_names,
            output_field_names=output_field_names,
            function_type=FunctionType.TEXTEMBEDDING,
            params=params,
        )
        self.dim = dim
        self.params = params
