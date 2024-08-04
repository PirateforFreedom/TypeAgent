import os
import uuid
from typing import Any, List, Optional

import numpy as np
import requests
from data_sources.text_splitters.basetextsplitter import TokenTextSplitter
# from llama_index.core.base.embeddings import BaseEmbedding
# from llama_index.core.embeddings import BaseEmbedding
# from llama_index.core.base.embeddings.base import BaseEmbedding
# from llama_index.bridge.pydantic import PrivateAttr
# from llama_index.embeddings.base import BaseEmbedding
# from llama_index.embeddings.huggingface_utils import format_text
import tiktoken
# from llama_index.core import Document as LlamaIndexDocument
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    TypeVar,
    Union,
    cast,
)
# from llama_index.core.base.embeddings import BaseEmbedding
# from llama_index.core.node_parser import SentenceSplitter
from abc import ABC, abstractmethod
from constants import (
    EMBEDDING_TO_TOKENIZER_DEFAULT,
    EMBEDDING_TO_TOKENIZER_MAP,
    MAX_EMBEDDING_DIM,
)
import os
import warnings
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)

import openai
import tiktoken
from credentials import typeagentCredentials
from data_types import EmbeddingConfig
from utils import is_valid_url, printd
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from langchain_core.runnables.config import run_in_executor
from langchain_core.pydantic_v1 import (
    BaseModel,
    Extra,
    Field,
    SecretStr,
    root_validator,
)
from langchain_core.utils import (
    convert_to_secret_str,
    get_from_dict_or_env,
    get_pydantic_field_names,
)
class Embeddings(ABC):
    """Interface for embedding models."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous Embed search docs."""
        return await run_in_executor(None, self.embed_documents, texts)

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronous Embed query text."""
        return await run_in_executor(None, self.embed_query, text)


def _process_batched_chunked_embeddings(
    num_texts: int,
    tokens: List[Union[List[int], str]],
    batched_embeddings: List[List[float]],
    indices: List[int],
    skip_empty: bool,
) -> List[Optional[List[float]]]:
    # for each text, this is the list of embeddings (list of list of floats)
    # corresponding to the chunks of the text
    results: List[List[List[float]]] = [[] for _ in range(num_texts)]

    # for each text, this is the token length of each chunk
    # for transformers tokenization, this is the string length
    # for tiktoken, this is the number of tokens
    num_tokens_in_batch: List[List[int]] = [[] for _ in range(num_texts)]

    for i in range(len(indices)):
        if skip_empty and len(batched_embeddings[i]) == 1:
            continue
        results[indices[i]].append(batched_embeddings[i])
        num_tokens_in_batch[indices[i]].append(len(tokens[i]))

    # for each text, this is the final embedding
    embeddings: List[Optional[List[float]]] = []
    for i in range(num_texts):
        # an embedding for each chunk
        _result: List[List[float]] = results[i]

        if len(_result) == 0:
            # this will be populated with the embedding of an empty string
            # in the sync or async code calling this
            embeddings.append(None)
            continue

        elif len(_result) == 1:
            # if only one embedding was produced, use it
            embeddings.append(_result[0])
            continue

        else:
            # else we need to weighted average
            # should be same as
            # average = np.average(_result, axis=0, weights=num_tokens_in_batch[i])
            total_weight = sum(num_tokens_in_batch[i])
            average = [
                sum(
                    val * weight
                    for val, weight in zip(embedding, num_tokens_in_batch[i])
                )
                / total_weight
                for embedding in zip(*_result)
            ]

            # should be same as
            # embeddings.append((average / np.linalg.norm(average)).tolist())
            magnitude = sum(val**2 for val in average) ** 0.5
            embeddings.append([val / magnitude for val in average])

    return embeddings
class OpenAIEmbeddings(BaseModel, Embeddings):
    """OpenAI embedding models.

    To use, you should have the
    environment variable ``OPENAI_API_KEY`` set with your API key or pass it
    as a named parameter to the constructor.

    In order to use the library with Microsoft Azure endpoints, use
    AzureOpenAIEmbeddings.

    Example:
        .. code-block:: python

            from langchain_openai import OpenAIEmbeddings

            model = OpenAIEmbeddings(model="text-embedding-3-large")
    """

    client: Any = Field(default=None, exclude=True)  #: :meta private:
    async_client: Any = Field(default=None, exclude=True)  #: :meta private:
    model: str = "text-embedding-ada-002"
    dimensions: Optional[int] = None
    """The number of dimensions the resulting output embeddings should have.

    Only supported in `text-embedding-3` and later models.
    """
    # to support Azure OpenAI Service custom deployment names
    deployment: Optional[str] = model
    # TODO: Move to AzureOpenAIEmbeddings.
    openai_api_version: Optional[str] = Field(default=None, alias="api_version")
    """Automatically inferred from env var `OPENAI_API_VERSION` if not provided."""
    # to support Azure OpenAI Service custom endpoints
    openai_api_base: Optional[str] = Field(default=None, alias="base_url")
    """Base URL path for API requests, leave blank if not using a proxy or service
        emulator."""
    # to support Azure OpenAI Service custom endpoints
    openai_api_type: Optional[str] = None
    # to support explicit proxy for OpenAI
    openai_proxy: Optional[str] = None
    embedding_ctx_length: int = 8191
    """The maximum number of tokens to embed at once."""
    openai_api_key: Optional[SecretStr] = Field(default=None, alias="api_key")
    """Automatically inferred from env var `OPENAI_API_KEY` if not provided."""
    openai_organization: Optional[str] = Field(default=None, alias="organization")
    """Automatically inferred from env var `OPENAI_ORG_ID` if not provided."""
    allowed_special: Union[Literal["all"], Set[str], None] = None
    disallowed_special: Union[Literal["all"], Set[str], Sequence[str], None] = None
    chunk_size: int = 1000
    """Maximum number of texts to embed in each batch"""
    max_retries: int = 2
    """Maximum number of retries to make when generating."""
    request_timeout: Optional[Union[float, Tuple[float, float], Any]] = Field(
        default=None, alias="timeout"
    )
    """Timeout for requests to OpenAI completion API. Can be float, httpx.Timeout or
        None."""
    headers: Any = None
    tiktoken_enabled: bool = True
    """Set this to False for non-OpenAI implementations of the embeddings API, e.g.
    the `--extensions openai` extension for `text-generation-webui`"""
    tiktoken_model_name: Optional[str] = None
    """The model name to pass to tiktoken when using this class.
    Tiktoken is used to count the number of tokens in documents to constrain
    them to be under a certain limit. By default, when set to None, this will
    be the same as the embedding model name. However, there are some cases
    where you may want to use this Embedding class with a model name not
    supported by tiktoken. This can include when using Azure embeddings or
    when using one of the many model providers that expose an OpenAI-like
    API but with different models. In those cases, in order to avoid erroring
    when tiktoken is called, you can specify a model name to use here."""
    show_progress_bar: bool = False
    """Whether to show a progress bar when embedding."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""
    skip_empty: bool = False
    """Whether to skip empty strings when embedding or raise an error.
    Defaults to not skipping."""
    default_headers: Union[Mapping[str, str], None] = None
    default_query: Union[Mapping[str, object], None] = None
    # Configure a custom httpx client. See the
    # [httpx documentation](https://www.python-httpx.org/api/#client) for more details.
    retry_min_seconds: int = 4
    """Min number of seconds to wait between retries"""
    retry_max_seconds: int = 20
    """Max number of seconds to wait between retries"""
    http_client: Union[Any, None] = None
    """Optional httpx.Client. Only used for sync invocations. Must specify 
        http_async_client as well if you'd like a custom client for async invocations.
    """
    http_async_client: Union[Any, None] = None
    """Optional httpx.AsyncClient. Only used for async invocations. Must specify 
        http_client as well if you'd like a custom client for sync invocations."""
    check_embedding_ctx_length: bool = True
    """Whether to check the token length of inputs and automatically split inputs 
        longer than embedding_ctx_length."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        allow_population_by_field_name = True

    @root_validator(pre=True)
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = get_pydantic_field_names(cls)
        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name in extra:
                raise ValueError(f"Found {field_name} supplied twice.")
            if field_name not in all_required_field_names:
                warnings.warn(
                    f"""WARNING! {field_name} is not default parameter.
                    {field_name} was transferred to model_kwargs.
                    Please confirm that {field_name} is what you intended."""
                )
                extra[field_name] = values.pop(field_name)

        invalid_model_kwargs = all_required_field_names.intersection(extra.keys())
        if invalid_model_kwargs:
            raise ValueError(
                f"Parameters {invalid_model_kwargs} should be specified explicitly. "
                f"Instead they were passed in as part of `model_kwargs` parameter."
            )

        values["model_kwargs"] = extra
        return values

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        openai_api_key = get_from_dict_or_env(
            values, "openai_api_key", "OPENAI_API_KEY"
        )
        values["openai_api_key"] = (
            convert_to_secret_str(openai_api_key) if openai_api_key else None
        )
        values["openai_api_base"] = values["openai_api_base"] or os.getenv(
            "OPENAI_API_BASE"
        )
        values["openai_api_type"] = get_from_dict_or_env(
            values,
            "openai_api_type",
            "OPENAI_API_TYPE",
            default="",
        )
        values["openai_proxy"] = get_from_dict_or_env(
            values,
            "openai_proxy",
            "OPENAI_PROXY",
            default="",
        )
        if values["openai_api_type"] in ("azure", "azure_ad", "azuread"):
            default_api_version = "2023-05-15"
            # Azure OpenAI embedding models allow a maximum of 16 texts
            # at a time in each batch
            # See: https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#embeddings
            values["chunk_size"] = min(values["chunk_size"], 16)
        else:
            default_api_version = ""
        values["openai_api_version"] = get_from_dict_or_env(
            values,
            "openai_api_version",
            "OPENAI_API_VERSION",
            default=default_api_version,
        )
        # Check OPENAI_ORGANIZATION for backwards compatibility.
        values["openai_organization"] = (
            values["openai_organization"]
            or os.getenv("OPENAI_ORG_ID")
            or os.getenv("OPENAI_ORGANIZATION")
        )
        if values["openai_api_type"] in ("azure", "azure_ad", "azuread"):
            raise ValueError(
                "If you are using Azure, "
                "please use the `AzureOpenAIEmbeddings` class."
            )
        client_params = {
            "api_key": (
                values["openai_api_key"].get_secret_value()
                if values["openai_api_key"]
                else None
            ),
            "organization": values["openai_organization"],
            "base_url": values["openai_api_base"],
            "timeout": values["request_timeout"],
            "max_retries": values["max_retries"],
            "default_headers": values["default_headers"],
            "default_query": values["default_query"],
        }
        if not values.get("client"):
            sync_specific = {"http_client": values["http_client"]}
            values["client"] = openai.OpenAI(
                **client_params, **sync_specific
            ).embeddings
        if not values.get("async_client"):
            async_specific = {"http_client": values["http_async_client"]}
            values["async_client"] = openai.AsyncOpenAI(
                **client_params, **async_specific
            ).embeddings
        return values

    @property
    def _invocation_params(self) -> Dict[str, Any]:
        params: Dict = {"model": self.model, **self.model_kwargs}
        if self.dimensions is not None:
            params["dimensions"] = self.dimensions
        return params

    def _tokenize(
        self, texts: List[str], chunk_size: int
    ) -> Tuple[Iterable[int], List[Union[List[int], str]], List[int]]:
        """
        Take the input `texts` and `chunk_size` and return 3 iterables as a tuple:

        We have `batches`, where batches are sets of individual texts
        we want responses from the openai api. The length of a single batch is
        `chunk_size` texts.

        Each individual text is also split into multiple texts based on the
        `embedding_ctx_length` parameter (based on number of tokens).

        This function returns a 3-tuple of the following:

        _iter: An iterable of the starting index in `tokens` for each *batch*
        tokens: A list of tokenized texts, where each text has already been split
            into sub-texts based on the `embedding_ctx_length` parameter. In the
            case of tiktoken, this is a list of token arrays. In the case of
            HuggingFace transformers, this is a list of strings.
        indices: An iterable of the same length as `tokens` that maps each token-array
            to the index of the original text in `texts`.
        """
        tokens: List[Union[List[int], str]] = []
        indices: List[int] = []
        model_name = self.tiktoken_model_name or self.model

        # If tiktoken flag set to False
        if not self.tiktoken_enabled:
            try:
                from transformers import AutoTokenizer
            except ImportError:
                raise ValueError(
                    "Could not import transformers python package. "
                    "This is needed for OpenAIEmbeddings to work without "
                    "`tiktoken`. Please install it with `pip install transformers`. "
                )

            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=model_name
            )
            for i, text in enumerate(texts):
                # Tokenize the text using HuggingFace transformers
                tokenized: List[int] = tokenizer.encode(text, add_special_tokens=False)

                # Split tokens into chunks respecting the embedding_ctx_length
                for j in range(0, len(tokenized), self.embedding_ctx_length):
                    token_chunk: List[int] = tokenized[
                        j : j + self.embedding_ctx_length
                    ]

                    # Convert token IDs back to a string
                    chunk_text: str = tokenizer.decode(token_chunk)
                    tokens.append(chunk_text)
                    indices.append(i)
        else:
            try:
                encoding = tiktoken.encoding_for_model(model_name)
            except KeyError:
                encoding = tiktoken.get_encoding("cl100k_base")
            encoder_kwargs: Dict[str, Any] = {
                k: v
                for k, v in {
                    "allowed_special": self.allowed_special,
                    "disallowed_special": self.disallowed_special,
                }.items()
                if v is not None
            }
            for i, text in enumerate(texts):
                if self.model.endswith("001"):
                    # See: https://github.com/openai/openai-python/
                    #      issues/418#issuecomment-1525939500
                    # replace newlines, which can negatively affect performance.
                    text = text.replace("\n", " ")

                if encoder_kwargs:
                    token = encoding.encode(text, **encoder_kwargs)
                else:
                    token = encoding.encode_ordinary(text)

                # Split tokens into chunks respecting the embedding_ctx_length
                for j in range(0, len(token), self.embedding_ctx_length):
                    tokens.append(token[j : j + self.embedding_ctx_length])
                    indices.append(i)

        if self.show_progress_bar:
            try:
                from tqdm.auto import tqdm

                _iter: Iterable = tqdm(range(0, len(tokens), chunk_size))
            except ImportError:
                _iter = range(0, len(tokens), chunk_size)
        else:
            _iter = range(0, len(tokens), chunk_size)
        return _iter, tokens, indices

    # please refer to
    # https://github.com/openai/openai-cookbook/blob/main/examples/Embedding_long_inputs.ipynb
    def _get_len_safe_embeddings(
        self, texts: List[str], *, engine: str, chunk_size: Optional[int] = None
    ) -> List[List[float]]:
        """
        Generate length-safe embeddings for a list of texts.

        This method handles tokenization and embedding generation, respecting the
        set embedding context length and chunk size. It supports both tiktoken
        and HuggingFace tokenizer based on the tiktoken_enabled flag.

        Args:
            texts (List[str]): A list of texts to embed.
            engine (str): The engine or model to use for embeddings.
            chunk_size (Optional[int]): The size of chunks for processing embeddings.

        Returns:
            List[List[float]]: A list of embeddings for each input text.
        """
        _chunk_size = chunk_size or self.chunk_size
        _iter, tokens, indices = self._tokenize(texts, _chunk_size)
        batched_embeddings: List[List[float]] = []
        for i in _iter:
            response = self.client.create(
                input=tokens[i : i + _chunk_size], **self._invocation_params
            )
            if not isinstance(response, dict):
                response = response.model_dump()
            batched_embeddings.extend(r["embedding"] for r in response["data"])

        embeddings = _process_batched_chunked_embeddings(
            len(texts), tokens, batched_embeddings, indices, self.skip_empty
        )
        _cached_empty_embedding: Optional[List[float]] = None

        def empty_embedding() -> List[float]:
            nonlocal _cached_empty_embedding
            if _cached_empty_embedding is None:
                average_embedded = self.client.create(
                    input="", **self._invocation_params
                )
                if not isinstance(average_embedded, dict):
                    average_embedded = average_embedded.model_dump()
                _cached_empty_embedding = average_embedded["data"][0]["embedding"]
            return _cached_empty_embedding

        return [e if e is not None else empty_embedding() for e in embeddings]

    # please refer to
    # https://github.com/openai/openai-cookbook/blob/main/examples/Embedding_long_inputs.ipynb
    async def _aget_len_safe_embeddings(
        self, texts: List[str], *, engine: str, chunk_size: Optional[int] = None
    ) -> List[List[float]]:
        """
        Asynchronously generate length-safe embeddings for a list of texts.

        This method handles tokenization and asynchronous embedding generation,
        respecting the set embedding context length and chunk size. It supports both
        `tiktoken` and HuggingFace `tokenizer` based on the tiktoken_enabled flag.

        Args:
            texts (List[str]): A list of texts to embed.
            engine (str): The engine or model to use for embeddings.
            chunk_size (Optional[int]): The size of chunks for processing embeddings.

        Returns:
            List[List[float]]: A list of embeddings for each input text.
        """

        _chunk_size = chunk_size or self.chunk_size
        _iter, tokens, indices = self._tokenize(texts, _chunk_size)
        batched_embeddings: List[List[float]] = []
        _chunk_size = chunk_size or self.chunk_size
        for i in range(0, len(tokens), _chunk_size):
            response = await self.async_client.create(
                input=tokens[i : i + _chunk_size], **self._invocation_params
            )

            if not isinstance(response, dict):
                response = response.model_dump()
            batched_embeddings.extend(r["embedding"] for r in response["data"])

        embeddings = _process_batched_chunked_embeddings(
            len(texts), tokens, batched_embeddings, indices, self.skip_empty
        )
        _cached_empty_embedding: Optional[List[float]] = None

        async def empty_embedding() -> List[float]:
            nonlocal _cached_empty_embedding
            if _cached_empty_embedding is None:
                average_embedded = await self.async_client.create(
                    input="", **self._invocation_params
                )
                if not isinstance(average_embedded, dict):
                    average_embedded = average_embedded.model_dump()
                _cached_empty_embedding = average_embedded["data"][0]["embedding"]
            return _cached_empty_embedding

        return [e if e is not None else await empty_embedding() for e in embeddings]

    def embed_documents(
        self, texts: List[str], chunk_size: Optional[int] = 0
    ) -> List[List[float]]:
        """Call out to OpenAI's embedding endpoint for embedding search docs.

        Args:
            texts: The list of texts to embed.
            chunk_size: The chunk size of embeddings. If None, will use the chunk size
                specified by the class.

        Returns:
            List of embeddings, one for each text.
        """
        if not self.check_embedding_ctx_length:
            embeddings: List[List[float]] = []
            for text in texts:
                response = self.client.create(
                    input=text,
                    **self._invocation_params,
                )
                if not isinstance(response, dict):
                    response = response.dict()
                embeddings.extend(r["embedding"] for r in response["data"])
            return embeddings

        # NOTE: to keep things simple, we assume the list may contain texts longer
        #       than the maximum context and use length-safe embedding function.
        engine = cast(str, self.deployment)
        return self._get_len_safe_embeddings(texts, engine=engine)

    async def aembed_documents(
        self, texts: List[str], chunk_size: Optional[int] = 0
    ) -> List[List[float]]:
        """Call out to OpenAI's embedding endpoint async for embedding search docs.

        Args:
            texts: The list of texts to embed.
            chunk_size: The chunk size of embeddings. If None, will use the chunk size
                specified by the class.

        Returns:
            List of embeddings, one for each text.
        """
        if not self.check_embedding_ctx_length:
            embeddings: List[List[float]] = []
            for text in texts:
                response = await self.async_client.create(
                    input=text,
                    **self._invocation_params,
                )
                if not isinstance(response, dict):
                    response = response.dict()
                embeddings.extend(r["embedding"] for r in response["data"])
            return embeddings

        # NOTE: to keep things simple, we assume the list may contain texts longer
        #       than the maximum context and use length-safe embedding function.
        engine = cast(str, self.deployment)
        return await self._aget_len_safe_embeddings(texts, engine=engine)

    def embed_query(self, text: str) -> List[float]:
        """Call out to OpenAI's embedding endpoint for embedding query text.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        return self.embed_documents([text])[0]

    async def aembed_query(self, text: str) -> List[float]:
        """Call out to OpenAI's embedding endpoint async for embedding query text.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        embeddings = await self.aembed_documents([text])
        return embeddings[0]
class AzureOpenAIEmbeddings(OpenAIEmbeddings):
    """`Azure OpenAI` Embeddings API.

    To use, you should have the
    environment variable ``AZURE_OPENAI_API_KEY`` set with your API key or pass it
    as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain_openai import AzureOpenAIEmbeddings

            openai = AzureOpenAIEmbeddings(model="text-embedding-3-large")
    """

    azure_endpoint: Union[str, None] = None
    """Your Azure endpoint, including the resource.

        Automatically inferred from env var `AZURE_OPENAI_ENDPOINT` if not provided.

        Example: `https://example-resource.azure.openai.com/`
    """
    deployment: Optional[str] = Field(default=None, alias="azure_deployment")
    """A model deployment.

        If given sets the base client URL to include `/deployments/{azure_deployment}`.
        Note: this means you won't be able to use non-deployment endpoints.
    """
    openai_api_key: Optional[SecretStr] = Field(default=None, alias="api_key")
    """Automatically inferred from env var `AZURE_OPENAI_API_KEY` if not provided."""
    azure_ad_token: Optional[SecretStr] = None
    """Your Azure Active Directory token.

        Automatically inferred from env var `AZURE_OPENAI_AD_TOKEN` if not provided.

        For more:
        https://www.microsoft.com/en-us/security/business/identity-access/microsoft-entra-id.
    """
    azure_ad_token_provider: Union[Callable[[], str], None] = None
    """A function that returns an Azure Active Directory token.

        Will be invoked on every request.
    """
    openai_api_version: Optional[str] = Field(default=None, alias="api_version")
    """Automatically inferred from env var `OPENAI_API_VERSION` if not provided."""
    validate_base_url: bool = True
    chunk_size: int = 2048
    """Maximum number of texts to embed in each batch"""

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        # Check OPENAI_KEY for backwards compatibility.
        # TODO: Remove OPENAI_API_KEY support to avoid possible conflict when using
        # other forms of azure credentials.
        openai_api_key = (
            values["openai_api_key"]
            or os.getenv("AZURE_OPENAI_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        values["openai_api_key"] = (
            convert_to_secret_str(openai_api_key) if openai_api_key else None
        )
        values["openai_api_base"] = values["openai_api_base"] or os.getenv(
            "OPENAI_API_BASE"
        )
        values["openai_api_version"] = values["openai_api_version"] or os.getenv(
            "OPENAI_API_VERSION", default="2023-05-15"
        )
        values["openai_api_type"] = get_from_dict_or_env(
            values, "openai_api_type", "OPENAI_API_TYPE", default="azure"
        )
        values["openai_organization"] = (
            values["openai_organization"]
            or os.getenv("OPENAI_ORG_ID")
            or os.getenv("OPENAI_ORGANIZATION")
        )
        values["openai_proxy"] = get_from_dict_or_env(
            values,
            "openai_proxy",
            "OPENAI_PROXY",
            default="",
        )
        values["azure_endpoint"] = values["azure_endpoint"] or os.getenv(
            "AZURE_OPENAI_ENDPOINT"
        )
        azure_ad_token = values["azure_ad_token"] or os.getenv("AZURE_OPENAI_AD_TOKEN")
        values["azure_ad_token"] = (
            convert_to_secret_str(azure_ad_token) if azure_ad_token else None
        )
        # For backwards compatibility. Before openai v1, no distinction was made
        # between azure_endpoint and base_url (openai_api_base).
        openai_api_base = values["openai_api_base"]
        if openai_api_base and values["validate_base_url"]:
            if "/openai" not in openai_api_base:
                values["openai_api_base"] += "/openai"
                raise ValueError(
                    "As of openai>=1.0.0, Azure endpoints should be specified via "
                    "the `azure_endpoint` param not `openai_api_base` "
                    "(or alias `base_url`). "
                )
            if values["deployment"]:
                raise ValueError(
                    "As of openai>=1.0.0, if `deployment` (or alias "
                    "`azure_deployment`) is specified then "
                    "`openai_api_base` (or alias `base_url`) should not be. "
                    "Instead use `deployment` (or alias `azure_deployment`) "
                    "and `azure_endpoint`."
                )
        client_params = {
            "api_version": values["openai_api_version"],
            "azure_endpoint": values["azure_endpoint"],
            "azure_deployment": values["deployment"],
            "api_key": (
                values["openai_api_key"].get_secret_value()
                if values["openai_api_key"]
                else None
            ),
            "azure_ad_token": (
                values["azure_ad_token"].get_secret_value()
                if values["azure_ad_token"]
                else None
            ),
            "azure_ad_token_provider": values["azure_ad_token_provider"],
            "organization": values["openai_organization"],
            "base_url": values["openai_api_base"],
            "timeout": values["request_timeout"],
            "max_retries": values["max_retries"],
            "default_headers": values["default_headers"],
            "default_query": values["default_query"],
        }
        if not values.get("client"):
            sync_specific = {"http_client": values["http_client"]}
            values["client"] = openai.AzureOpenAI(
                **client_params, **sync_specific
            ).embeddings
        if not values.get("async_client"):
            async_specific = {"http_client": values["http_async_client"]}
            values["async_client"] = openai.AsyncAzureOpenAI(
                **client_params, **async_specific
            ).embeddings
        return values

    @property
    def _llm_type(self) -> str:
        return "azure-openai-chat"


class OllamaEmbeddings(BaseModel, Embeddings):
    """Ollama locally runs large language models.

    To use, follow the instructions at https://ollama.ai/.

    Example:
        .. code-block:: python

            from langchain_community.embeddings import OllamaEmbeddings
            ollama_emb = OllamaEmbeddings(
                model="llama:7b",
            )
            r1 = ollama_emb.embed_documents(
                [
                    "Alpha is the first letter of Greek alphabet",
                    "Beta is the second letter of Greek alphabet",
                ]
            )
            r2 = ollama_emb.embed_query(
                "What is the second letter of Greek alphabet"
            )

    """

    base_url: str = "http://localhost:11434"
    """Base url the model is hosted under."""
    model: str = "llama2"
    """Model name to use."""

    embed_instruction: str = "passage: "
    """Instruction used to embed documents."""
    query_instruction: str = "query: "
    """Instruction used to embed the query."""

    mirostat: Optional[int] = None
    """Enable Mirostat sampling for controlling perplexity.
    (default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)"""

    mirostat_eta: Optional[float] = None
    """Influences how quickly the algorithm responds to feedback
    from the generated text. A lower learning rate will result in
    slower adjustments, while a higher learning rate will make
    the algorithm more responsive. (Default: 0.1)"""

    mirostat_tau: Optional[float] = None
    """Controls the balance between coherence and diversity
    of the output. A lower value will result in more focused and
    coherent text. (Default: 5.0)"""

    num_ctx: Optional[int] = None
    """Sets the size of the context window used to generate the
    next token. (Default: 2048)	"""

    num_gpu: Optional[int] = None
    """The number of GPUs to use. On macOS it defaults to 1 to
    enable metal support, 0 to disable."""

    num_thread: Optional[int] = None
    """Sets the number of threads to use during computation.
    By default, Ollama will detect this for optimal performance.
    It is recommended to set this value to the number of physical
    CPU cores your system has (as opposed to the logical number of cores)."""

    repeat_last_n: Optional[int] = None
    """Sets how far back for the model to look back to prevent
    repetition. (Default: 64, 0 = disabled, -1 = num_ctx)"""

    repeat_penalty: Optional[float] = None
    """Sets how strongly to penalize repetitions. A higher value (e.g., 1.5)
    will penalize repetitions more strongly, while a lower value (e.g., 0.9)
    will be more lenient. (Default: 1.1)"""

    temperature: Optional[float] = None
    """The temperature of the model. Increasing the temperature will
    make the model answer more creatively. (Default: 0.8)"""

    stop: Optional[List[str]] = None
    """Sets the stop tokens to use."""

    tfs_z: Optional[float] = None
    """Tail free sampling is used to reduce the impact of less probable
    tokens from the output. A higher value (e.g., 2.0) will reduce the
    impact more, while a value of 1.0 disables this setting. (default: 1)"""

    top_k: Optional[int] = None
    """Reduces the probability of generating nonsense. A higher value (e.g. 100)
    will give more diverse answers, while a lower value (e.g. 10)
    will be more conservative. (Default: 40)"""

    top_p: Optional[float] = None
    """Works together with top-k. A higher value (e.g., 0.95) will lead
    to more diverse text, while a lower value (e.g., 0.5) will
    generate more focused and conservative text. (Default: 0.9)"""

    show_progress: bool = False
    """Whether to show a tqdm progress bar. Must have `tqdm` installed."""

    headers: Optional[dict] = None
    """Additional headers to pass to endpoint (e.g. Authorization, Referer).
    This is useful when Ollama is hosted on cloud services that require
    tokens for authentication.
    """

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Ollama."""
        return {
            "model": self.model,
            "options": {
                "mirostat": self.mirostat,
                "mirostat_eta": self.mirostat_eta,
                "mirostat_tau": self.mirostat_tau,
                "num_ctx": self.num_ctx,
                "num_gpu": self.num_gpu,
                "num_thread": self.num_thread,
                "repeat_last_n": self.repeat_last_n,
                "repeat_penalty": self.repeat_penalty,
                "temperature": self.temperature,
                "stop": self.stop,
                "tfs_z": self.tfs_z,
                "top_k": self.top_k,
                "top_p": self.top_p,
            },
        }

    model_kwargs: Optional[dict] = None
    """Other model keyword args"""

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {**{"model": self.model}, **self._default_params}

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def _process_emb_response(self, input: str) -> List[float]:
        """Process a response from the API.

        Args:
            response: The response from the API.

        Returns:
            The response as a dictionary.
        """
        headers = {
            "Content-Type": "application/json",
            **(self.headers or {}),
        }

        try:
            res = requests.post(
                f"{self.base_url}/api/embeddings",
                headers=headers,
                json={"model": self.model, "prompt": input, **self._default_params},
            )
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error raised by inference endpoint: {e}")

        if res.status_code != 200:
            raise ValueError(
                "Error raised by inference API HTTP code: %s, %s"
                % (res.status_code, res.text)
            )
        try:
            t = res.json()
            return t["embedding"]
        except requests.exceptions.JSONDecodeError as e:
            raise ValueError(
                f"Error raised by inference API: {e}.\nResponse: {res.text}"
            )

    def _embed(self, input: List[str]) -> List[List[float]]:
        if self.show_progress:
            try:
                from tqdm import tqdm

                iter_ = tqdm(input, desc="OllamaEmbeddings")
            except ImportError:
                print(
                    "Unable to show progress bar because tqdm could not be imported. "
                    "Please install with `pip install tqdm`."
                )
                iter_ = input
        else:
            iter_ = input
        return [self._process_emb_response(prompt) for prompt in iter_]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using an Ollama deployed embedding model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        instruction_pairs = [f"{self.embed_instruction}{text}" for text in texts]
        embeddings = self._embed(instruction_pairs)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using a Ollama deployed embedding model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        instruction_pair = f"{self.query_instruction}{text}"
        embedding = self._embed([instruction_pair])[0]
        return embedding
# def parse_and_chunk_text(text: str, chunk_size: int) -> List[str]:
#     parser = SentenceSplitter(chunk_size=chunk_size)
#     llama_index_docs = [LlamaIndexDocument(text=text)]
#     nodes = parser.get_nodes_from_documents(llama_index_docs)
#     return [n.text for n in nodes]

def generate_passages( documents: str, chunk_size: int = 1024) -> Iterator[Tuple[str, Dict]]:  # -> Iterator[Passage]:
        # use llama index to run embeddings code
        # from llama_index.core.node_parser import SentenceSplitter
        # from llama_index.core.node_parser import TokenTextSplitter

        parser = TokenTextSplitter(chunk_size=chunk_size)
        # for document in documents:
        passages_from_docs=parser.split_text(text=documents)
            # llama_index_docs = [LlamaIndexDocument(text=document.text, metadata=document.metadata)]
            # nodes = parser.get_nodes_from_documents(llama_index_docs)
        for nodedocs in passages_from_docs:
                # passage = Passage(
                #    text=node.text,
                #    doc_id=document.id,
                # )
            yield nodedocs, None

def truncate_text(text: str, max_length: int, encoding) -> str:
    # truncate the text based on max_length and encoding
    encoded_text = encoding.encode(text)[:max_length]
    return encoding.decode(encoded_text)


# def check_and_split_text(text: str, embedding_model: str) -> List[str]:
#     """Split text into chunks of max_length tokens or less"""

#     if embedding_model in EMBEDDING_TO_TOKENIZER_MAP:
#         encoding = tiktoken.get_encoding(EMBEDDING_TO_TOKENIZER_MAP[embedding_model])
#     else:
#         print(f"Warning: couldn't find tokenizer for model {embedding_model}, using default tokenizer {EMBEDDING_TO_TOKENIZER_DEFAULT}")
#         encoding = tiktoken.get_encoding(EMBEDDING_TO_TOKENIZER_DEFAULT)

#     num_tokens = len(encoding.encode(text))

#     # determine max length
#     if hasattr(encoding, "max_length"):
#         # TODO(fix) this is broken
#         max_length = encoding.max_length
#     else:
#         # TODO: figure out the real number
#         printd(f"Warning: couldn't find max_length for tokenizer {embedding_model}, using default max_length 8191")
#         max_length = 8191

#     # truncate text if too long
#     if num_tokens > max_length:
#         print(f"Warning: text is too long ({num_tokens} tokens), truncating to {max_length} tokens.")
#         # First, apply any necessary formatting
#         formatted_text = format_text(text, embedding_model)
#         # Then truncate
#         text = truncate_text(formatted_text, max_length, encoding)

#     return [text]


class EmbeddingEndpoint:
    """Implementation for OpenAI compatible endpoint"""

    # """ Based off llama index https://github.com/run-llama/llama_index/blob/a98bdb8ecee513dc2e880f56674e7fd157d1dc3a/llama_index/embeddings/text_embeddings_inference.py """

    # _user: str = PrivateAttr()
    # _timeout: float = PrivateAttr()
    # _base_url: str = PrivateAttr()

    def __init__(
        self,
        model: str,
        base_url: str,
        user: str,
        timeout: float = 60.0,
        **kwargs: Any,
    ):
        if not is_valid_url(base_url):
            raise ValueError(
                f"Embeddings endpoint was provided an invalid URL (set to: '{base_url}'). Make sure embedding_endpoint is set correctly in your typeagent config."
            )
        self.model_name = model
        self._user = user
        self._base_url = base_url
        self._timeout = timeout

    def _call_api(self, text: str) -> List[float]:
        if not is_valid_url(self._base_url):
            raise ValueError(
                f"Embeddings endpoint does not have a valid URL (set to: '{self._base_url}'). Make sure embedding_endpoint is set correctly in your typeagent config."
            )
        import httpx

        headers = {"Content-Type": "application/json"}
        json_data = {"input": text, "model": self.model_name, "user": self._user}

        with httpx.Client() as client:
            response = client.post(
                f"{self._base_url}/embeddings",
                headers=headers,
                json=json_data,
                timeout=self._timeout,
            )

        response_json = response.json()

        if isinstance(response_json, list):
            # embedding directly in response
            embedding = response_json
        elif isinstance(response_json, dict):
            # TEI embedding packaged inside openai-style response
            try:
                embedding = response_json["data"][0]["embedding"]
            except (KeyError, IndexError):
                raise TypeError(f"Got back an unexpected payload from text embedding function, response=\n{response_json}")
        else:
            # unknown response, can't parse
            raise TypeError(f"Got back an unexpected payload from text embedding function, response=\n{response_json}")

        return embedding

    def get_text_embedding(self, text: str) -> List[float]:
        return self._call_api(text)


def default_embedding_model():
    # default to hugging face model running local
    # warning: this is a terrible model
    # from llama_index.embeddings.huggingface import HuggingFaceEmbedding
  
    # os.environ["TOKENIZERS_PARALLELISM"] = "False"
    # model = "BAAI/bge-small-en-v1.5"
    # return HuggingFaceEmbedding(model_name=model)
    return "no implenment"


def query_embedding(embedding_model, query_text: str):
    """Generate padded embedding for querying database"""
    query_vec = embedding_model.get_text_embedding(query_text)
    query_vec = np.array(query_vec)
    query_vec = np.pad(query_vec, (0, MAX_EMBEDDING_DIM - query_vec.shape[0]), mode="constant").tolist()
    return query_vec


def embedding_model(config: EmbeddingConfig, user_id: Optional[uuid.UUID] = None):
    """Return LlamaIndex embedding model to use for embeddings"""

    endpoint_type = config.embedding_endpoint_type

    # TODO refactor to pass credentials through args
    credentials = typeagentCredentials.load()

    if endpoint_type == "openai":
        # assert credentials.openai_key is not None
        # from llama_index.embeddings.openai import OpenAIEmbedding
        # from langchain_openai import OpenAIEmbeddings
        # additional_kwargs = {"user_id": user_id} if user_id else {}
        model = OpenAIEmbeddings(
            openai_api_base=config.embedding_endpoint,
            openai_api_key=credentials.openai_key,
            model= 'text-embedding-ada-002',
        )
        return model

    elif endpoint_type == "azure":
       
        deployment = credentials.azure_embedding_deployment if credentials.azure_embedding_deployment is not None else model
    
        return AzureOpenAIEmbeddings(
            azure_deployment=deployment,
            api_key=credentials.azure_embedding_api_key,
            azure_endpoint=credentials.azure_embedding_endpoint,
            api_version=credentials.azure_embedding_version,
        )

    elif endpoint_type == "hugging-face":
        return EmbeddingEndpoint(
            model=config.embedding_model,
            base_url=config.embedding_endpoint,
            user=user_id,
        )
    elif endpoint_type == "ollama":

        # from llama_index.embeddings.ollama import OllamaEmbedding

        ollama_additional_kwargs = {}
        callback_manager = None

        model =OllamaEmbeddings(
            model_name=config.embedding_model,
            base_url=config.embedding_endpoint,
            ollama_additional_kwargs=ollama_additional_kwargs or {},
            callback_manager=callback_manager or None,
        )
        return model

    else:
        return default_embedding_model()
