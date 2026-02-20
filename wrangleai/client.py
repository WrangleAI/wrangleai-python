import os
import json
import httpx
import logging
from typing import Optional, List, Union, Generator, Any, Dict, overload, BinaryIO
from .types import (
    WrangleObject, WrangleModel, SLMConfig,
    ChatCompletion, ChatCompletionChunk, ModelsListResponse,
    FileObject, FileDeleted, FileListResponse,
    VectorStore, VectorStoreDeleted, VectorStoreListResponse,
    VectorStoreFile, VectorStoreFileDeleted, VectorStoreFileListResponse,
    VectorStoreSearchResponse, SustainabilityReport
)
from .exceptions import (
    WrangleError, AuthenticationError, RateLimitError,
    BadRequestError, APIError, APIConnectionError, PermissionDeniedError,
    NotFoundError, UnprocessableEntityError, _make_status_error
)

# Setup logger
logger = logging.getLogger("wrangleai")

class WrangleAI:
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        # base_url: str = "https://gateway.wrangleai.com/v1",
        base_url: str = "https://staging-gateway.wrangleai.com/v1",
        # base_url: str = "https://bd1851h1-8080.uks1.devtunnels.ms/v1",
        rag_base_url: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 0
    ):
        """
        Initialize the Wrangle AI Client.
        
        Args:
            api_key: Your Wrangle AI API Key. Defaults to env var WRANGLE_API_KEY.
            base_url: The API endpoint.
            rag_base_url: The RAG API endpoint (files, vector stores). Auto-detected if not provided.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retries for failed requests (default: 0 for backward compatibility).
                         Retries are performed for 408, 429, 500, 502, 503, 504 status codes with exponential backoff.
        """
        self.api_key = api_key or os.environ.get("WRANGLE_API_KEY")
        if not self.api_key:
            raise ValueError("The WrangleAI client requires an api_key argument or WRANGLE_API_KEY environment variable.")

        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        
        # Auto-detect RAG base URL (port 8085) if not provided
        if rag_base_url:
            self.rag_base_url = rag_base_url.rstrip("/")
        else:
            # Replace port 8080 with 8085 for RAG endpoints
            self.rag_base_url = self.base_url.replace(":8080", ":8085")
        
        self._last_request_id: Optional[str] = None

        self._client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            timeout=timeout
        )
        
        # RAG client for files and vector stores (port 8085)
        self._rag_client = httpx.Client(
            base_url=self.rag_base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}"
            },
            timeout=timeout
        )

        # Initialize Namespaces
        self.chat = Chat(self)
        self.models = Models(self)
        self.usage = Usage(self)
        self.cost = Cost(self)
        self.keys = Keys(self)
        self.files = Files(self)
        self.vector_stores = VectorStores(self)
        self.sustainability = Sustainability(self)

    def with_options(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None
    ) -> "WrangleAI":
        """
        Create a new client instance with modified configuration.
        
        Args:
            api_key: Override API key
            base_url: Override base URL
            timeout: Override default timeout
            max_retries: Override max retry attempts
        
        Returns:
            New WrangleAI client instance with updated options
        
        Example:
            ```python
            client = WrangleAI(api_key="key1")
            
            # Create a new client with different settings
            custom_client = client.with_options(
                timeout=120.0,
                max_retries=5
            )
            ```
        """
        return WrangleAI(
            api_key=api_key or self.api_key,
            base_url=base_url or self.base_url,
            timeout=timeout if timeout is not None else 60.0,
            max_retries=max_retries if max_retries is not None else self.max_retries
        )

    def close(self):
        """Close the underlying HTTP connections."""
        self._client.close()
        self._rag_client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _request(self, method: str, path: str, **kwargs) -> Any:
        """Make request to main server with retry logic."""
        import time
        
        retries = 0
        max_retries = self.max_retries
        
        # Log request details (debug level)
        logger.debug(f"Making {method} request to {path}")
        if "json" in kwargs:
            logger.debug(f"Request body: {json.dumps(kwargs['json'], indent=2)}")
        
        while True:
            try:
                response = self._client.request(method, path, **kwargs)
                response.raise_for_status()
                
                # Store request ID for later retrieval if needed
                self._last_request_id = response.headers.get("x-request-id")
                
                # Log response details (debug level)
                logger.debug(f"Response status: {response.status_code}")
                logger.debug(f"Response headers: {dict(response.headers)}")
                
                result = response.json()
                logger.debug(f"Response body: {json.dumps(result, indent=2)}")
                
                return result
            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code
                
                # Parse error message
                try:
                    err_body = e.response.json()
                    if isinstance(err_body.get("error"), dict):
                        msg = err_body["error"].get("message", str(e))
                    else:
                        msg = err_body.get("error") or str(e)
                except Exception:
                    msg = str(e)
                    err_body = None
                
                # Check if we should retry (408, 429, 500, 502, 503, 504)
                should_retry = status_code in {408, 429, 500, 502, 503, 504}
                
                if should_retry and retries < max_retries:
                    retries += 1
                    # Exponential backoff: 1s, 2s, 4s, 8s... (capped at 60s)
                    wait_time = min(2 ** (retries - 1), 60)
                    logger.warning(f"Request failed with status {status_code}, retrying in {wait_time}s (attempt {retries}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                
                # Log error
                logger.error(f"Request failed with status {status_code}: {msg}")
                
                # Use centralized error mapping
                raise _make_status_error(status_code, err_body, msg)
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                logger.error(f"Connection error: {str(e)}")
                raise APIConnectionError(f"Connection error: {str(e)}")
    
    def _rag_request(self, method: str, path: str, **kwargs) -> Any:
        """Make requests to RAG server (port 8085) with retry logic."""
        import time
        
        retries = 0
        max_retries = self.max_retries
        
        while True:
            try:
                response = self._rag_client.request(method, path, **kwargs)
                response.raise_for_status()
                self._last_request_id = response.headers.get("x-request-id")
                return response.json()
            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code
                
                try:
                    err_body = e.response.json()
                    if isinstance(err_body.get("error"), dict):
                        msg = err_body["error"].get("message", str(e))
                    else:
                        msg = err_body.get("error") or str(e)
                except Exception:
                    msg = str(e)
                    err_body = None
                
                # Check if we should retry (408, 429, 500, 502, 503, 504)
                should_retry = status_code in {408, 429, 500, 502, 503, 504}
                
                if should_retry and retries < max_retries:
                    retries += 1
                    # Exponential backoff: 1s, 2s, 4s, 8s...
                    wait_time = min(2 ** (retries - 1), 60)
                    time.sleep(wait_time)
                    continue
                
                # Use centralized error mapping
                raise _make_status_error(status_code, err_body, msg)
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                raise APIConnectionError(f"Connection error: {str(e)}")

# --- Chat Namespace ---
class Chat:
    def __init__(self, client: WrangleAI):
        self.completions = Completions(client)

class Completions:
    def __init__(self, client: WrangleAI):
        self._client = client

    @overload
    def create(
        self,
        *,
        messages: List[Dict[str, Any]],
        model: WrangleModel = "auto",
        stream: bool = False,
        temperature: Optional[float] = None,
        slm: Optional[SLMConfig] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        legacy_response: bool = False,
        **kwargs
    ) -> Union[ChatCompletion, WrangleObject]:
        ...
    
    @overload
    def create(
        self,
        *,
        messages: List[Dict[str, Any]],
        model: WrangleModel = "auto",
        stream: bool = True,
        temperature: Optional[float] = None,
        slm: Optional[SLMConfig] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        legacy_response: bool = False,
        **kwargs
    ) -> Union[Generator[ChatCompletionChunk, None, None], Generator[WrangleObject, None, None]]:
        ...
    
    def create(
        self,
        *,
        messages: List[Dict[str, Any]],
        model: WrangleModel = "auto",
        stream: bool = False,
        temperature: Optional[float] = None,
        slm: Optional[SLMConfig] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        legacy_response: bool = False,
        timeout: Optional[float] = None,
        extra_headers: Optional[dict] = None,
        **kwargs
    ) -> Union[ChatCompletion, WrangleObject, Generator[ChatCompletionChunk, None, None], Generator[WrangleObject, None, None]]:
        """
        Create a chat completion.
        
        Args:
            messages: A list of messages comprising the conversation.
            model: ID of the model to use (e.g. "auto", "gpt-4o").
            stream: If True, returns an iterator of chunks.
            temperature: Sampling temperature.
            slm: Efficiency-first routing configuration.
            tools: Tool definitions for function calling.
            tool_choice: Control which tool is called.
            legacy_response: If True, returns WrangleObject instead of Pydantic models.
            timeout: Optional per-request timeout override (seconds).
            extra_headers: Optional additional headers to include in the request.
            **kwargs: Additional parameters.
        
        Returns:
            ChatCompletion or Generator[ChatCompletionChunk] (or legacy WrangleObject equivalents).
        """
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **kwargs
        }
        if slm:
            payload["slm"] = slm
        if temperature is not None:
            payload["temperature"] = temperature
        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice

        # Prepare request kwargs
        request_kwargs = {"json": payload}
        if timeout is not None:
            request_kwargs["timeout"] = timeout
        if extra_headers:
            request_kwargs["headers"] = extra_headers

        if stream:
            return self._stream_request(payload, legacy_response, timeout=timeout, extra_headers=extra_headers)
        else:
            data = self._client._request("POST", "/chat/completions", **request_kwargs)
            if legacy_response:
                return WrangleObject(data)
            completion = ChatCompletion.model_validate(data)
            # Set request ID from last request
            completion.request_id = self._client._last_request_id
            return completion

    def _stream_request(self, payload, legacy_response: bool = False, timeout: Optional[float] = None, extra_headers: Optional[dict] = None):
        """Stream SSE responses with Python 3.13 compatibility."""
        request_kwargs = {"json": payload}
        if timeout is not None:
            request_kwargs["timeout"] = timeout
        if extra_headers:
            request_kwargs["headers"] = extra_headers
            
        with self._client._client.stream("POST", "/chat/completions", **request_kwargs) as response:
            # Capture request ID from headers
            request_id = response.headers.get("x-request-id")
            
            if response.status_code != 200:
                content = response.read().decode('utf-8')
                try:
                    err_json = json.loads(content)
                    msg = err_json.get("error", {}).get("message") or content
                    err_body = err_json
                except Exception:
                    msg = content
                    err_body = None
                
                # Use centralized error mapping
                raise _make_status_error(response.status_code, err_body, msg)

            try:
                for line in response.iter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data = line[6:]
                        if data.strip() == "[DONE]":
                            return
                        try:
                            chunk = json.loads(data)
                            if legacy_response:
                                yield WrangleObject(chunk)
                            else:
                                chunk_obj = ChatCompletionChunk.model_validate(chunk)
                                chunk_obj.request_id = request_id
                                yield chunk_obj
                        except json.JSONDecodeError:
                            pass
            except GeneratorExit:
                # Handle cleanup when generator is closed early
                return

# --- Models Namespace ---
class Models:
    def __init__(self, client: WrangleAI):
        self._client = client

    def list(self) -> ModelsListResponse:
        """
        Lists the currently available models.
        
        Returns:
            ModelsListResponse: List of available models
        """
        data = self._client._request("GET", "/models")
        return ModelsListResponse.model_validate(data)

# --- Usage Namespace ---
class Usage:
    def __init__(self, client: WrangleAI):
        self._client = client

    def retrieve(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> WrangleObject:
        params = {}
        if start_date:
            params["startDate"] = start_date
        if end_date:
            params["endDate"] = end_date

        data = self._client._request("GET", "/usage", params=params)
        return WrangleObject(data)

    def retrieve_by_model(self, model: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> WrangleObject:
        params = {"model": model}
        if start_date:
            params["startDate"] = start_date
        if end_date:
            params["endDate"] = end_date

        data = self._client._request("GET", "/usage/model", params=params)
        return WrangleObject(data)

# --- Cost Namespace ---
class Cost:
    def __init__(self, client: WrangleAI):
        self._client = client

    def retrieve(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> WrangleObject:
        params = {}
        if start_date:
            params["startDate"] = start_date
        if end_date:
            params["endDate"] = end_date

        data = self._client._request("GET", "/cost", params=params)
        return WrangleObject(data)

# --- Keys Namespace ---
class Keys:
    def __init__(self, client: WrangleAI):
        self._client = client

    def verify(self) -> WrangleObject:
        # The keys/verify endpoint requires X-API-Key header instead of Bearer token
        data = self._client._request(
            "GET", 
            "/keys/verify",
            headers={"X-API-Key": self._client.api_key}
        )
        return WrangleObject(data)


# --- Sustainability Namespace ---
class Sustainability:
    def __init__(self, client: WrangleAI):
        self._client = client

    def retrieve(
        self, 
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> SustainabilityReport:
        """
        Retrieve sustainability metrics (energy consumption and carbon emissions).
        
        Args:
            start_date: Start date for the report (ISO format: YYYY-MM-DD)
            end_date: End date for the report (ISO format: YYYY-MM-DD)
        
        Returns:
            SustainabilityReport with emissions data and equivalents
        """
        params = {}
        if start_date:
            params["startDate"] = start_date
        if end_date:
            params["endDate"] = end_date
        
        data = self._client._request("GET", "/sustainability", params=params)
        return SustainabilityReport.model_validate(data)


# --- Files Namespace ---
class Files:
    def __init__(self, client: WrangleAI):
        self._client = client

    def create(
        self, 
        file: BinaryIO,
        purpose: str = "assistants",
        filename: Optional[str] = None,
        *,
        timeout: Optional[float] = None,
        extra_headers: Optional[dict] = None
    ) -> FileObject:
        """
        Upload a file to WrangleAI.
        
        Args:
            file: File object opened in binary mode
            purpose: The intended purpose of the file (default: "assistants")
            filename: Optional filename with extension (e.g., 'document.pdf')
            timeout: Optional per-request timeout override (seconds)
            extra_headers: Optional additional headers to include in the request
        
        Returns:
            FileObject with file metadata
        """
        import time
        
        files = {"file": (filename, file) if filename else file}
        data = {"purpose": purpose}
        
        # Prepare request kwargs
        request_kwargs = {"files": files, "data": data}
        if timeout is not None:
            request_kwargs["timeout"] = timeout
        if extra_headers:
            # Merge with existing headers
            request_kwargs["headers"] = extra_headers
        
        # Implement retry logic with error handling
        retries = 0
        max_retries = self._client.max_retries
        
        while True:
            try:
                response = self._client._rag_client.post("/files", **request_kwargs)
                response.raise_for_status()
                return FileObject(**response.json())
            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code
                
                try:
                    err_body = e.response.json()
                    if isinstance(err_body.get("error"), dict):
                        msg = err_body["error"].get("message", str(e))
                    else:
                        msg = err_body.get("error") or str(e)
                except Exception:
                    msg = str(e)
                    err_body = None
                
                # Check if we should retry
                should_retry = status_code in {408, 429, 500, 502, 503, 504}
                
                if should_retry and retries < max_retries:
                    retries += 1
                    wait_time = min(2 ** (retries - 1), 60)
                    time.sleep(wait_time)
                    continue
                
                # Use centralized error mapping
                raise _make_status_error(status_code, err_body, msg)
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                raise APIConnectionError(f"Connection error: {str(e)}")

    def list(
        self,
        purpose: Optional[str] = None,
        limit: Optional[int] = None,
        order: Optional[str] = None,
        after: Optional[str] = None
    ) -> FileListResponse:
        """
        List files.
        
        Args:
            purpose: Filter by file purpose
            limit: Number of files to return (1-10000)
            order: Sort order ('asc' or 'desc')
            after: Cursor for pagination
        
        Returns:
            FileListResponse with list of files
        """
        params = {}
        if purpose:
            params["purpose"] = purpose
        if limit:
            params["limit"] = limit
        if order:
            params["order"] = order
        if after:
            params["after"] = after
        
        data = self._client._rag_request("GET", "/files", params=params)
        return FileListResponse(**data)

    def retrieve(self, file_id: str) -> FileObject:
        """
        Get file metadata.
        
        Args:
            file_id: The ID of the file
        
        Returns:
            FileObject with file metadata
        """
        data = self._client._rag_request("GET", f"/files/{file_id}")
        return FileObject(**data)

    def delete(self, file_id: str) -> FileDeleted:
        """
        Delete a file.
        
        Args:
            file_id: The ID of the file to delete
        
        Returns:
            FileDeleted confirmation
        """
        data = self._client._rag_request("DELETE", f"/files/{file_id}")
        return FileDeleted(**data)


# --- Vector Store Files Namespace ---
class VectorStoreFiles:
    def __init__(self, client: WrangleAI):
        self._client = client

    def create(
        self,
        vector_store_id: str,
        file_id: str,
        attributes: Optional[Dict[str, Union[str, int, bool]]] = None,
        chunking_strategy: Optional[Dict[str, Any]] = None
    ) -> VectorStoreFile:
        """
        Add a file to a vector store.
        
        Args:
            vector_store_id: The ID of the vector store
            file_id: The ID of the file to add
            attributes: Optional metadata key-value pairs
            chunking_strategy: Optional chunking configuration
        
        Returns:
            VectorStoreFile object
        """
        payload = {"file_id": file_id}
        if attributes:
            payload["attributes"] = attributes
        if chunking_strategy:
            payload["chunking_strategy"] = chunking_strategy
        
        data = self._client._rag_request(
            "POST",
            f"/vector_stores/{vector_store_id}/files",
            json=payload
        )
        return VectorStoreFile(**data)

    def list(
        self,
        vector_store_id: str,
        limit: Optional[int] = None,
        order: Optional[str] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
        filter: Optional[str] = None
    ) -> VectorStoreFileListResponse:
        """
        List files in a vector store.
        
        Args:
            vector_store_id: The ID of the vector store
            limit: Number of files to return (1-100)
            order: Sort order ('asc' or 'desc')
            after: Cursor for pagination
            before: Cursor for pagination
            filter: Filter by status ('in_progress', 'completed', 'failed', 'cancelled')
        
        Returns:
            VectorStoreFileListResponse with list of files
        """
        params = {}
        if limit:
            params["limit"] = limit
        if order:
            params["order"] = order
        if after:
            params["after"] = after
        if before:
            params["before"] = before
        if filter:
            params["filter"] = filter
        
        data = self._client._rag_request(
            "GET",
            f"/vector_stores/{vector_store_id}/files",
            params=params
        )
        return VectorStoreFileListResponse(**data)

    def retrieve(
        self,
        vector_store_id: str,
        file_id: str
    ) -> VectorStoreFile:
        """
        Get a vector store file.
        
        Args:
            vector_store_id: The ID of the vector store
            file_id: The ID of the file
        
        Returns:
            VectorStoreFile object
        """
        data = self._client._rag_request(
            "GET",
            f"/vector_stores/{vector_store_id}/files/{file_id}"
        )
        return VectorStoreFile(**data)

    def update(
        self,
        vector_store_id: str,
        file_id: str,
        attributes: Dict[str, Union[str, int, bool]]
    ) -> VectorStoreFile:
        """
        Update file attributes in a vector store.
        
        Args:
            vector_store_id: The ID of the vector store
            file_id: The ID of the file
            attributes: Metadata key-value pairs to update
        
        Returns:
            VectorStoreFile object with updated attributes
        """
        data = self._client._rag_request(
            "POST",
            f"/vector_stores/{vector_store_id}/files/{file_id}",
            json={"attributes": attributes}
        )
        return VectorStoreFile(**data)

    def delete(
        self,
        vector_store_id: str,
        file_id: str
    ) -> VectorStoreFileDeleted:
        """
        Remove a file from a vector store.
        
        Args:
            vector_store_id: The ID of the vector store
            file_id: The ID of the file to remove
        
        Returns:
            VectorStoreFileDeleted confirmation
        """
        data = self._client._rag_request(
            "DELETE",
            f"/vector_stores/{vector_store_id}/files/{file_id}"
        )
        return VectorStoreFileDeleted(**data)


# --- Vector Stores Namespace ---
class VectorStores:
    def __init__(self, client: WrangleAI):
        self._client = client
        self.files = VectorStoreFiles(client)

    def create(
        self,
        name: Optional[str] = None,
        file_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, str]] = None,
        expires_after: Optional[Dict[str, Any]] = None,
        chunking_strategy: Optional[Dict[str, Any]] = None
    ) -> VectorStore:
        """
        Create a vector store.
        
        Args:
            name: The name of the vector store
            file_ids: List of file IDs to add to the vector store
            metadata: Optional metadata key-value pairs
            expires_after: Expiration policy configuration
            chunking_strategy: Chunking configuration for files
        
        Returns:
            VectorStore object
        """
        payload = {}
        if name:
            payload["name"] = name
        if file_ids:
            payload["file_ids"] = file_ids
        if metadata:
            payload["metadata"] = metadata
        if expires_after:
            payload["expires_after"] = expires_after
        if chunking_strategy:
            payload["chunking_strategy"] = chunking_strategy
        
        data = self._client._rag_request("POST", "/vector_stores", json=payload)
        return VectorStore(**data)

    def list(
        self,
        limit: Optional[int] = None,
        order: Optional[str] = None,
        after: Optional[str] = None,
        before: Optional[str] = None
    ) -> VectorStoreListResponse:
        """
        List vector stores.
        
        Args:
            limit: Number of vector stores to return (1-100)
            order: Sort order ('asc' or 'desc')
            after: Cursor for pagination
            before: Cursor for pagination
        
        Returns:
            VectorStoreListResponse with list of vector stores
        """
        params = {}
        if limit:
            params["limit"] = limit
        if order:
            params["order"] = order
        if after:
            params["after"] = after
        if before:
            params["before"] = before
        
        data = self._client._rag_request("GET", "/vector_stores", params=params)
        return VectorStoreListResponse(**data)

    def retrieve(self, vector_store_id: str) -> VectorStore:
        """
        Get a vector store.
        
        Args:
            vector_store_id: The ID of the vector store
        
        Returns:
            VectorStore object
        """
        data = self._client._rag_request("GET", f"/vector_stores/{vector_store_id}")
        return VectorStore(**data)

    def update(
        self,
        vector_store_id: str,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        expires_after: Optional[Dict[str, Any]] = None
    ) -> VectorStore:
        """
        Update a vector store.
        
        Args:
            vector_store_id: The ID of the vector store
            name: Updated name
            metadata: Updated metadata key-value pairs
            expires_after: Updated expiration policy
        
        Returns:
            VectorStore object with updates
        """
        payload = {}
        if name:
            payload["name"] = name
        if metadata:
            payload["metadata"] = metadata
        if expires_after:
            payload["expires_after"] = expires_after
        
        data = self._client._rag_request(
            "POST",
            f"/vector_stores/{vector_store_id}",
            json=payload
        )
        return VectorStore(**data)

    def delete(self, vector_store_id: str) -> VectorStoreDeleted:
        """
        Delete a vector store.
        
        Args:
            vector_store_id: The ID of the vector store to delete
        
        Returns:
            VectorStoreDeleted confirmation
        """
        data = self._client._rag_request("DELETE", f"/vector_stores/{vector_store_id}")
        return VectorStoreDeleted(**data)

    def search(
        self,
        vector_store_id: str,
        query: Union[str, List[str]],
        filters: Optional[Dict[str, Any]] = None,
        max_num_results: Optional[int] = None,
        ranking_options: Optional[Dict[str, Any]] = None,
        rewrite_query: Optional[bool] = None
    ) -> VectorStoreSearchResponse:
        """
        Search a vector store.
        
        Args:
            vector_store_id: The ID of the vector store
            query: Search query string or list of strings
            filters: File attribute filters (comparison or compound)
            max_num_results: Maximum results to return (1-50)
            ranking_options: Re-ranking configuration
            rewrite_query: Whether to rewrite the query for vector search
        
        Returns:
            VectorStoreSearchResponse with search results
        """
        payload = {"query": query}
        if filters:
            payload["filters"] = filters
        if max_num_results:
            payload["max_num_results"] = max_num_results
        if ranking_options:
            payload["ranking_options"] = ranking_options
        if rewrite_query is not None:
            payload["rewrite_query"] = rewrite_query
        
        data = self._client._rag_request(
            "POST",
            f"/vector_stores/{vector_store_id}/search",
            json=payload
        )
        return VectorStoreSearchResponse(**data)
