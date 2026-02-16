import os
import json
import httpx
from typing import Optional, List, Union, Generator, Any, Dict, overload
from .types import (
    WrangleObject, WrangleModel, SLMConfig,
    ChatCompletion, ChatCompletionChunk
)
from .exceptions import (
    WrangleError, AuthenticationError, RateLimitError,
    BadRequestError, APIError, APIConnectionError
)

class WrangleAI:
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        # base_url: str = "https://gateway.wrangleai.com/v1",
        base_url: str = "https://staging-gateway.wrangleai.com/v1",
        # base_url: str = "https://bd1851h1-8080.uks1.devtunnels.ms/v1",
        timeout: float = 60.0
    ):
        """
        Initialize the Wrangle AI Client.
        
        Args:
            api_key: Your Wrangle AI API Key. Defaults to env var WRANGLE_API_KEY.
            base_url: The API endpoint.
            timeout: Request timeout in seconds.
        """
        self.api_key = api_key or os.environ.get("WRANGLE_API_KEY")
        if not self.api_key:
            raise ValueError("The WrangleAI client requires an api_key argument or WRANGLE_API_KEY environment variable.")

        self.base_url = base_url.rstrip("/")
        self._last_request_id: Optional[str] = None

        self._client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            timeout=timeout
        )

        # Initialize Namespaces
        self.chat = Chat(self)
        self.usage = Usage(self)
        self.cost = Cost(self)
        self.keys = Keys(self)

    def close(self):
        """Close the underlying HTTP connections."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _request(self, method: str, path: str, **kwargs) -> Any:
        try:
            response = self._client.request(method, path, **kwargs)
            response.raise_for_status()
            # Store request ID for later retrieval if needed
            self._last_request_id = response.headers.get("x-request-id")
            return response.json()
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
            
            # Map status codes to specific exceptions
            if status_code in (401, 403):
                raise AuthenticationError(msg, status_code, err_body)
            elif status_code == 429:
                raise RateLimitError(msg, status_code, err_body)
            elif status_code == 400:
                raise BadRequestError(msg, status_code, err_body)
            elif status_code >= 500:
                raise APIError(msg, status_code, err_body)
            else:
                raise WrangleError(msg, status_code, err_body)
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
        messages: List[Dict[str, Any]],
        model: WrangleModel,
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
        messages: List[Dict[str, Any]],
        model: WrangleModel,
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
        messages: List[Dict[str, Any]],
        model: WrangleModel,
        stream: bool = False,
        temperature: Optional[float] = None,
        slm: Optional[SLMConfig] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        legacy_response: bool = False,
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

        if stream:
            return self._stream_request(payload, legacy_response)
        else:
            data = self._client._request("POST", "/chat/completions", json=payload)
            if legacy_response:
                return WrangleObject(data)
            completion = ChatCompletion.model_validate(data)
            # Set request ID from last request
            completion._request_id = self._client._last_request_id
            return completion

    def _stream_request(self, payload, legacy_response: bool = False):
        """Stream SSE responses with Python 3.13 compatibility."""
        with self._client._client.stream("POST", "/chat/completions", json=payload) as response:
            # Capture request ID from headers
            request_id = response.headers.get("x-request-id")
            
            if response.status_code != 200:
                content = response.read().decode('utf-8')
                try:
                    err_json = json.loads(content)
                    msg = err_json.get("error", {}).get("message") or content
                    err_body = err_json
                except:
                    msg = content
                    err_body = None
                
                # Map status codes to exceptions
                status_code = response.status_code
                if status_code in (401, 403):
                    raise AuthenticationError(msg, status_code, err_body)
                elif status_code == 429:
                    raise RateLimitError(msg, status_code, err_body)
                elif status_code == 400:
                    raise BadRequestError(msg, status_code, err_body)
                elif status_code >= 500:
                    raise APIError(msg, status_code, err_body)
                else:
                    raise WrangleError(msg, status_code, err_body)

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
                                chunk_obj._request_id = request_id
                                yield chunk_obj
                        except json.JSONDecodeError:
                            pass
            except GeneratorExit:
                # Handle cleanup when generator is closed early
                return

# --- Usage Namespace ---
class Usage:
    def __init__(self, client: WrangleAI):
        self._client = client

    def retrieve(self, start_date: str = None, end_date: str = None) -> WrangleObject:
        params = {}
        if start_date:
            params["startDate"] = start_date
        if end_date:
            params["endDate"] = end_date

        data = self._client._request("GET", "/usage", params=params)
        return WrangleObject(data)

    def retrieve_by_model(self, model: str, start_date: str = None, end_date: str = None) -> WrangleObject:
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

    def retrieve(self, start_date: str = None, end_date: str = None) -> WrangleObject:
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