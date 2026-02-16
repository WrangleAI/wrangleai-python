"""Async client for WrangleAI SDK using httpx.AsyncClient."""

import os
import json
import httpx
from typing import Optional, List, Union, AsyncGenerator, Any, Dict, overload
from .types import (
    WrangleObject, WrangleModel, SLMConfig,
    ChatCompletion, ChatCompletionChunk
)
from .exceptions import (
    WrangleError, AuthenticationError, RateLimitError,
    BadRequestError, APIError, APIConnectionError
)


class AsyncWrangleAI:
    """Async WrangleAI client for asyncio applications."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        # base_url: str = "https://gateway.wrangleai.com/v1",
        base_url: str = "https://bd1851h1-8080.uks1.devtunnels.ms/v1",
        timeout: float = 60.0
    ):
        """
        Initialize the Async Wrangle AI Client.
        
        Args:
            api_key: Your Wrangle AI API Key. Defaults to env var WRANGLE_API_KEY.
            base_url: The API endpoint.
            timeout: Request timeout in seconds.
        """
        self.api_key = api_key or os.environ.get("WRANGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "The AsyncWrangleAI client requires an api_key argument or WRANGLE_API_KEY environment variable."
            )

        self.base_url = base_url.rstrip("/")

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            timeout=timeout
        )

        # Initialize Namespaces
        self.chat = AsyncChat(self)
        self.usage = AsyncUsage(self)
        self.cost = AsyncCost(self)
        self.keys = AsyncKeys(self)

    async def aclose(self):
        """Close the underlying HTTP connections."""
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.aclose()

    async def _request(self, method: str, path: str, **kwargs) -> Any:
        """Make an async HTTP request with error handling."""
        try:
            response = await self._client.request(method, path, **kwargs)
            response.raise_for_status()
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


# --- Async Chat Namespace ---
class AsyncChat:
    def __init__(self, client: AsyncWrangleAI):
        self.completions = AsyncCompletions(client)


class AsyncCompletions:
    def __init__(self, client: AsyncWrangleAI):
        self._client = client

    @overload
    async def create(
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
    async def create(
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
    ) -> Union[AsyncGenerator[ChatCompletionChunk, None], AsyncGenerator[WrangleObject, None]]:
        ...
    
    async def create(
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
    ) -> Union[ChatCompletion, WrangleObject, AsyncGenerator[ChatCompletionChunk, None], AsyncGenerator[WrangleObject, None]]:
        """
        Create a chat completion asynchronously.
        
        Args:
            messages: A list of messages comprising the conversation.
            model: ID of the model to use (e.g. "auto", "gpt-4o").
            stream: If True, returns an async iterator of chunks.
            temperature: Sampling temperature.
            slm: Efficiency-first routing configuration.
            tools: Tool definitions for function calling.
            tool_choice: Control which tool is called.
            legacy_response: If True, returns WrangleObject instead of Pydantic models.
            **kwargs: Additional parameters.
        
        Returns:
            ChatCompletion or AsyncGenerator[ChatCompletionChunk] (or legacy WrangleObject equivalents).
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
            data = await self._client._request("POST", "/chat/completions", json=payload)
            if legacy_response:
                return WrangleObject(data)
            return ChatCompletion.model_validate(data)

    async def _stream_request(
        self, 
        payload: Dict[str, Any], 
        legacy_response: bool = False
    ) -> AsyncGenerator[Union[ChatCompletionChunk, WrangleObject], None]:
        """Stream SSE responses asynchronously."""
        async with self._client._client.stream("POST", "/chat/completions", json=payload) as response:
            if response.status_code != 200:
                content = (await response.aread()).decode('utf-8')
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

            async for line in response.aiter_lines():
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
                            yield ChatCompletionChunk.model_validate(chunk)
                    except json.JSONDecodeError:
                        pass


# --- Async Usage Namespace ---
class AsyncUsage:
    def __init__(self, client: AsyncWrangleAI):
        self._client = client

    async def retrieve(
        self, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None
    ) -> WrangleObject:
        """Retrieve usage statistics."""
        params = {}
        if start_date:
            params["startDate"] = start_date
        if end_date:
            params["endDate"] = end_date

        data = await self._client._request("GET", "/usage", params=params)
        return WrangleObject(data)

    async def retrieve_by_model(
        self, 
        model: str, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None
    ) -> WrangleObject:
        """Retrieve usage statistics by model."""
        params = {"model": model}
        if start_date:
            params["startDate"] = start_date
        if end_date:
            params["endDate"] = end_date

        data = await self._client._request("GET", "/usage/model", params=params)
        return WrangleObject(data)


# --- Async Cost Namespace ---
class AsyncCost:
    def __init__(self, client: AsyncWrangleAI):
        self._client = client

    async def retrieve(
        self, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None
    ) -> WrangleObject:
        """Retrieve cost statistics."""
        params = {}
        if start_date:
            params["startDate"] = start_date
        if end_date:
            params["endDate"] = end_date

        data = await self._client._request("GET", "/cost", params=params)
        return WrangleObject(data)


# --- Async Keys Namespace ---
class AsyncKeys:
    def __init__(self, client: AsyncWrangleAI):
        self._client = client

    async def verify(self) -> WrangleObject:
        """Verify API key validity."""
        # The keys/verify endpoint requires X-API-Key header instead of Bearer token
        data = await self._client._request(
            "GET", 
            "/keys/verify",
            headers={"X-API-Key": self._client.api_key}
        )
        return WrangleObject(data)
