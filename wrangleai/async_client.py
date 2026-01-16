import os
import json
import httpx
from typing import Optional, List, Union, AsyncGenerator, Any, Dict
from .types import (
    ChatCompletion, ChatCompletionChunk, WrangleModel, SLMConfig, 
    UsageReport, CostReport, KeyInfo
)
from .exceptions import (
    APIConnectionError, AuthenticationError, RateLimitError, 
    BadRequestError, InternalServerError, APIStatusError
)

class AsyncWrangleAI:
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        base_url: str = "https://gateway.wrangleai.com/v1",
        timeout: float = 60.0,
        http_client: Optional[httpx.AsyncClient] = None
    ):
        """Initialize the Async Wrangle AI Client."""
        self.api_key = api_key or os.environ.get("WRANGLE_API_KEY")
        if not self.api_key:
            raise ValueError("The AsyncWrangleAI client requires an api_key.")

        self.base_url = base_url.rstrip("/")

        if http_client:
            self._client = http_client
        else:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                timeout=timeout
            )

        self.chat = AsyncChat(self)
        self.usage = AsyncUsage(self)
        self.cost = AsyncCost(self)
        self.keys = AsyncKeys(self)

    async def close(self):
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    def _handle_error(self, response: httpx.Response):
        """Reuse error handling logic."""
        # This mirrors the sync logic, re-implemented here or shared via util
        try:
            err_body = response.json()
            if isinstance(err_body.get("error"), dict):
                msg = err_body["error"].get("message")
            else:
                msg = err_body.get("error")
        except:
            msg = response.text

        error_msg = f"Error {response.status_code}: {msg}"
        
        if response.status_code == 400:
            raise BadRequestError(error_msg, response.status_code, err_body)
        elif response.status_code == 401 or response.status_code == 403:
            raise AuthenticationError(error_msg, response.status_code, err_body)
        elif response.status_code == 429:
            raise RateLimitError(error_msg, response.status_code, err_body)
        elif response.status_code >= 500:
            raise InternalServerError(error_msg, response.status_code, err_body)
        else:
            raise APIStatusError(error_msg, response.status_code, err_body)

    async def _request(self, method: str, path: str, **kwargs) -> Any:
        try:
            response = await self._client.request(method, path, **kwargs)
            if response.is_error:
                self._handle_error(response)
            return response.json()
        except httpx.ConnectError as e:
            raise APIConnectionError(f"Connection failed: {e}") from e
        except httpx.TimeoutException as e:
            raise APIConnectionError(f"Request timed out: {e}") from e

class AsyncChat:
    def __init__(self, client: AsyncWrangleAI):
        self.completions = AsyncCompletions(client)

class AsyncCompletions:
    def __init__(self, client: AsyncWrangleAI):
        self._client = client

    async def create(
        self,
        messages: List[Dict[str, Any]],
        model: WrangleModel,
        stream: bool = False,
        temperature: Optional[float] = None,
        slm: Optional[SLMConfig] = None, 
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs
    ) -> Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **kwargs
        }
        if slm:
            payload["slm"] = slm.model_dump() if hasattr(slm, "model_dump") else slm
        if temperature: payload["temperature"] = temperature
        if tools: payload["tools"] = tools
        if tool_choice: payload["tool_choice"] = tool_choice

        if stream:
            return self._stream_request(payload)
        else:
            data = await self._client._request("POST", "/chat/completions", json=payload)
            return ChatCompletion.model_validate(data)

    async def _stream_request(self, payload):
        try:
            async with self._client._client.stream("POST", "/chat/completions", json=payload) as response:
                if response.status_code != 200:
                    self._client._handle_error(response)

                async for line in response.aiter_lines():
                    if not line: continue
                    if line.startswith("data: "):
                        data = line[6:]
                        if data.strip() == "[DONE]":
                            return
                        try:
                            chunk = json.loads(data)
                            yield ChatCompletionChunk.model_validate(chunk)
                        except json.JSONDecodeError:
                            continue
        except httpx.ConnectError as e:
            raise APIConnectionError(f"Stream connection failed: {e}") from e

class AsyncUsage:
    def __init__(self, client: AsyncWrangleAI):
        self._client = client

    async def retrieve(self, start_date: str = None, end_date: str = None) -> UsageReport:
        params = {}
        if start_date: params["startDate"] = start_date
        if end_date: params["endDate"] = end_date
        data = await self._client._request("GET", "/usage", params=params)
        return UsageReport.model_validate(data)

class AsyncCost:
    def __init__(self, client: AsyncWrangleAI):
        self._client = client

    async def retrieve(self, start_date: str = None, end_date: str = None) -> CostReport:
        params = {}
        if start_date: params["startDate"] = start_date
        if end_date: params["endDate"] = end_date
        data = await self._client._request("GET", "/cost", params=params)
        return CostReport.model_validate(data)

class AsyncKeys:
    def __init__(self, client: AsyncWrangleAI):
        self._client = client

    async def verify(self) -> KeyInfo:
        data = await self._client._request("GET", "/keys/verify", headers={"X-API-Key": self._client.api_key})
        return KeyInfo.model_validate(data)
