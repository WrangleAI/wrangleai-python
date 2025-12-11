import os
import json
import httpx
from typing import Optional, List, Union, Generator, Any, Dict
from .types import WrangleObject, WrangleModel

class WrangleAI:
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        base_url: str = "https://gateway.wrangleai.com/v1",
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
            return response.json()
        except httpx.HTTPStatusError as e:
            try:
                err_body = e.response.json()
                if isinstance(err_body.get("error"), dict):
                    msg = err_body["error"].get("message")
                else:
                    msg = err_body.get("error") or str(e)
            except Exception:
                msg = str(e)
            
            raise Exception(f"WrangleAI Error [{e.response.status_code}]: {msg}")

# --- Chat Namespace ---
class Chat:
    def __init__(self, client: WrangleAI):
        self.completions = Completions(client)

class Completions:
    def __init__(self, client: WrangleAI):
        self._client = client

    def create(
        self,
        messages: List[Dict[str, Any]],
        model: WrangleModel,
        stream: bool = False,
        temperature: Optional[float] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs
    ) -> Union[WrangleObject, Generator[WrangleObject, None, None]]:
        """
        Create a chat completion.
        
        Args:
            messages: A list of messages comprising the conversation.
            model: ID of the model to use (e.g. "auto", "gpt-4o").
            stream: If True, returns an iterator of chunks.
        """
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **kwargs
        }

        if temperature is not None:
            payload["temperature"] = temperature
        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice

        if stream:
            return self._stream_request(payload)
        else:
            data = self._client._request("POST", "/chat/completions", json=payload)
            return WrangleObject(data)

    def _stream_request(self, payload):
        with self._client._client.stream("POST", "/chat/completions", json=payload) as response:
            if response.status_code != 200:
                content = response.read().decode('utf-8')
                try:
                    err_json = json.loads(content)
                    msg = err_json.get("error", {}).get("message") or content
                except:
                    msg = content
                raise Exception(f"WrangleAI Error [{response.status_code}]: {msg}")

            for line in response.iter_lines():
                if not line: continue
                if line.startswith("data: "):
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        return
                    try:
                        chunk = json.loads(data)
                        yield WrangleObject(chunk)
                    except json.JSONDecodeError:
                        pass

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
        try:
            response = self._client._client.get(
                "/keys/verify", 
                headers={"X-API-Key": self._client.api_key}
            )
            response.raise_for_status()
            return WrangleObject(response.json())
        except httpx.HTTPStatusError as e:
             raise Exception(f"WrangleAI Error: {e}")