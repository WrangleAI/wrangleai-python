from typing import Any, Dict, List, Optional, Union

try:
    from typing import Literal, TypedDict
except ImportError:
    from typing_extensions import Literal, TypedDict

try:
    from pydantic import BaseModel, Field, ConfigDict
except ImportError:
    BaseModel = None  # type: ignore
    Field = None  # type: ignore
    ConfigDict = None  # type: ignore

WrangleModel = Union[
    Literal["auto", "gpt-4", "gpt-4o", "gpt-4o-mini", "gemini-1.5-pro"], 
    str
]


# --- TypedDict for Request Parameters (OpenAI-compatible) ---

class MessageParam(TypedDict, total=False):
    """
    Message parameter for chat completions.
    
    Attributes:
        role: The role of the message author (user, assistant, system, etc.)
        content: The message content
        name: Optional name for the message author
        tool_calls: Optional tool calls in the message
    """
    role: str
    content: str
    name: Optional[str]
    tool_calls: Optional[List[Dict[str, Any]]]


class SLMConfig(TypedDict, total=False):
    """
    Configuration for Efficiency-First Routing.
    
    Attributes:
        useSlm (bool): Enable routing to Small Language Models.
        useCase (str, optional): The specific domain (e.g., 'coding', 'chat').
    """
    useSlm: bool
    useCase: Optional[str]


# --- Pydantic Models for Type-Safe Responses ---

if BaseModel is not None:
    class FunctionCall(BaseModel):
        """Function call details within a tool call."""
        model_config = ConfigDict(extra="allow")
        
        name: str
        arguments: str  # JSON string
    
    
    class ToolCall(BaseModel):
        """Tool call in a completion response."""
        model_config = ConfigDict(extra="allow")
        
        id: str
        type: Literal["function"]
        function: FunctionCall
    
    
    class Delta(BaseModel):
        """Incremental content delta in streaming responses."""
        model_config = ConfigDict(extra="allow")
        
        content: Optional[str] = None
        role: Optional[str] = None
        tool_calls: Optional[List[ToolCall]] = None
    
    
    class Message(BaseModel):
        """Complete message in non-streaming responses."""
        model_config = ConfigDict(extra="allow")
        
        role: str
        content: Optional[str] = None
        tool_calls: Optional[List[ToolCall]] = None
    
    
    class Choice(BaseModel):
        """A choice in a completion response."""
        model_config = ConfigDict(extra="allow")
        
        index: int
        delta: Optional[Delta] = None  # For streaming
        message: Optional[Message] = None  # For non-streaming
        finish_reason: Optional[str] = None
    
    
    class Usage(BaseModel):
        """Token usage information."""
        model_config = ConfigDict(extra="allow")
        
        prompt_tokens: int
        completion_tokens: int
        total_tokens: int
    
    
    class ChatCompletionChunk(BaseModel):
        """Streaming chat completion chunk."""
        model_config = ConfigDict(extra="allow")
        
        id: str
        object: str = "chat.completion.chunk"
        created: int
        model: str
        choices: List[Choice]
        _request_id: Optional[str] = None
        
        def to_json(self, **kwargs) -> str:
            """OpenAI-compatible alias for model_dump_json()."""
            return self.model_dump_json(**kwargs)
        
        def to_dict(self, **kwargs) -> Dict[str, Any]:
            """OpenAI-compatible alias for model_dump()."""
            return self.model_dump(**kwargs)
    
    
    class ChatCompletion(BaseModel):
        """Non-streaming chat completion response."""
        model_config = ConfigDict(extra="allow")
        
        id: str
        object: str = "chat.completion"
        created: int
        model: str
        choices: List[Choice]
        usage: Usage
        _request_id: Optional[str] = None
        
        def to_json(self, **kwargs) -> str:
            """OpenAI-compatible alias for model_dump_json()."""
            return self.model_dump_json(**kwargs)
        
        def to_dict(self, **kwargs) -> Dict[str, Any]:
            """OpenAI-compatible alias for model_dump()."""
            return self.model_dump(**kwargs)
else:
    # Fallback stubs if pydantic is not installed
    class FunctionCall:  # type: ignore
        pass
    
    class ToolCall:  # type: ignore
        pass
    
    class Delta:  # type: ignore
        pass
    
    class Message:  # type: ignore
        pass
    
    class Choice:  # type: ignore
        pass
    
    class Usage:  # type: ignore
        pass
    
    class ChatCompletionChunk:  # type: ignore
        pass
    
    class ChatCompletion:  # type: ignore
        pass

class WrangleObject:
    def __init__(self, data: Any):
        self._data = data
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    setattr(self, key, WrangleObject(value))
                else:
                    setattr(self, key, value)

    def __len__(self):
        if isinstance(self._data, (list, dict, str)):
            return len(self._data)
        return 0

    def __getattr__(self, name):
        return None
    
    def __getitem__(self, index):
        if isinstance(self._data, list):
            val = self._data[index]
            return WrangleObject(val) if isinstance(val, (dict, list)) else val
        raise TypeError("WrangleObject is not subscriptable")

    def __iter__(self):
        if isinstance(self._data, list):
            for item in self._data:
                yield WrangleObject(item) if isinstance(item, (dict, list)) else item
        else:
            raise TypeError("WrangleObject is not iterable")

    def __repr__(self):
        return f"{self._data}"
    
    def to_dict(self) -> Any:
        return self._data