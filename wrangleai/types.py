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


# --- Models API Response (OpenAI-compatible) ---

if BaseModel is not None:
    class Model(BaseModel):
        """
        Represents a model object from the models list endpoint.
        Compatible with OpenAI's Model response.
        """
        model_config = ConfigDict(extra="allow")
        
        id: str
        object: str = "model"
        created: int
        owned_by: str
    
    
    class ModelsListResponse(BaseModel):
        """
        Response from the models list endpoint.
        Compatible with OpenAI's models.list() response.
        """
        model_config = ConfigDict(extra="allow")
        
        object: str = "list"
        data: List["Model"]


# --- Files API Response Models ---

if BaseModel is not None:
    class FileObject(BaseModel):
        """
        Represents a file that has been uploaded to WrangleAI.
        Compatible with OpenAI's File object.
        """
        model_config = ConfigDict(extra="allow")
        
        id: str
        object: str = "file"
        bytes: int
        created_at: int
        filename: str
        purpose: str
        status: Optional[str] = None
        status_details: Optional[str] = None
        expires_at: Optional[int] = None
    
    
    class FileDeleted(BaseModel):
        """Response when a file is deleted."""
        model_config = ConfigDict(extra="allow")
        
        id: str
        object: str = "file"
        deleted: bool
    
    
    class FileListResponse(BaseModel):
        """Response from files list endpoint."""
        model_config = ConfigDict(extra="allow")
        
        object: str = "list"
        data: List["FileObject"]
        has_more: bool
        first_id: Optional[str] = None
        last_id: Optional[str] = None


# --- Vector Stores API Response Models ---

if BaseModel is not None:
    class VectorStoreFileCounts(BaseModel):
        """File counts for a vector store."""
        model_config = ConfigDict(extra="allow")
        
        total: int
        in_progress: int
        completed: int
        failed: int
        cancelled: int
    
    
    class VectorStoreExpiresAfter(BaseModel):
        """Expiration policy for a vector store."""
        model_config = ConfigDict(extra="allow")
        
        anchor: str
        days: int
    
    
    class VectorStore(BaseModel):
        """
        Represents a vector store for RAG operations.
        Compatible with OpenAI's VectorStore object.
        """
        model_config = ConfigDict(extra="allow")
        
        id: str
        object: str = "vector_store"
        created_at: int
        name: str
        usage_bytes: int
        file_counts: VectorStoreFileCounts
        status: str
        expires_after: Optional[VectorStoreExpiresAfter] = None
        expires_at: Optional[int] = None
        last_active_at: Optional[int] = None
        metadata: Optional[Dict[str, Any]] = None
    
    
    class VectorStoreDeleted(BaseModel):
        """Response when a vector store is deleted."""
        model_config = ConfigDict(extra="allow")
        
        id: str
        object: str = "vector_store.deleted"
        deleted: bool
    
    
    class VectorStoreListResponse(BaseModel):
        """Response from vector stores list endpoint."""
        model_config = ConfigDict(extra="allow")
        
        object: str = "list"
        data: List["VectorStore"]
        has_more: bool
        first_id: Optional[str] = None
        last_id: Optional[str] = None


# --- Vector Store Files API Response Models ---

if BaseModel is not None:
    class VectorStoreFileError(BaseModel):
        """Error information for a vector store file."""
        model_config = ConfigDict(extra="allow")
        
        code: str
        message: str
    
    
    class StaticChunkingStrategy(BaseModel):
        """Static chunking strategy parameters."""
        model_config = ConfigDict(extra="allow")
        
        max_chunk_size_tokens: int
        chunk_overlap_tokens: int
    
    
    class VectorStoreFile(BaseModel):
        """
        Represents a file attached to a vector store.
        Compatible with OpenAI's VectorStoreFile object.
        """
        model_config = ConfigDict(extra="allow")
        
        id: str
        object: str = "vector_store.file"
        created_at: int
        vector_store_id: str
        status: str
        usage_bytes: int
        last_error: Optional[VectorStoreFileError] = None
        chunking_strategy: Optional[Dict[str, Any]] = None
        attributes: Optional[Dict[str, Union[str, int, bool]]] = None
    
    
    class VectorStoreFileDeleted(BaseModel):
        """Response when a vector store file is deleted."""
        model_config = ConfigDict(extra="allow")
        
        id: str
        object: str = "vector_store.file.deleted"
        deleted: bool
    
    
    class VectorStoreFileListResponse(BaseModel):
        """Response from vector store files list endpoint."""
        model_config = ConfigDict(extra="allow")
        
        object: str = "list"
        data: List["VectorStoreFile"]
        has_more: bool
        first_id: Optional[str] = None
        last_id: Optional[str] = None


# --- Vector Store Search API Response Models ---

if BaseModel is not None:
    class SearchResultContent(BaseModel):
        """Content chunk from search result."""
        model_config = ConfigDict(extra="allow")
        
        type: str
        text: str
    
    
    class SearchResultItem(BaseModel):
        """Individual search result item."""
        model_config = ConfigDict(extra="allow")
        
        file_id: str
        filename: str
        score: float
        content: List[SearchResultContent]
        attributes: Optional[Dict[str, Union[str, int, bool]]] = None
    
    
    class VectorStoreSearchResponse(BaseModel):
        """Response from vector store search endpoint."""
        model_config = ConfigDict(extra="allow")
        
        object: str = "vector_store.search_results.page"
        data: List[SearchResultItem]
        search_query: List[str]
        has_more: bool
        next_page: Optional[str] = None


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