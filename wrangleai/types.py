# File: wrangleai/types.py
from typing import List, Optional, Union, Dict, Any, Literal
from pydantic import BaseModel, Field

# --- Common ---
class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class FunctionCall(BaseModel):
    name: str
    arguments: str

class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: FunctionCall

# --- Chat Models ---
class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None

class Choice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None

class ChatCompletion(BaseModel):
    id: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    created: int
    model: str
    choices: List[Choice]
    usage: Optional[Usage] = None
    output: Optional[List[Dict[str, Any]]] = None # For Grounding/Web Search responses

# --- Streaming Models ---
class Delta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None

class ChunkChoice(BaseModel):
    index: int
    delta: Delta
    finish_reason: Optional[str] = None

class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"]
    created: int
    model: str
    choices: List[ChunkChoice]
    usage: Optional[Usage] = None

# --- Usage/Cost Models ---
class ModelUsageStats(BaseModel):
    model: str
    requests: int
    input_tokens: int = Field(alias="inputTokens")
    output_tokens: int = Field(alias="outputTokens")
    total_tokens: int = Field(alias="totalTokens")
    total_cost: float = Field(alias="total_cost")

class UsageReport(BaseModel):
    total_requests: int
    total_tokens: int
    total_cost: float
    optimized: bool
    usage_by_model: List[ModelUsageStats]

class CostReport(BaseModel):
    total_cost: float

class KeyInfo(BaseModel):
    valid: bool
    message: str
    apiKeyId: str
    keyStatus: str
    expiry: Optional[str] = None

# Type Alias for Routing
WrangleModel = Union[
    Literal["auto", "gpt-4o", "gpt-4o-mini", "gemini-2.5-pro", "gemini-2.5-flash"], 
    str
]

class SLMConfig(BaseModel):
    useSlm: bool
    useCase: Optional[str] = None
