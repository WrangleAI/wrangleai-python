# File: wrangleai/__init__.py
from .client import WrangleAI
from .async_client import AsyncWrangleAI
from .exceptions import (
    WrangleError, APIConnectionError, APIStatusError, 
    AuthenticationError, RateLimitError, BadRequestError, InternalServerError
)
from .types import (
    ChatCompletion, ChatCompletionChunk, Choice, Usage, 
    WrangleModel, SLMConfig
)

__version__ = "0.3.0"
__all__ = [
    "WrangleAI",
    "AsyncWrangleAI",
    "WrangleError",
    "APIConnectionError",
    "AuthenticationError",
    "RateLimitError",
    "BadRequestError",
    "InternalServerError",
    "ChatCompletion",
    "ChatCompletionChunk"
]