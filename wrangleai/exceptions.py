"""Custom exceptions for the WrangleAI SDK."""

from typing import Optional


class WrangleError(Exception):
    """Base exception for all WrangleAI SDK errors."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, response_body: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.response_body = response_body

"""
Error Details in OpenAI SDK:
Status Code	    Error Type

    400	        BadRequestError
    401	        AuthenticationError
    403	        PermissionDeniedError
    404	        NotFoundError
    422	        UnprocessableEntityError
    429	        RateLimitError
    >=500	      InternalServerError
    N/A	        APIConnectionError

"""

class AuthenticationError(WrangleError):
    """Raised when API key is invalid or missing (401)."""
    pass


class PermissionDeniedError(WrangleError):
    """Raised when the user does not have permission to access the resource (403)."""
    pass


class RateLimitError(WrangleError):
    """Raised when rate limit is exceeded (429)."""
    pass


class UnprocessableEntity(WrangleError):
    """Raised when Unprocessable Entity (422)."""
    pass


class BadRequestError(WrangleError):
    """Raised when request is malformed or invalid (400)."""
    pass


class APIError(WrangleError):
    """Raised when the API returns a server error (500+)."""
    pass


class APIConnectionError(WrangleError):
    """Raised when network connection fails or times out."""
    pass
