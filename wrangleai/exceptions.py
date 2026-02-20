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


class NotFoundError(WrangleError):
    """Raised when resource is not found (404)."""
    pass


class UnprocessableEntityError(WrangleError):
    """Raised when request is valid but unprocessable (422)."""
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


def _make_status_error(
    status_code: int,
    response_body: Optional[dict] = None,
    message: Optional[str] = None
) -> WrangleError:
    """Create appropriate error based on HTTP status code.
    
    Args:
        status_code: HTTP status code
        response_body: Response body dict (if available)
        message: Error message (if already parsed)
    
    Returns:
        Appropriate WrangleError subclass instance
    """
    if message is None:
        message = "Unknown error"
        if response_body:
            if isinstance(response_body.get("error"), dict):
                message = response_body["error"].get("message", "Unknown error")
            else:
                message = response_body.get("error", "Unknown error")
    
    if status_code == 400:
        return BadRequestError(message, status_code, response_body)
    elif status_code == 401:
        return AuthenticationError(message, status_code, response_body)
    elif status_code == 403:
        return PermissionDeniedError(message, status_code, response_body)
    elif status_code == 404:
        return NotFoundError(message, status_code, response_body)
    elif status_code == 422:
        return UnprocessableEntityError(message, status_code, response_body)
    elif status_code == 429:
        return RateLimitError(message, status_code, response_body)
    elif status_code >= 500:
        return APIError(message, status_code, response_body)
    else:
        return WrangleError(message, status_code, response_body)
