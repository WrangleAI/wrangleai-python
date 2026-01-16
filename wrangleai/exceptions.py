class WrangleError(Exception):
    """Base exception for all Wrangle AI errors."""
    def __init__(self, message: str, request=None):
        super().__init__(message)
        self.request = request

class APIConnectionError(WrangleError):
    """Network communication failed (timeouts, DNS issues)."""
    pass

class APIStatusError(WrangleError):
    """The server returned a non-200 status code."""
    def __init__(self, message: str, status_code: int, response_body: dict = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body

class AuthenticationError(APIStatusError):
    """401/403 Errors: Invalid API Key or Permissions."""
    pass

class RateLimitError(APIStatusError):
    """429 Errors: Rate limit exceeded."""
    pass

class BadRequestError(APIStatusError):
    """400 Errors: Invalid parameters."""
    pass

class InternalServerError(APIStatusError):
    """5xx Errors: Server side issues."""
    pass