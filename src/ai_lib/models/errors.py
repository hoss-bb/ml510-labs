"""Recognize external failures without hiding programming errors."""



class ExternalCallError(RuntimeError):
    """An external service returned unusable evidence or an incomplete response."""


def expected_call_error(error):
    """Recognize service boundaries without swallowing ordinary programming bugs."""
    from httpx import HTTPError
    from requests import RequestException
    from langchain_core.exceptions import OutputParserException
    from google.genai.errors import APIError
    from pydantic import ValidationError
    return isinstance(error, (ExternalCallError, HTTPError, RequestException,
        APIError, OutputParserException,
        ValidationError, TimeoutError, ConnectionError)) or any(
            cls.__module__.startswith("cohere.errors")
            or (cls.__module__.startswith("langchain_google_genai") and cls.__name__ == "GoogleGenerativeAIError")
            or (cls.__module__.startswith("groq") and cls.__name__ == "APIError")
            or (cls.__module__ == "langgraph.errors" and cls.__name__ == "GraphRecursionError")
            or cls.__module__ == "langchain.agents.structured_output"
            for cls in type(error).__mro__)


def api_error_guidance(error: Exception) -> str:
    """Explain a failed call without reproducing provider payloads or credentials."""
    if isinstance(error, ExternalCallError):
        return "The service returned unusable data. Inspect the failed stage and service availability before retrying."
    if isinstance(error, (TimeoutError, ConnectionError)):
        return "The request timed out or could not connect. Check network access before retrying."
    text = str(error).lower()
    if any(word in text for word in ("429", "quota", "rate limit", "resource_exhausted")):
        return "Provider quota reached. Check your account limits and retry later; no provider was substituted."
    if any(word in text for word in ("401", "403", "api key", "unauthorized")):
        return "Provider access failed. Check the selected provider and its key in the root .env."
    return "API call failed. Check AI_MODEL availability, network access, and your provider account."
