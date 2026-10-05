"""Choose a provider; keep prompts and application logic in the notebook."""
from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

KEY_NAMES = {"gemini": "GOOGLE_API_KEY", "groq": "GROQ_API_KEY", "cohere": "COHERE_API_KEY"}


class ExternalCallError(RuntimeError):
    """An external service returned unusable evidence or an incomplete response."""


def expected_call_error(error):
    """Recognize service boundaries without swallowing ordinary programming bugs."""
    from httpx import HTTPError
    from requests import RequestException
    from langchain_core.exceptions import OutputParserException
    from google.genai.errors import APIError
    from groq import APIError as GroqError
    from langgraph.errors import GraphRecursionError
    from pydantic import ValidationError
    return isinstance(error, (ExternalCallError, HTTPError, RequestException,
        APIError, GroqError, OutputParserException, GraphRecursionError,
        ValidationError, TimeoutError, ConnectionError)) or any(
            cls.__module__.startswith("cohere.errors")
            or cls.__module__ == "langchain.agents.structured_output"
            for cls in type(error).__mro__)


@dataclass(frozen=True)
class ModelSettings:
    provider: str
    model: str
    temperature: float = 0.0
    max_tokens: int | None = 1024
    thinking_budget: int | None = None

    def __post_init__(self):
        if self.provider not in KEY_NAMES:
            raise ValueError("AI_PROVIDER must be gemini, groq, or cohere.")
        if not self.model.strip():
            raise ValueError("Set AI_MODEL to a model available to your provider account.")


def load_settings(root: Path) -> ModelSettings:
    """Read the root .env; existing process variables take precedence."""
    load_dotenv(Path(root) / ".env", override=False)
    # The teaching application writes local traces only.
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    provider = os.getenv("AI_PROVIDER", "gemini")
    model = os.getenv("AI_MODEL", "")
    budget = 0 if provider == "gemini" and model.startswith("gemini-2.5-flash") else None
    return ModelSettings(provider, model, max_tokens=None if provider == "cohere" else 1024,
                         thinking_budget=budget)


def require_key(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Set {name} in the repository root .env, then rerun setup. Do not paste keys into a notebook.")
    return value


def build_chat_model(settings: ModelSettings):
    """Return a native LangChain chat model, not a new model abstraction."""
    key = require_key(KEY_NAMES[settings.provider])
    common = {"model": settings.model, "temperature": settings.temperature,
              "max_tokens": settings.max_tokens, "max_retries": 2}
    if settings.provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(api_key=key, timeout=60, thinking_budget=settings.thinking_budget, **common)
    if settings.provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(api_key=key, timeout=60, **common)
    from langchain_cohere import ChatCohere
    # This integration has no constructor-level token budget. Record its native
    # default rather than claiming that the Gemini/Groq budget applies.
    return ChatCohere(cohere_api_key=key, timeout_seconds=60,
                      model=settings.model, temperature=settings.temperature)


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
