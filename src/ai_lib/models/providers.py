"""Construct native LangChain models with provider imports loaded on demand."""
from .configuration import KEY_NAMES, ModelSettings, require_key


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
