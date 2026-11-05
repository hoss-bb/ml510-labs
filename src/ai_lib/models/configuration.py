"""Provider settings and local environment loading."""
from dataclasses import dataclass
import os
from pathlib import Path
from dotenv import load_dotenv


KEY_NAMES = {"gemini": "GOOGLE_API_KEY", "groq": "GROQ_API_KEY", "cohere": "COHERE_API_KEY"}


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
