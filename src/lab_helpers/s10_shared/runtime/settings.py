"""Load local settings; process variables take precedence."""
from dataclasses import dataclass
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(os.environ.get("S10_ROOT", str(Path.cwd()))).expanduser().resolve()
APP = ROOT / "apps" / "listing_review"


@dataclass(frozen=True)
class Settings:
    data: Path
    model: Path
    database: Path
    upstream_url: str
    api_url: str
    poll_seconds: float


def load_environment() -> None:
    """Use process settings, then the application file, then the course fallback."""
    load_dotenv(APP / ".env", override=False)
    # Reuse an existing course key locally; Compose supplies keys to API and prompt job.
    load_dotenv(ROOT / ".env", override=False)


def settings() -> Settings:
    load_environment()
    data = Path(os.getenv("S10_DATA", str(ROOT / "local/listing_review")))
    poll = float(os.getenv("S10_POLL_SECONDS", "15"))
    if not 1 <= poll <= 86400:
        raise ValueError("S10_POLL_SECONDS must be between 1 and 86400.")
    return Settings(data, Path(os.getenv("S10_MODEL", str(data / "model"))),
                    Path(os.getenv("S10_DATABASE", str(data / "state" / "app.sqlite"))),
                    os.getenv("S10_UPSTREAM_URL", "http://127.0.0.1:8001"),
                    os.getenv("S10_API_URL", "http://127.0.0.1:8000"), poll)
