"""Stable fingerprints and atomic redacted JSON records."""
import os
from ai_lib.models.configuration import KEY_NAMES
import hashlib
import json
from pathlib import Path


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def save_json(path, value):
    """Replace a complete JSON record atomically."""
    value = redact_record(value)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def redact_record(value):
    """Exclude credential fields and configured API-key values from saved records."""
    if isinstance(value, dict):
        return {str(k): redact_record(v) for k, v in value.items()
                if not any(word in str(k).lower() for word in ("api_key", "authorization", "secret", "password"))}
    if isinstance(value, (list, tuple)):
        return [redact_record(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str):
            for name in [*KEY_NAMES.values(), "TAVILY_API_KEY"]:
                key = os.getenv(name)
                if key:
                    value = value.replace(key, "[REDACTED]")
        return value
    return f"<{type(value).__name__}>"
