"""Read the editable example without importing a model or AI integration."""
import json
from pathlib import Path
from .prediction.schemas import Features


def load_example_listing(path=None):
    """Read an explicit resource or the example beneath the configured S10 root."""
    if path is None:
        from .runtime.settings import APP
        path = APP / "resources/example_listing.json"
    path = Path(path)
    example = json.loads(path.read_text(encoding="utf-8"))
    Features.model_validate(example)
    return example
