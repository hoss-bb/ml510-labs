"""Configuration-aware reuse of successful calls only."""
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from pydantic import BaseModel
from ai_lib.models.errors import api_error_guidance, expected_call_error
from .serialization import fingerprint, save_json


class ResultCache:
    """Reuse successful calls only when their complete declared inputs match."""
    def __init__(self, directory):
        self.directory = Path(directory)

    def call(self, inputs, configuration, function, *, reuse=True):
        key = fingerprint({"inputs": inputs, "configuration": configuration})
        path = self.directory / f"{key}.json"
        if reuse and path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["status"] == "ok" and record.get("value") is not None and not (
                isinstance(record.get("value"), dict) and record["value"].get("failure_reason")
            ):
                return {**record, "execution": "reused", "original_seconds": record["seconds"], "seconds": 0.0}
        started = perf_counter()
        try:
            from langsmith import tracing_context
            # The evaluation runner creates its own tracing context. Override it
            # here as well as at setup; our explicit LocalTrace still records calls.
            with tracing_context(enabled=False):
                value = function()
            if value is None:
                from ai_lib.models.errors import ExternalCallError
                raise ExternalCallError("The call returned no structured response.")
            if isinstance(value, BaseModel):
                value = value.model_dump()
            record = {"status": "ok", "value": value}
            if isinstance(value, dict) and value.get("failure_reason"):
                record.update(status="error", error_type="IncompleteResponse",
                              failure_reason=value["failure_reason"])
        except Exception as error:
            if not expected_call_error(error):
                raise
            record = {"status": "error", "error_type": type(error).__name__,
                      "guidance": ("Graph step limit reached. Inspect the path and configured bounds."
                                   if type(error).__name__ == "GraphRecursionError" else api_error_guidance(error))}
        record.update(stage=configuration.get("stage", "call"), key=key, execution="live", seconds=perf_counter()-started,
                      at_utc=datetime.now(timezone.utc).isoformat())
        # Provider errors are returned for the failure table, never reused as answers.
        save_json(path, record)
        return record
