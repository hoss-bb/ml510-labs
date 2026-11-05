"""Local callback traces without credentials or provider clients."""
from datetime import datetime, timezone
from time import perf_counter
from langchain_core.callbacks import BaseCallbackHandler
from ai_lib.models.errors import api_error_guidance
from .serialization import redact_record


class LocalTrace(BaseCallbackHandler):
    raise_error = True
    def __init__(self, run, *, case_id=None, configuration_id=None):
        self.run = run
        self.case_id = case_id
        self.configuration_id = configuration_id
        self.events = []
        self.started = {}

    def _safe(self, value):
        return redact_record(value)

    def event(self, kind, **details):
        self.events.append({"kind": kind, "at_utc": datetime.now(timezone.utc).isoformat(),
                            "case_id": self.case_id, "configuration_id": self.configuration_id,
                            "execution": "live", **self._safe(details)})
        self.run.record("trace", {"execution": "mixed" if any(e["execution"] != "live" for e in self.events) else "live", "events": self.events})

    @staticmethod
    def message(message):
        return {"type": message.type, "content": message.content,
                "tool_calls": getattr(message, "tool_calls", []),
                "tool_call_id": getattr(message, "tool_call_id", None),
                "status": getattr(message, "status", None),
                "usage": getattr(message, "usage_metadata", None),
                "completion": {key: value for key, value in getattr(message, "response_metadata", {}).items()
                               if key in {"finish_reason", "stop_reason", "model_name", "model", "model_provider"}}}

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        self.started[str(run_id)] = perf_counter()
        self.event("model_input", run_id=str(run_id), messages=[[self.message(m) for m in batch] for batch in messages])

    def on_llm_end(self, response, *, run_id, **kwargs):
        elapsed = perf_counter() - self.started.pop(str(run_id), perf_counter())
        outputs = []
        for batch in response.generations:
            for generation in batch:
                outputs.append(self.message(generation.message))
        self.event("model_output", run_id=str(run_id), seconds=elapsed, responses=outputs)

    def on_llm_error(self, error, *, run_id, **kwargs):
        self.started.pop(str(run_id), None)
        self.event("model_error", run_id=str(run_id), guidance=api_error_guidance(error))

    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        self.started[str(run_id)] = perf_counter()
        self.event("tool_input", run_id=str(run_id), name=serialized.get("name"), arguments=input_str)

    def on_tool_end(self, output, *, run_id, **kwargs):
        if hasattr(output, "content"):
            output = self.message(output)
        elapsed = perf_counter() - self.started.pop(str(run_id), perf_counter())
        self.event("tool_output", run_id=str(run_id), seconds=elapsed, result=output)

    def on_tool_error(self, error, *, run_id, **kwargs):
        self.started.pop(str(run_id), None)
        self.event("tool_error", run_id=str(run_id), guidance=api_error_guidance(error))
