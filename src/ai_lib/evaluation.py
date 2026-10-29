"""Small evaluation utilities; rubrics and experiments stay in notebooks."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from ai_lib.models import api_error_guidance, expected_call_error


def judge_score(value):
    """Accept binary judgments only; strings and missing scores are not verdicts."""
    if type(value) is bool:
        return value
    if type(value) in (int, float) and value in (0, 1):
        return bool(value)
    raise ValueError("Judge score must be a boolean or numeric 0 or 1.")


class SupportAnswer(BaseModel):
    """A response and its evidence IDs; valid structure does not establish truth."""
    model_config = ConfigDict(extra="forbid")
    answer: str = Field(min_length=1)
    source_ids: list[str]
    disposition: Literal["answer", "clarify"]


def check_answer(value, allowed_ids):
    try:
        answer = SupportAnswer.model_validate(value)
    except (ValueError, TypeError):
        return {"structure": False, "citations": False}
    citations = set(answer.source_ids).issubset(set(allowed_ids))
    if answer.disposition == "answer":
        citations = citations and bool(answer.source_ids)
    return {"structure": True, "citations": bool(citations)}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def save_json(path, value):
    """Replace a complete JSON record atomically."""
    from ai_lib.recording import redact_record
    value = redact_record(value)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


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
                from ai_lib.models import ExternalCallError
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


def build_judge(model, rubric, *, name):
    from openevals.llm import create_llm_as_judge
    return create_llm_as_judge(judge=model, prompt=rubric, feedback_key=name,
                              continuous=False, use_reasoning=True)


def paired_scores(left, right, metric):
    """Compare the same cases and references; preserve unavailable judgments."""
    keys = ["case_id", "reference_hash"]
    for table in (left, right):
        if table.empty or table.case_id.duplicated().any():
            raise ValueError("Provide nonempty results with unique case IDs.")
    if set(map(tuple, left[keys].values)) != set(map(tuple, right[keys].values)):
        raise ValueError("Comparison requires identical case IDs and references.")
    result = left[keys+[metric]].merge(right[keys+[metric]], on=keys, suffixes=("_before", "_after"))
    result["difference"] = result[f"{metric}_after"].astype(float) - result[f"{metric}_before"].astype(float)
    return result


def result_summary(rows):
    frame = pd.DataFrame(rows)
    fields = [f for f in ("correctness_ordinal", "correctness", "relevance", "groundedness", "retrieval_relevance",
                          "structure", "citations", "seconds") if f in frame]
    grouped = frame.groupby("method", sort=False)
    means = grouped[fields].mean()
    means.insert(0, "cases", grouped.size())
    for metric in fields:
        if metric != "seconds":
            means[metric+"_judged"] = grouped[metric].count()
    return means.reset_index()


def local_evaluation_client():
    """Use local examples and forbid accidental hosted requests from this client."""
    import requests
    from langsmith import Client

    class LocalSession(requests.Session):
        def request(self, *args, **kwargs):
            raise RuntimeError("This evaluation client is local-only.")

    return Client(auto_batch_tracing=False, info={"version": "local"}, session=LocalSession())
