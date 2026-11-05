"""Answer structure, citation and assessment prerequisites."""
import re
from ai_lib.evaluation.schemas import SupportAnswer


def check_answer(value, allowed_ids):
    try:
        answer = SupportAnswer.model_validate(value)
    except (ValueError, TypeError):
        return {"structure": False, "citations": False}
    citations = set(answer.source_ids).issubset(set(allowed_ids))
    if answer.disposition == "answer":
        citations = citations and bool(answer.source_ids)
    return {"structure": True, "citations": bool(citations)}


def assessment_gate(case, value, evidence):
    """Missing assessment inputs are errors; a malformed candidate is a failure."""
    if evidence is None or any(not case.get(k) for k in ("question", "expected_answer")):
        return {"verdict": "ERROR", "reason": "Required assessment evidence is missing"}
    checks = check_answer(value, [d.metadata["passage_id"] for d in evidence])
    if not all(checks.values()):
        return {"verdict": "FAIL", "reason": "Invalid answer structure or source IDs"}
    return {"verdict": "PASS", "reason": "Ready for semantic assessment"}


def redact_email(text):
    """Request-wide teaching rule; not a general personal-data detector."""
    return re.sub(r"[A-Za-z0-9_.+%-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[REDACTED_EMAIL]", text)
