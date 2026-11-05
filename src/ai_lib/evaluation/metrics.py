"""Inspectable metric arithmetic and strict score validation."""
from collections import Counter
import re


def judge_score(value):
    """Accept binary judgments only; strings and missing scores are not verdicts."""
    if type(value) is bool:
        return value
    if type(value) in (int, float) and value in (0, 1):
        return bool(value)
    raise ValueError("Judge score must be a boolean or numeric 0 or 1.")


def lexical_scores(answer, reference):
    """Lowercase word tokens, preserving multiplicity; no stemming or synonyms."""
    a, r = (re.findall(r"\w+", text.casefold()) for text in (answer, reference))
    overlap = sum((Counter(a) & Counter(r)).values())
    return {"exact_match": a == r,
            "token_f1": 2 * overlap / (len(a) + len(r)) if a or r else None}


def ordinal_score(value):
    """Four correctness anchors: 0 incorrect through 3 correct and complete."""
    if type(value) not in (int, float) or value not in (0, 1, 2, 3):
        raise ValueError("Correctness must be an integer anchor from 0 to 3.")
    return int(value)


def unit_score(unit_ids, verdicts, *, labels, positive, ranked=False):
    """Require one verdict per unit; keep unknown support separate from errors."""
    ids = list(unit_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("Evaluation units need unique IDs.")
    observed = [v["id"] for v in verdicts]
    if len(observed) != len(set(observed)) or set(observed) != set(ids):
        raise ValueError("Missing, duplicate or unexpected unit verdicts.")
    if positive not in labels or any(v["verdict"] not in labels for v in verdicts):
        raise ValueError("Unexpected verdict label.")
    by_id = {v["id"]: v["verdict"] for v in verdicts}
    flags = [by_id[key] == positive for key in ids]
    if not ids:
        return {"score": None, "status": "not_applicable", "units": 0,
                "positive": 0, "reason": "No applicable units"}
    count = sum(flags)
    if ranked:
        if not count:
            return {"score": None, "status": "not_applicable", "units": len(ids),
                    "positive": 0, "reason": "No useful returned nodes; retrieval failure"}
        cumulative = 0
        precisions = []
        for rank, flag in enumerate(flags, 1):
            cumulative += flag
            if flag:
                precisions.append(cumulative / rank)
        score = sum(precisions) / count
    else:
        score = count / len(ids)
    return {"score": score, "status": "ok", "units": len(ids),
            "positive": count, "reason": "Complete verdict coverage"}


def judge_disagreements(reference_pass, judged_pass):
    """Rates against explicit reviewed labels, not against another hidden judge."""
    if len(reference_pass) != len(judged_pass) or not reference_pass:
        raise ValueError("Supply nonempty aligned reviewed and judge verdicts.")
    if any(type(x) is not bool for x in [*reference_pass, *judged_pass]):
        raise ValueError("Verdicts must be booleans; errors are not verdicts.")
    failures = sum(not x for x in reference_pass)
    valid = sum(reference_pass)
    missed = sum(not r and j for r, j in zip(reference_pass, judged_pass))
    rejected = sum(r and not j for r, j in zip(reference_pass, judged_pass))
    return {"reference_failures": failures, "reference_valid": valid,
            "missed_failures": missed, "false_rejections": rejected,
            "missed_failure_rate": missed / failures if failures else None,
            "false_rejection_rate": rejected / valid if valid else None}
