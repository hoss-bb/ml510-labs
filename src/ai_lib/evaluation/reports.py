"""Matched comparisons and summaries with explicit denominators."""
import pandas as pd


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


def paired_outcomes(table, *, expected_cases, baseline, candidate,
                    case_column="case_id", method_column="method",
                    outcome_column="passed", error_column="error"):
    """Summarize two methods on an explicit roster without selecting a winner.

    Each method must supply exactly one row per case. A null outcome or a
    non-null error marks execution incomplete, never a failed Boolean judgment.
    Improvements/regressions count only cases completed by both methods.
    """
    import numpy as np
    cases = list(expected_cases)
    if not cases or pd.Index(cases).hasnans or len(set(cases)) != len(cases):
        raise ValueError("Expected cases must be nonempty, unique and present.")
    if baseline == candidate:
        raise ValueError("Compare two distinct methods.")
    columns = [case_column, method_column, outcome_column, error_column]
    if table.columns.has_duplicates or any(name not in table for name in columns):
        raise ValueError("Provide case, method, Boolean outcome and execution-error columns.")
    if table[[case_column, method_column]].isna().any().any():
        raise ValueError("Case and method identities must be present.")
    expected = {(method, case) for method in [baseline, candidate] for case in cases}
    keys = list(zip(table[method_column], table[case_column]))
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Require exactly one row per expected case and method.")
    outcomes, completed = {}, {}
    for method in [baseline, candidate]:
        rows = table.loc[table[method_column] == method].set_index(case_column).reindex(cases)
        verdicts, flags = [], []
        for value, error in zip(rows[outcome_column], rows[error_column]):
            if not pd.isna(value) and not isinstance(value, (bool, np.bool_)):
                raise ValueError("Outcomes must be Boolean judgments or missing, not numeric scores.")
            valid = pd.isna(error) and not pd.isna(value)
            flags.append(bool(valid))
            verdicts.append(bool(value) if valid else None)
        outcomes[method], completed[method] = verdicts, flags
    paired = [a and b for a, b in zip(completed[baseline], completed[candidate])]
    improvements = sum(ok and not a and b for ok, a, b in zip(paired, outcomes[baseline], outcomes[candidate]))
    regressions = sum(ok and a and not b for ok, a, b in zip(paired, outcomes[baseline], outcomes[candidate]))
    return {"complete": all(paired), "case_count": len(cases), "paired_completed": sum(paired),
            "completed_cases": {method: sum(flags) for method, flags in completed.items()},
            "incomplete_cases": {method: [case for case, ok in zip(cases, flags) if not ok]
                                 for method, flags in completed.items()},
            "passed_cases": {method: sum(value is True for value in values) for method, values in outcomes.items()},
            "improvements": int(improvements), "regressions": int(regressions)}
