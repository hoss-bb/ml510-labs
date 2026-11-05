"""Compare fixed prompt candidates and export the selection; never deploy it automatically."""
from pathlib import Path
from ai_lib.evaluation.reports import paired_outcomes
from .compare_prompts import compare, CASES
from .templates import assemble
from lab_helpers.s10_shared.reviews.configuration import generation_settings
from lab_helpers.s10_shared.reviews.prompts import save_prompts
from lab_helpers.s10_shared.runtime.settings import ROOT, settings
from ml_lib.experiment.artifacts import RunArtifacts


def select_prompt(table):
    outcomes = table.rename(columns={"variant": "method", "case": "case_id", "action_matches": "passed"}).copy()
    # A missing application action is incomplete execution, not a failed judgment.
    outcomes.loc[outcomes.action.isna(), "error"] = "missing_action"
    comparison = paired_outcomes(outcomes, expected_cases=[name for name, _, _ in CASES],
                                 baseline="starting", candidate="candidate")
    if not comparison["complete"]:
        return {"complete": False, "selected": None, "reason": "Execution incomplete; no prompts exported."}
    counts, regressions = comparison["passed_cases"], comparison["regressions"]
    selected = "candidate" if counts["candidate"] > counts["starting"] and regressions == 0 else "starting"
    return {"complete": True, "selected": selected, "passed_cases": counts,
            "case_count": comparison["case_count"], "regressions": regressions,
            "reason": "Candidate requires more passes and no regression; otherwise retain starting prompts."}


def export_selection(run, table, generation, variants, output_root):
    selection = select_prompt(table)
    run.record("selection", selection)
    if not selection["complete"]:
        run.write_overview("Prompt experiment incomplete", selection["reason"])
        return None
    target = output_root / run.directory.name
    save_prompts(target, variants[selection["selected"]])
    run.record("experiment", {
        "generation": generation, "selection": selection,
        "evaluation": str(run.directory.relative_to(ROOT)),
        "scope": "Two constructed development cases; inspect drafts and support before deployment."})
    run.record("selected_prompts", {"path": str(target), "selection": selection})
    run.write_overview("Prompt selection", f"Selected {selection['selected']}. Inspect unsuccessful checks and full drafts; selection on two cases is not general quality assurance. Active prompts were not replaced.")
    return target


def run_job():
    config = settings()
    generation = generation_settings()
    variants = {name: assemble(name) for name in ["starting", "candidate"]}
    run = RunArtifacts(ROOT, "s10_prompt_eng", config={"generation": generation, "cases": "constructed stay-condition pair"},
                       dataset_path=Path(__file__).with_name("compare_prompts.py"))
    try:
        table = compare(run, generation, variants=variants)
    except Exception as error:
        run.record("selection", {"complete": False, "selected": None, "error_type": type(error).__name__})
        run.write_overview("Prompt experiment incomplete", "Execution stopped unexpectedly. No prompts exported; inspect the saved case and traceback.")
        raise
    target = export_selection(run, table, generation, variants, config.data / "prompt_candidates")
    print(table.to_string(index=False))
    print(f"Evaluation: {run.directory}")
    if target is None:
        raise SystemExit("Incomplete experiment: no selected prompts. Inspect the saved failures.")
    print(f"Selected prompts: {target}\nActive prompts were not replaced.")
