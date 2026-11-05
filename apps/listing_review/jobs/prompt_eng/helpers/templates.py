"""Assemble editable job templates; runtime reviews use instructions saved with each job."""
from pathlib import Path
from lab_helpers.s10_shared.reviews.prompts import STAGES


def assemble(variant):
    if variant not in {"starting", "candidate"}:
        raise ValueError("Choose starting or candidate templates.")
    folder = Path(__file__).resolve().parents[1] / "templates"
    policy = (folder / "policy.txt").read_text(encoding="utf-8")
    name = "examples" if variant == "starting" else "candidate_examples"
    examples = (folder / f"{name}.txt").read_text(encoding="utf-8")
    return {stage: policy + "\n" + examples + "\n" + (folder / f"{stage}.txt").read_text(encoding="utf-8") for stage in STAGES}
