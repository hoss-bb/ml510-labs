"""Compare two prompt examples on fixed constructed cases, without deployment changes."""
from .templates import assemble
import pandas as pd
from ai_lib.models.errors import expected_call_error
from lab_helpers.s10_shared.reviews import chain
from lab_helpers.s10_shared.examples import load_example_listing

CASES = [
    ("conflict", "Guests must book more than 14 nights.", ["confirm_facts"]),
    ("legitimate", "Guests must book at least 14 nights.", ["no_change", "guest_note"]),
]


def compare(run, generation, runner=chain.review, *, variants=None):
    variants = variants if variants is not None else {name: assemble(name) for name in ["starting", "candidate"]}
    example = load_example_listing()
    rows = []
    for variant in ["starting", "candidate"]:
        instructions = variants[variant]
        for name, description, expected in CASES:
            snapshot = {"listing": {**example, "description": description, "advertised_price": 220},
                        "prediction": {"estimate": 170, "currency": "CAD"},
                        "review_context": {"fields": {"minimum_nights": 14,
                            "amenities": None, "bathroom_description": None}, "provenance": None},
                        "instructions": instructions, "ai_settings": generation}
            stages, output, error = {}, None, None
            try:
                output = runner(snapshot, lambda stage, value: stages.update({stage: value}), instructions=instructions)
            except Exception as exc:
                error = getattr(exc, "code", type(exc).__name__)
                if not expected_call_error(exc):
                    run.record(f"{variant}-{name}", {"snapshot": snapshot, "instructions": instructions,
                               "stages": stages, "error": error})
                    raise
            run.record(f"{variant}-{name}", {"snapshot": snapshot, "instructions": instructions,
                       "stages": stages, "output": output, "error": error})
            rows.append({"variant": variant, "case": name, "expected": ", ".join(expected),
                         "action": output["action"] if output else None, "error": error,
                         "action_matches": output["action"] in expected if output else False})
    table = pd.DataFrame(rows)
    run.save_table("comparison", table)
    run.write_overview("Prompt comparison", "Constructed development cases. Inspect supporting facts and drafts, not only action matches. Failed calls are not correct answers. Nothing is deployed.")
    return table
