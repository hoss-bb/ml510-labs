"""Execute notebooks on the user's local data. No installs or downloads."""
import argparse
import json
import os
from pathlib import Path
from time import perf_counter
import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lab", choices=["s4", "s5", "s6", "s7", "s8", "s9"], default="s4")
    parser.add_argument("--scope", choices=["core", "all"], default=None,
                        help="S4 defaults to core (00-03); other sessions run all stages.")
    parser.add_argument("--kernel", default="ml510")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Per-cell timeout in seconds (default: S4 7200; S5 43200 for full-data CPU training).")
    args = parser.parse_args()
    timeout = args.timeout if args.timeout is not None else (7200 if args.lab == "s4" else 43200)
    root = Path(__file__).resolve().parents[1]
    names = [
        "00_start_here.ipynb",
        "01_linear_regression.ipynb",
        "02_trees_and_ensembles.ipynb",
        "03_lagged_forecasting.ipynb",
        "04_sequence_forecasting.ipynb",
        "05_model_capacity.ipynb",
    ]
    scope = args.scope or ("core" if args.lab == "s4" else "all")
    if args.lab in {"s5", "s6"}:
        if scope != "all":
            parser.error("S5 and S6 run all six stages; use --scope all.")
        names = ["00_start_here.ipynb", "01_logistic_regression.ipynb", "02_trees_and_ensembles.ipynb",
                 "03_neural_models.ipynb", "04_probabilities.ipynb", "05_review_policy.ipynb"]
        if args.lab == "s6":
            names = ["00_start_here.ipynb", "01_operating_regimes.ipynb", "02_challenge_groups.ipynb",
                     "03_anomaly_references.ipynb", "04_autoencoder.ipynb", "05_temporal_autoencoder.ipynb"]
    elif args.lab == "s7":
        if scope != "all":
            parser.error("S7 runs all five stages; use --scope all.")
        names = ["00_validation.ipynb", "01_learning.ipynb", "02_groups.ipynb",
                 "03_behavior.ipynb", "04_decision.ipynb"]
    elif scope == "core":
        names = names[:4]
    if args.lab == "s8":
        if scope != "all":
            parser.error("S8 follows all six stages; use --scope all.")
        names = ["00_start_here.ipynb", "01_chains.ipynb", "02_retrieval.ipynb",
                 "03_conversation.ipynb", "04_tools.ipynb", "05_graphs.ipynb"]
        if args.timeout is None:
            timeout = 1800
    if args.lab == "s9":
        if scope != "all":
            parser.error("S9 follows all six stages; use --scope all.")
        names = ["00_start_here.ipynb", "01_evaluation_cases.ipynb", "02_rag_evaluation.ipynb",
                 "03_improve_rag.ipynb", "04_guardrails.ipynb", "05_controlled_assistant.ipynb"]
        if args.timeout is None:
            timeout = 7200
    folder = {"s4": "s4_regression", "s5": "s5_classification", "s6": "s6_unsupervised", "s7": "s7_evaluation", "s8": "s8_assistant", "s9": "s9_evaluation"}[args.lab]
    files = [root / "notebooks" / folder / name for name in names]
    missing = [path.name for path in files if not path.is_file()]
    if missing:
        parser.error(
            f"Missing {args.lab.upper()} notebooks: " + ", ".join(missing) + ". "
            f"Download {folder} from Moodle, extract it if zipped, and copy the folder "
            f"into notebooks/ so notebooks/{folder}/{names[0]} exists. "
            "See notebooks/README.md."
        )
    out = root / "notebooks" / f"{args.lab}_corrections"
    if args.lab == "s7":
        from lab_helpers.s7_evaluation import load_evidence
        load_evidence(root / "data/processed/s7_evidence")
    for path in files:
        destination = out / path.name
        if destination.exists():
            existing = nbformat.read(destination, as_version=4)
            if existing.metadata.get("instructor_corrections") or any(
                cell.metadata.get("instructor_note") for cell in existing.cells
            ):
                parser.error(
                    f"Preserve or rename the annotated instructor copy before regenerating: {destination}"
                )
    out.mkdir(parents=True, exist_ok=True)
    previous_run = None
    for phase, path in enumerate(files):
        print("Executing", path.name, flush=True)
        book = nbformat.read(path, as_version=4)
        environment = dict(os.environ)
        environment.pop("ML510_S5_SOURCE_RUN", None)
        environment.pop("ML510_S6_SOURCE_RUN", None)
        environment.pop("ML510_S9_SOURCE_RUN", None)
        handoff_start = 1 if args.lab == "s9" else (3 if args.lab == "s5" else 4)
        handoff_variable = f"ML510_{args.lab.upper()}_SOURCE_RUN"
        if args.lab in {"s5", "s6", "s9"}:
            if phase >= handoff_start:
                environment[handoff_variable] = previous_run
        if args.lab in {"s5", "s6", "s7", "s9"}:
            # Read the run created by this kernel, never a directory named "latest".
            book.cells.append(nbformat.v4.new_code_cell("import json\nprint(json.dumps(str(run.directory)))"))
        started = perf_counter()
        def checkpoint(**kwargs):
            nbformat.write(book, out / path.name)
        client = NotebookClient(book, kernel_name=args.kernel, timeout=timeout,
                                resources={"metadata": {"path": str(root)}}, on_cell_executed=checkpoint)
        client.execute(env=environment)
        if args.lab in {"s5", "s6", "s7", "s9"}:
            capture = book.cells.pop()
            text = "".join(o.get("text", "") for o in capture.outputs if o.output_type == "stream").strip()
            previous_run = json.loads(text)
            book.metadata["artifact_run"] = str(Path(previous_run).relative_to(root).as_posix())
            if args.lab in {"s5", "s6"} and phase >= handoff_start:
                book.metadata["source_run"] = environment[handoff_variable]
        book.metadata["execution_seconds"] = perf_counter() - started
        nbformat.write(book, out / path.name)
        print("Saved", out / path.name, "seconds:", round(book.metadata["execution_seconds"], 2), flush=True)


if __name__ == "__main__":
    main()
