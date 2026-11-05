"""S9 cases and recording; application and evaluation choices stay visible."""
from dataclasses import asdict, replace
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import re

import pandas as pd
from pydantic import BaseModel, Field

from ai_lib.models.configuration import load_settings
from ai_lib.models.providers import build_chat_model
from ai_lib.experiment.serialization import fingerprint, save_json
from ai_lib.experiment.caching import ResultCache
from ai_lib.evaluation.reports import result_summary
from ai_lib.experiment.recording import LocalTrace
from ml_lib.experiment.artifacts import RunArtifacts
from lab_helpers.s8_assistant import (project_root, raw_directory, load_tasks, load_documents,
    reference_answer, question)


def prepare_cases(root):
    """Select conversations before outputs; use metadata to cover case categories."""
    download = json.loads((raw_directory(root)/"download.json").read_text(encoding="utf-8"))
    for name, record in download["files"].items():
        with (raw_directory(root)/name).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != record["sha256"]:
            raise ValueError(f"Raw MTRAG file changed: {name}. Restore the verified download first.")
    tasks = load_tasks(root)
    familiar = set(json.loads((root / "configs/s8_examples.json").read_text())["teaching_conversations"])
    ids = sorted({t["conversation_id"] for t in tasks} - familiar)
    random.Random(42).shuffle(ids)
    split_at = len(ids) // 2
    cases = []
    for split, conversations in (("development", ids[:split_at]), ("reserved", ids[split_at:])):
        for conversation in conversations:
            candidates = sorted([t for t in tasks if t["conversation_id"] == conversation], key=lambda t: int(t["turn"]))
            selected, seen = [], set()
            while candidates and len(selected) < 3:
                def categories(task):
                    return {"first" if int(task["turn"]) == 1 else "followup", *task["Answerability"], *task["Question Type"]}
                best = max(candidates, key=lambda t: len(categories(t)-seen))
                selected.append(best)
                seen.update(categories(best))
                candidates.remove(best)
            for task in sorted(selected, key=lambda t: int(t["turn"])):
                cases.append({"case_id": task["task_id"], "conversation_id": conversation,
                    "split": split, "origin": "mtrag", "turn": int(task["turn"]),
                    "question": question(task), "history": task["input"][:-1],
                    "expected_answer": reference_answer(task),
                    "source_ids": [c["document_id"] for c in task["contexts"]],
                    "answerability": task["Answerability"], "question_type": task["Question Type"]})
    corpus = fingerprint(json.loads((raw_directory(root)/"download.json").read_text()))
    bundle = {"seed": 42, "dataset": corpus, "cases": cases}
    path = root / "data/processed/s9_evaluation/cases.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != bundle:
        raise ValueError("The frozen case selection differs. Preserve it before preparing a new campaign.")
    save_json(path, bundle)
    return bundle


def role_settings(root):
    base = load_settings(root)
    result = {"application": base}
    for role in ("judge", "synthetic"):
        prefix = "S9_" + role.upper()
        provider = os.getenv(prefix+"_PROVIDER", base.provider)
        model = os.getenv(prefix+"_MODEL", base.model)
        budget = 0 if provider == "gemini" and model.startswith("gemini-2.5-flash") else None
        result[role] = replace(base, provider=provider, model=model, thinking_budget=budget,
                               max_tokens=None if provider == "cohere" else 4096)
    return result


def setup_run(label):
    root = project_root()
    logging.getLogger("google_genai.models").setLevel(logging.ERROR)
    settings = role_settings(root)
    bundle = prepare_cases(root)
    config = {role: asdict(value) for role, value in settings.items()}
    run = RunArtifacts(root, label, config=config, dataset_path=raw_directory(root)/"cloud.jsonl")
    run.record("cases", bundle)
    run.record("experiment", {"models": config, "selection_hash": fingerprint(bundle),
                              "upload_results": False, "status": "in_progress"})
    trace = LocalTrace(run, configuration_id=fingerprint(config))
    cache = ResultCache(root / "data/processed/s9_evaluation/calls")
    models = {role: build_chat_model(value) for role, value in settings.items()}
    models["judge"].callbacks = [trace]
    return root, run, trace, cache, models, config, bundle


def load_source(root, source_run, bundle):
    if not source_run:
        raise ValueError("Set source_run to the preceding notebook's printed artifact path.")
    folder = (root / source_run).resolve()
    if not folder.is_relative_to((root / "artifacts").resolve()):
        raise ValueError("Choose a run inside artifacts/.")
    if not (folder/"status.json").exists() or not json.loads((folder/"status.json").read_text())["completed"]:
        raise ValueError("The source run is incomplete; finish the preceding notebook first.")
    saved = json.loads((folder/"cases.json").read_text(encoding="utf-8"))
    if saved != bundle:
        raise ValueError("Source dataset or selected cases differ.")
    return folder


def conversation_text(case):
    preceding = "\n".join(f"{x['speaker']}: {x['text']}" for x in case["history"])
    return preceding + "\nuser: " + case["question"]


def case_documents(case, lookup):
    missing = set(case["source_ids"]) - lookup.keys()
    if missing:
        raise ValueError("Reference passages are missing from the prepared corpus.")
    return [lookup[key] for key in case["source_ids"]]


def save_rows(run, rows, name="results"):
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.drop_duplicates(["case_id", "method"], keep="last")
    run.save_table(name, frame)
    if name == "results" and not frame.empty:
        run.save_table("summary", result_summary(frame.to_dict("records")))
    return frame


def finish(run, root, finding):
    from IPython.display import Markdown, display
    results_path = run.directory / "tables/results.csv"
    if not results_path.exists():
        raise RuntimeError("No evaluation results were saved. Execute the experiment before finishing.")
    results = pd.read_csv(results_path)
    failed = results.status.ne("ok").any()
    if "assessment_gate" in results:
        failed = failed or results.assessment_gate.eq("ERROR").any()
    if "service_failure" in results:
        failed = failed or results.service_failure.fillna(False).any()
    expected_path = run.directory / "expected-cases.json"
    if expected_path.exists():
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        observed = set(zip(results.case_id, results.method))
        failed = failed or any((case_id, method) not in observed
            for method, ids in expected.items() for case_id in ids)
    # Some methods deliberately have fewer metrics; inspect each saved judgment instead.
    answers_path = run.directory / "answers.json"
    if answers_path.exists():
        answers = json.loads(answers_path.read_text(encoding="utf-8"))
        failed = failed or any(j["status"] != "ok" for a in answers.values() for j in a["judgments"].values())
    if failed:
        run.record("status", {"completed": False, "reason": "application or evaluator calls failed"})
        run.write_overview("Incomplete S9 experiment", "Inspect failed calls, then rerun after fixing provider access or configuration.")
        raise RuntimeError("Some application or evaluator calls failed. This run cannot be used as a completed handoff.")
    run.record("status", {"completed": True})
    overview = run.write_overview("Inspect this S9 experiment", finding,
        inspect_first=["tables/summary.csv", "tables/results.csv", "cases.json", "trace.json"])
    relative = "../../" + overview.relative_to(root).as_posix()
    display(Markdown(f"{finding}\n\n[Open saved evidence]({relative})"))
    print("source_run =", repr(run.directory.relative_to(root).as_posix()))


class SyntheticCase(BaseModel):
    question: str = Field(min_length=10)
    expected_answer: str = Field(min_length=10)
    required_facts: list[str] = Field(min_length=1)
    source_ids: list[str] = Field(min_length=1)
    supporting_excerpts: list[str] = Field(min_length=1)


class ProposedCases(BaseModel):
    cases: list[SyntheticCase] = Field(min_length=2, max_length=2)


class CaseReview(BaseModel):
    supported: bool
    unambiguous: bool
    reason: str


def validate_synthetic(case, lookup):
    problems = []
    if not case.source_ids or any(key not in lookup for key in case.source_ids):
        problems.append("unknown source")
    texts = [lookup[key].page_content for key in case.source_ids if key in lookup]
    if not case.supporting_excerpts or any(not quote.strip() or not any(quote in text for text in texts)
                                          for quote in case.supporting_excerpts):
        problems.append("supporting excerpt is not verbatim in a cited source")
    return problems


def normalized_question(text):
    return " ".join(re.findall(r"\w+", text.lower()))


def near_duplicate(question, accepted, index, threshold=0.95, *, vectors=None):
    if not accepted:
        return None
    if vectors is None:
        vectors = {}
    def embedded(text):
        if text not in vectors:
            vectors[text] = index.embed_query(text)
        return vectors[text]
    vector = embedded(question)
    for previous in accepted:
        if normalized_question(question) == normalized_question(previous["question"]):
            return previous["case_id"]
        if float(vector @ embedded(previous["question"])) >= threshold:
            return previous["case_id"]
    return None


def assess_result(case, docs, result, method, *, rag=False, reuse=True,
                  cache, model_config, rubrics, evaluators, rows, answers, run, evaluator_types=None):
    from ai_lib.guardrails.answers import check_answer
    from ai_lib.evaluation.metrics import judge_score
    from ai_lib.retrieval.formatting import format_context
    from ai_lib.evaluation.schemas import evaluator_schema
    evaluator_types = evaluator_types or {}
    for name, kind in evaluator_types.items():
        evaluator_schema(kind)
        if name not in evaluators:
            raise ValueError(f"Unknown evaluator: {name}")
        if kind == "ordinal" and name != "correctness":
            raise ValueError("The S9 ordinal anchors apply to correctness only.")
    names = ["correctness", "relevance"]
    if rag:
        names += ["groundedness", "retrieval_relevance"]
    answer = result.get("value", {})
    record = {"case_id": case["case_id"], "method": method,
        "reference_hash": fingerprint(case["expected_answer"]), "origin": case["origin"],
        "status": result["status"], "service_failure": bool(result.get("failure_reason")), "execution": result["execution"], "seconds": result["seconds"],
        "disposition": answer.get("disposition", "error") if result["status"] == "ok" else "error",
        **check_answer(answer, [d.metadata["passage_id"] for d in docs])}
    from ai_lib.evaluation.metrics import ordinal_score
    from ai_lib.guardrails.answers import assessment_gate
    gate = assessment_gate(case, answer, docs) if result["status"] == "ok" else {"verdict": "ERROR", "reason": "Application execution failed"}
    record["assessment_gate"] = gate["verdict"]
    record["correctness_ordinal"] = None
    judgments = {}
    for name in names:
        record[name] = None
        if result["status"] != "ok" or gate["verdict"] != "PASS":
            continue
        inputs = conversation_text(case)
        output = answer["answer"]
        if name in {"groundedness", "correctness"}:
            inputs += "\nEvidence:\n" + format_context(docs)
        if name == "retrieval_relevance":
            output = format_context(docs)
        arguments = {"inputs": inputs, "outputs": output,
                     "reference_outputs": case["expected_answer"]}
        judged = cache.call(arguments, {"stage": "judge", "model": model_config["judge"],
            "rubric": rubrics[name], "schema": evaluator_schema(evaluator_types.get(name, "binary"))}, lambda: evaluators[name](**arguments), reuse=reuse)
        judgments[name] = judged
        if judged["status"] == "ok":
            if evaluator_types.get(name, "binary") == "ordinal":
                record["correctness_ordinal"] = ordinal_score(judged["value"]["score"])
                record[name] = record["correctness_ordinal"] == 3
            else:
                record[name] = judge_score(judged["value"]["score"])
    rows[:] = [row for row in rows if (row["case_id"], row["method"]) != (case["case_id"], method)]
    rows.append(record)
    answers[case["case_id"]+"/"+method] = {"case": case, "response": result,
        "evidence": [{"id": d.metadata["passage_id"], "text": d.page_content} for d in docs],
        "judgments": judgments}
    run.record("answers", answers)
    save_rows(run, rows)
    print(method, case["case_id"].split("<::>")[-1], record["status"], flush=True)
    return record
