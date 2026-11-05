"""MTRAG examples and artifact setup; application logic stays in notebooks."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
import pandas as pd

from ai_lib.models.configuration import load_settings
from ai_lib.models.providers import build_chat_model
from ai_lib.experiment.recording import LocalTrace
from ml_lib.experiment.artifacts import RunArtifacts


def project_root():
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src").is_dir():
            return candidate
    raise FileNotFoundError("Open this notebook inside the ml510-labs repository.")


def raw_directory(root):
    folder = Path(root) / "data/raw/mtrag_cloud"
    if not (folder / "cloud.jsonl").is_file():
        raise FileNotFoundError("Run python scripts/prepare_s8.py from the repository root first.")
    return folder


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_documents(root):
    documents = []
    for row in read_jsonl(raw_directory(root) / "cloud.jsonl"):
        documents.append(Document(page_content=row["text"], metadata={
            "passage_id": row["_id"], "url": row.get("url", ""), "title": row.get("title", ""),
        }))
    return documents


def load_tasks(root):
    return [task for task in read_jsonl(raw_directory(root) / "reference.jsonl")
            if "ibmcloud" in task["Collection"]]


def teaching_tasks(root):
    tasks = load_tasks(root)
    selection = json.loads((Path(root) / "configs/s8_examples.json").read_text(encoding="utf-8"))
    ids = set(selection["teaching_conversations"])
    return [task for task in tasks if task["conversation_id"] in ids]


def load_qrels(root):
    rows = pd.read_csv(raw_directory(root) / "qrels.tsv", sep="\t")
    return {str(query): set(group.loc[group["score"] > 0, "corpus-id"])
            for query, group in rows.groupby("query-id", sort=False)}


def history_messages(task):
    """Only preceding turns, never the current reference answer."""
    messages = []
    for item in task["input"][:-1]:
        cls = HumanMessage if item["speaker"] == "user" else AIMessage
        messages.append(cls(content=item["text"]))
    return messages


def question(task):
    return task["input"][-1]["text"]


def reference_answer(task):
    return "\n".join(target["text"] for target in task["targets"])


def reference_documents(task, documents):
    lookup = {d.metadata["passage_id"]: d for d in documents}
    return [lookup[c["document_id"]] for c in task["contexts"]]


def passage_table(documents):
    return pd.DataFrame([{"passage_id": d.metadata["passage_id"],
                          "score": d.metadata.get("score"), "excerpt": d.page_content.strip()[:180]}
                         for d in documents])


def retrieval_evidence(task, retrieved, qrels):
    relevant = qrels.get(task["task_id"])
    if not relevant:
        return {"task_id": task["task_id"], "judged": False, "hit_at_5": None, "recall_at_5": None}
    found = {d.metadata["passage_id"] for d in retrieved[:5]} & relevant
    return {"task_id": task["task_id"], "judged": True,
            "hit_at_5": int(bool(found)), "recall_at_5": len(found) / len(relevant)}


def verify_data(root):
    folder = raw_directory(root)
    download = json.loads((folder / "download.json").read_text(encoding="utf-8"))
    for name, record in download["files"].items():
        with (folder / name).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != record["sha256"]:
            raise ValueError(f"Raw dataset file changed: {name}")
    documents, tasks = load_documents(root), load_tasks(root)
    ids = [d.metadata["passage_id"] for d in documents]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Cloud passage IDs must be nonempty and unique.")
    task_ids = {t["task_id"] for t in tasks}
    if len(task_ids) != len(tasks):
        raise ValueError("Duplicate task IDs.")
    known = set(ids)
    for task in tasks:
        for context in task["contexts"]:
            if context["document_id"] not in known:
                raise ValueError(f"Unresolved reference passage: {context['document_id']}")
    qrels = load_qrels(root)
    if not set(qrels).issubset(task_ids) or any(not values.issubset(known) for values in qrels.values()):
        raise ValueError("Relevance judgments do not match the corpus and tasks.")
    selected = teaching_tasks(root)
    teaching_ids = {t["conversation_id"] for t in selected}
    all_ids = {t["conversation_id"] for t in tasks}
    if len(teaching_ids) != 6:
        raise ValueError("Expected six teaching conversations.")
    return {"revision": download["revision"], "files": download["files"], "passages": len(ids),
            "published_passage_count": 61022, "empty_passages": sum(not d.page_content.strip() for d in documents),
            "tasks": len(tasks), "conversations": len(all_ids), "teaching_tasks": len(selected),
            "teaching_conversations": sorted(teaching_ids), "reserved_conversations": sorted(all_ids - teaching_ids)}


def index_directory(root):
    with (raw_directory(root) / "cloud.jsonl").open("rb") as stream:
        fingerprint = hashlib.file_digest(stream, "sha256").hexdigest()[:12]
    return Path(root) / "data/processed/s8_indexes" / f"{fingerprint}-bm25-local-v3"


def setup_run(label):
    root = project_root()
    settings = load_settings(root)
    dataset = raw_directory(root) / "cloud.jsonl"
    run = RunArtifacts(root, label, config=asdict(settings), dataset_path=dataset)
    run.record("experiment", {"model": asdict(settings), "execution": "live",
                              "data": json.loads((dataset.parent / "download.json").read_text(encoding="utf-8"))})
    from importlib.metadata import version
    run.record("ai-packages", {name: version(name) for name in ["langchain-core", "langchain-google-genai",
               "langchain-groq", "langchain-cohere", "langchain-tavily", "langgraph", "fastembed", "qdrant-client"]})
    trace = LocalTrace(run)
    model = build_chat_model(settings)
    return root, run, trace, model


def finish_run(run, title, summary):
    from IPython.display import Markdown, display
    run.record("status", {"completed": True})
    overview = run.write_overview(title, summary, inspect_first=["tables/summary.csv", "answers.json", "trace.json"])
    relative = "../../" + overview.relative_to(project_root()).as_posix()
    display(Markdown(f"[{title}: open saved results]({relative})"))


def open_vector_index(root, embeddings):
    """Open the prepared collection; the caller closes the returned client."""
    from qdrant_client import QdrantClient
    from ai_lib.retrieval.qdrant import QdrantIndex
    folder = index_directory(root) / "qdrant"
    if not folder.exists():
        raise FileNotFoundError("Import the Moodle index with scripts/s8_qdrant.py first.")
    try:
        client = QdrantClient(path=str(folder))
    except RuntimeError as error:
        raise RuntimeError("Close the notebook or process using this Qdrant folder, then rerun setup.") from error
    try:
        index = QdrantIndex(client, embeddings)
        state = index.open()
    except Exception:
        client.close()
        raise
    return client, index, state
