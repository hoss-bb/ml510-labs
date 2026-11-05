"""Download the pinned MTRAG Cloud evidence and build local retrieval indexes."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import urllib.request
import zipfile

REVISION = "2c618bb98db3c8526433e22d8a2f7320f10a7470"
FILES = {
    "cloud.jsonl.zip": "corpora/passage_level/cloud.jsonl.zip",
    "conversations.json": "mtrag-human/conversations/conversations.json",
    "reference.jsonl": "mtrag-human/generation_tasks/reference.jsonl",
    "queries.jsonl": "mtrag-human/retrieval_tasks/cloud/cloud_lastturn.jsonl",
    "qrels.tsv": "mtrag-human/retrieval_tasks/cloud/qrels/dev.tsv",
    "LICENSE": "LICENSE",
}


def download(root):
    folder = root / "data/raw/mtrag_cloud"
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "download.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if previous and previous.get("revision") != REVISION:
        raise ValueError("Existing MTRAG files use another revision. Preserve them before downloading a different snapshot.")
    records = {}
    for name, remote in FILES.items():
        path = folder / name
        url = f"https://raw.githubusercontent.com/IBM/mt-rag-benchmark/{REVISION}/{remote}"
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".part")
            print("Downloading", name, flush=True)
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
            temporary.replace(path)
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        prior = previous.get("files", {}).get(name)
        if prior and prior["sha256"] != digest:
            raise ValueError(f"Raw file changed: {path}. Restore it before proceeding.")
        records[name] = {"url": url, "sha256": digest, "bytes": path.stat().st_size}
    with zipfile.ZipFile(folder / "cloud.jsonl.zip") as archive:
        members = [n for n in archive.namelist() if n.endswith(".jsonl")]
        if len(members) != 1:
            raise ValueError("Expected one corpus JSONL in the official archive.")
        target = folder / "cloud.jsonl"
        temporary = target.with_suffix(".part")
        if target.exists():
            with archive.open(members[0]) as stream:
                source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
            with target.open("rb") as stream:
                target_hash = hashlib.file_digest(stream, "sha256").hexdigest()
            if source_hash != target_hash:
                raise ValueError("Extracted Cloud corpus differs from its source archive.")
        else:
            with archive.open(members[0]) as source, temporary.open("wb") as output:
                while block := source.read(1024 * 1024):
                    output.write(block)
            temporary.replace(target)
        with target.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
    records["cloud.jsonl"] = {"sha256": digest, "bytes": target.stat().st_size}
    manifest_temp = manifest_path.with_suffix(".part")
    manifest_temp.write_text(json.dumps({"revision": REVISION, "files": records}, indent=2), encoding="utf-8")
    manifest_temp.replace(manifest_path)
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--build-qdrant", action="store_true",
                        help="Build/resume the full local vector collection; may take hours on CPU.")
    args = parser.parse_args()
    if args.download_only and args.build_qdrant:
        parser.error("--download-only cannot be combined with --build-qdrant")
    root = Path(__file__).resolve().parents[1]
    started = perf_counter()
    folder = download(root)
    if not args.download_only:
        from lab_helpers.s8_assistant import verify_data, load_documents, index_directory
        from ai_lib.retrieval.indexing import RetrievalIndex
        report = verify_data(root)
        documents = load_documents(root)
        index = RetrievalIndex(documents)
        index.build_lexical(index_directory(root))
        index.prepare_model()
        if args.build_qdrant:
            from qdrant_client import QdrantClient
            from ai_lib.retrieval.qdrant import QdrantIndex
            vector_folder = index_directory(root) / "qdrant"
            print("Close other clients using this local Qdrant folder before preparation.", flush=True)
            client = QdrantClient(path=str(vector_folder))
            try:
                vector_report = QdrantIndex(client, index).prepare()
            finally:
                client.close()
            vector_report["database_bytes"] = sum(p.stat().st_size for p in vector_folder.rglob("*") if p.is_file())
            (index_directory(root) / "qdrant-build.json").write_text(
                json.dumps(vector_report, indent=2), encoding="utf-8")
            report["qdrant"] = vector_report
        report["preparation_seconds"] = perf_counter() - started
        report["index_bytes"] = sum(p.stat().st_size for p in index_directory(root).rglob("*") if p.is_file())
        (root / "data/processed/mtrag_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({key: report[key] for key in
                         ["passages", "conversations", "teaching_tasks", "preparation_seconds", "index_bytes"]}, indent=2))
    print("Prepared", folder)


if __name__ == "__main__":
    main()
