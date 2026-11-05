"""BM25 retrieval and local candidate ranking."""
import hashlib
import json
from pathlib import Path
import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from .windows import tokenize, token_windows, token_window_spans


EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


class RetrievalIndex:
    """Search the full corpus lexically, then embed only the chosen candidates."""

    def __init__(self, documents, *, model_name=EMBEDDING_MODEL):
        if model_name != EMBEDDING_MODEL:
            raise ValueError(f"This index supports {EMBEDDING_MODEL}; use its matching prepared vectors.")
        self.documents = list(documents)
        ids = [d.metadata.get("passage_id") for d in self.documents]
        if not ids or any(not x for x in ids) or len(set(ids)) != len(ids):
            raise ValueError("Provide nonempty documents with unique passage_id metadata.")
        if not any(d.page_content.strip() for d in self.documents):
            raise ValueError("Provide at least one nonempty passage.")
        self.model_name = model_name
        self.fingerprint = hashlib.sha256(json.dumps(
            [(d.metadata, d.page_content) for d in self.documents],
            sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        self._bm25 = None
        self._embedder = None
        self._tokenizer = None
        self.folder = None

    @property
    def configuration(self):
        return {"format": 3, "corpus_sha256": self.fingerprint, "model": self.model_name,
                "method": "bm25_then_local", "window_tokens": 384, "overlap_tokens": 64,
                "normalization": "l2", "dimensions": 384,
                "query_prefix": "Represent this sentence for searching relevant passages: "}

    def build_lexical(self, folder):
        """Prepare or reuse the lexical index. No embedding or API calls occur."""
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        settings = self.folder / "retrieval.json"
        if settings.exists() and json.loads(settings.read_text(encoding="utf-8")) != self.configuration:
            raise ValueError("Retrieval settings or corpus changed. Prepare a matching cache directory.")
        self._write_json(settings, self.configuration)
        path = self.folder / "lexical.json"
        if path.exists():
            try:
                saved = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                raise ValueError("Invalid lexical cache. Remove lexical.json and prepare it again.") from error
            if not isinstance(saved, dict) or saved.get("corpus_sha256") != self.fingerprint:
                raise ValueError("Lexical index belongs to a different corpus.")
            tokens = saved.get("tokens")
            if (not isinstance(tokens, list) or len(tokens) != len(self.documents)
                    or any(not isinstance(row, list) or any(not isinstance(t, str) for t in row)
                           for row in tokens)):
                raise ValueError("Invalid lexical cache. Remove lexical.json and prepare it again.")
        else:
            tokens = [tokenize(d.page_content) for d in self.documents]
            self._write_json(path, {"corpus_sha256": self.fingerprint, "tokens": tokens})
        self._bm25 = BM25Okapi(tokens)

    @staticmethod
    def _write_json(path, value):
        temporary = path.with_suffix(".part")
        temporary.write_text(json.dumps(value), encoding="utf-8")
        temporary.replace(path)

    def prepare_model(self):
        """Download/load the small CPU model; do not embed the corpus."""
        if self.folder is None:
            raise ValueError("Build the lexical index before preparing local embeddings.")
        if self._embedder is None:
            from fastembed import TextEmbedding
            self._embedder = TextEmbedding(model_name=self.model_name,
                                           cache_dir=str(self.folder / "model"), threads=4)
        if self._tokenizer is None:
            from tokenizers import Tokenizer
            self._tokenizer = Tokenizer.from_str(self._embedder.model.tokenizer.to_str())
            self._tokenizer.no_truncation()
            self._tokenizer.no_padding()

    @staticmethod
    def _normalize(vectors):
        norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
        if not np.isfinite(vectors).all() or (norms == 0).any():
            raise ValueError("Embeddings must be finite and nonzero.")
        return vectors / norms

    def _vectors(self, text, *, query=False):
        self.prepare_model()
        # The separate namespace prevents query-prefix vectors becoming document vectors.
        kind = "queries" if query else "passages"
        cache = self.folder / kind
        cache.mkdir(exist_ok=True)
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        path = cache / f"{key}.npy"
        if path.exists():
            try:
                vectors = np.load(path, allow_pickle=False)
            except (ValueError, OSError) as error:
                raise ValueError(f"Invalid embedding cache: {path}. Remove this file and rerun.") from error
        else:
            if query:
                pieces = [self.configuration["query_prefix"] + text]
            else:
                pieces = token_windows(text, self._tokenizer)
            vectors = self._normalize(np.asarray(list(self._embedder.embed(pieces, batch_size=4)), dtype=np.float32))
            temporary = path.with_suffix(".part")
            with temporary.open("wb") as stream:
                np.save(stream, vectors, allow_pickle=False)
            temporary.replace(path)
        expected_rows = 1 if query else len(token_window_spans(text, self._tokenizer))
        if (vectors.shape != (expected_rows, self.configuration["dimensions"])
                or not np.issubdtype(vectors.dtype, np.floating)
                or not np.isfinite(vectors).all()
                or not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)):
            raise ValueError(f"Invalid cached local embeddings: {path}. Remove this file and rerun.")
        return vectors

    def embed_query(self, query):
        """Use the same query instruction for candidate and full-corpus search."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Provide a nonempty query.")
        return self._vectors(query, query=True)[0]

    def embed_passages(self, texts, *, batch_size=16):
        """Embed a bounded batch of passages, reusing compatible local vectors."""
        self.prepare_model()
        cache = self.folder / "passages"
        cache.mkdir(exist_ok=True)
        paths = [cache / (hashlib.sha256(text.encode('utf-8')).hexdigest() + '.npy')
                 for text in texts]
        missing = {}
        pieces = []
        for text, path in zip(texts, paths):
            if not path.exists() and path not in missing:
                windows = token_windows(text, self._tokenizer)
                missing[path] = (len(pieces), len(pieces) + len(windows))
                pieces.extend(windows)
        if pieces:
            vectors = self._normalize(np.asarray(
                list(self._embedder.embed(pieces, batch_size=batch_size)), dtype=np.float32))
            for path, (start, end) in missing.items():
                temporary = path.with_suffix('.part')
                with temporary.open('wb') as stream:
                    np.save(stream, vectors[start:end], allow_pickle=False)
                temporary.replace(path)
        return [self._vectors(text) for text in texts]

    def search(self, query, *, method="bm25", k=5, candidate_k=30):
        """Return original passages; semantic ranking cannot recover missed candidates."""
        if not isinstance(query, str) or not query.strip() or type(k) is not int or k < 1:
            raise ValueError("Search needs a nonempty query and positive integer k.")
        if method not in {"bm25", "local"}:
            raise ValueError("Retrieval method must be bm25 or local.")
        if type(candidate_k) is not int or candidate_k < k:
            raise ValueError("candidate_k must be an integer at least as large as k.")
        if self._bm25 is None:
            raise ValueError("Build the lexical index before searching.")
        lexical = np.asarray(self._bm25.get_scores(tokenize(query)), dtype=float)
        order = [i for i in np.argsort(-lexical, kind="stable") if self.documents[i].page_content.strip()]
        selected = order[:k if method == "bm25" else candidate_k]
        scores = [float(lexical[i]) for i in selected]
        if method == "local":
            query_vector = self._vectors(query, query=True)[0]
            scores = [float(np.max(self._vectors(self.documents[i].page_content) @ query_vector)) for i in selected]
        ranked = np.argsort(-np.asarray(scores), kind="stable")[:k]
        result = []
        for position in ranked:
            i = selected[position]
            result.append(Document(page_content=self.documents[i].page_content,
                          metadata={**self.documents[i].metadata, "score": scores[position],
                                    "bm25_score": float(lexical[i]), "candidate_rank": int(position) + 1,
                                    "retriever": "bm25" if method == "bm25" else "bm25_then_local"}))
        return result
