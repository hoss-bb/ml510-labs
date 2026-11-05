"""Persistent vector search using caller-owned Qdrant clients."""
from time import perf_counter
from uuid import NAMESPACE_URL, uuid5
from langchain_core.documents import Document
import numpy as np
from qdrant_client import models
from .windows import token_window_spans


class QdrantIndex:
    """Store passage windows in a caller-owned local or server Qdrant client."""

    def __init__(self, client, embeddings, *, collection_name="s8_cloud"):
        self.client = client
        self.embeddings = embeddings
        self.collection_name = collection_name
        self.documents = {d.metadata["passage_id"]: d for d in embeddings.documents
                          if d.page_content.strip()}
        self.configuration = {**embeddings.configuration, "method": "qdrant_cosine",
                              "vector_store_format": 1}
        self._ready = False

    def _state(self):
        if not self.client.collection_exists(self.collection_name):
            raise ValueError("Qdrant collection is missing. Run preparation with --build-qdrant.")
        info = self.client.get_collection(self.collection_name)
        state = (info.config.metadata or {}).get("preparation", {})
        vectors = info.config.params.vectors
        if (state.get("configuration") != self.configuration
                or not isinstance(vectors, models.VectorParams)
                or vectors.size != self.configuration["dimensions"]
                or vectors.distance != models.Distance.COSINE):
            raise ValueError("Qdrant corpus or embedding settings differ. Use a separate matching collection.")
        return state

    def _save_state(self, state):
        self.client.update_collection(self.collection_name, metadata={"preparation": state})

    def _point_id(self, passage_id, window):
        return str(uuid5(NAMESPACE_URL,
                        f"{self.embeddings.fingerprint}/{passage_id}/{window}"))

    def prepare(self, *, batch_size=128, progress=print):
        """Resume by passage batch; acknowledge a batch only after its points persist."""
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        started = perf_counter()
        if not self.client.collection_exists(self.collection_name):
            state = {"configuration": self.configuration, "complete": False,
                     "passages_completed": 0, "windows_completed": 0,
                     "preparation_seconds": 0.0}
            self.client.create_collection(
                self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.configuration["dimensions"], distance=models.Distance.COSINE),
                metadata={"preparation": state})
        state = self._state()
        if state["complete"]:
            return self.open()
        self.embeddings.prepare_model()
        docs = list(self.documents.values())
        previous_seconds = state["preparation_seconds"]
        for start in range(state["passages_completed"], len(docs), batch_size):
            batch = docs[start:start + batch_size]
            vectors = self.embeddings.embed_passages([d.page_content for d in batch])
            points = []
            for doc, matrix in zip(batch, vectors):
                spans = token_window_spans(doc.page_content, self.embeddings._tokenizer)
                if matrix.shape != (len(spans), self.configuration["dimensions"]):
                    raise ValueError("Cached vectors do not match the passage windows.")
                if not np.isfinite(matrix).all() or (np.linalg.norm(matrix, axis=1) == 0).any():
                    raise ValueError("Passage vectors must be finite and nonzero.")
                passage_id = doc.metadata["passage_id"]
                for window, ((begin, end), vector) in enumerate(zip(spans, matrix)):
                    points.append(models.PointStruct(
                        id=self._point_id(passage_id, window), vector=vector.tolist(),
                        payload={**doc.metadata, "window": window, "start": begin, "end": end}))
            # Deterministic IDs make replay safe if interruption precedes the checkpoint.
            self.client.upsert(self.collection_name, points=points, wait=True)
            state["passages_completed"] = start + len(batch)
            state["windows_completed"] += len(points)
            state["preparation_seconds"] = previous_seconds + perf_counter() - started
            self._save_state(state)
            if progress is not None:
                progress(f"Qdrant: {state['passages_completed']:,}/{len(docs):,} passages; "
                         f"{state['windows_completed']:,} windows; "
                         f"{state['preparation_seconds']:.0f}s", flush=True)
        self._verify_coverage(state)
        state["complete"] = True
        state["preparation_seconds"] = previous_seconds + perf_counter() - started
        self._save_state(state)
        self._ready = True
        return state

    def _verify_coverage(self, state):
        """Check every persisted window before declaring the full corpus ready."""
        expected = {}
        for passage_id, doc in self.documents.items():
            spans = token_window_spans(doc.page_content, self.embeddings._tokenizer)
            for window, (start, end) in enumerate(spans):
                expected[self._point_id(passage_id, window)] = (passage_id, window, start, end)
        if len(expected) != state["windows_completed"]:
            raise ValueError("Expected and recorded window counts differ.")
        offset = None
        while True:
            points, offset = self.client.scroll(self.collection_name, limit=256, offset=offset,
                                                with_payload=True, with_vectors=True)
            for point in points:
                payload = point.payload
                observed = tuple(payload.get(key) for key in ("passage_id", "window", "start", "end"))
                if expected.pop(str(point.id), None) != observed:
                    raise ValueError("Qdrant contains an unexpected passage window.")
                vector = np.asarray(point.vector)
                if (vector.shape != (self.configuration["dimensions"],)
                        or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0):
                    raise ValueError("Qdrant contains an invalid vector.")
            if offset is None:
                break
        if expected or state["passages_completed"] != len(self.documents):
            raise ValueError("Qdrant collection has incomplete corpus coverage.")

    def open(self):
        """Reject a partial build rather than silently searching a smaller corpus."""
        state = self._state()
        count = self.client.count(self.collection_name, exact=True).count
        if (not state.get("complete") or state["passages_completed"] != len(self.documents)
                or count != state["windows_completed"]):
            raise ValueError("Qdrant collection is incomplete. Resume preparation with --build-qdrant.")
        self._ready = True
        return state

    def search_vector(self, vector, *, k=5):
        """Rank distinct original passages by their highest-scoring window."""
        if type(k) is not int or k < 1:
            raise ValueError("k must be a positive integer.")
        vector = np.asarray(vector)
        if (vector.shape != (self.configuration["dimensions"],)
                or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0):
            raise ValueError("Query vector must have the configured dimensions and be finite and nonzero.")
        if not self._ready:
            self.open()
        found = self.client.query_points_groups(
            self.collection_name, query=vector.tolist(), group_by="passage_id",
            group_size=1, limit=min(k, len(self.documents)), with_payload=True)
        documents = []
        for group in found.groups:
            hit = group.hits[0]
            parent = self.documents[hit.payload["passage_id"]]
            documents.append(Document(page_content=parent.page_content, metadata={
                **parent.metadata, "score": float(hit.score), "window": hit.payload["window"],
                "retriever": "qdrant_cosine"}))
        return documents

    def search(self, query, *, k=5):
        return self.search_vector(self.embeddings.embed_query(query), k=k)
