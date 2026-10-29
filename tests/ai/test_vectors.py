"""Small real local collections; embeddings are synthetic and require no network."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from qdrant_client import QdrantClient

from ai_lib.retrieval import RetrievalIndex
from ai_lib.vector_store import QdrantIndex
from tests.ai.fixtures import documents, fake_embedder


class VectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        docs = documents()
        docs[0].page_content = 'red ' * 800
        self.embeddings = RetrievalIndex(docs)
        self.embeddings.build_lexical(self.folder / 'embeddings')
        fake_embedder(self.embeddings)
        self.client = QdrantClient(path=str(self.folder / 'qdrant'))
        self.addCleanup(lambda: self.client.close())
        self.index = QdrantIndex(self.client, self.embeddings)

    def test_persistence(self):
        state = self.index.prepare(progress=None)
        self.assertEqual(state['passages_completed'], 3)
        self.assertGreater(state['windows_completed'], 3)
        self.client.close()
        self.client = QdrantClient(path=str(self.folder / 'qdrant'))
        restored = QdrantIndex(self.client, self.embeddings)
        self.assertEqual(restored.open(), state)
        self.assertEqual(restored.search('blue', k=1)[0].metadata['passage_id'], 'b')

    def test_unique_parents(self):
        self.index.prepare(progress=None)
        result = self.index.search('red', k=3)
        self.assertEqual(len(result), 3)
        self.assertEqual(len({doc.metadata['passage_id'] for doc in result}), 3)
        original = {doc.metadata['passage_id']: doc.page_content for doc in self.embeddings.documents}
        for doc in result:
            self.assertEqual(doc.page_content, original[doc.metadata['passage_id']])

    def test_resume(self):
        original = self.index._save_state
        def interrupted(state):
            if state['passages_completed'] == 2:
                raise RuntimeError('interrupted after insertion')
            original(state)
        with patch.object(self.index, '_save_state', side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                self.index.prepare(batch_size=1, progress=None)
        self.client.close()
        self.client = QdrantClient(path=str(self.folder / 'qdrant'))
        self.index = QdrantIndex(self.client, self.embeddings)
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            self.index.open()
        state = QdrantIndex(self.client, self.embeddings).prepare(batch_size=1, progress=None)
        self.assertTrue(state['complete'])
        self.assertEqual(self.client.count('s8_cloud').count, state['windows_completed'])
        before = self.embeddings._embedder.embed.call_count
        self.index.prepare(progress=None)
        self.assertEqual(before, self.embeddings._embedder.embed.call_count)

    def test_configuration(self):
        self.index.prepare(progress=None)
        for key, value in [('corpus_sha256', 'other'), ('model', 'other'),
                           ('query_prefix', 'other'), ('window_tokens', 10), ('dimensions', 5)]:
            changed = QdrantIndex(self.client, self.embeddings)
            changed.configuration[key] = value
            with self.assertRaisesRegex(ValueError, 'settings differ'):
                changed.open()

    def test_missing_collection(self):
        with self.assertRaisesRegex(ValueError, 'missing'):
            self.index.search('red')

    def test_invalid_query_vector(self):
        self.index.prepare(progress=None)
        for vector in [np.zeros(384), np.ones(2), np.full(384, np.nan)]:
            with self.assertRaisesRegex(ValueError, 'Query vector'):
                self.index.search_vector(vector)

    def test_window_coverage(self):
        self.index.prepare(progress=None)
        state = self.index._state()
        state['complete'] = False
        self.index._save_state(state)
        self.client.delete('s8_cloud', points_selector=[self.index._point_id('a', 0)])
        with self.assertRaisesRegex(ValueError, 'incomplete corpus'):
            self.index.prepare(progress=None)

    def test_cached_vectors(self):
        self.embeddings.embed_passages([d.page_content for d in self.embeddings.documents])
        calls = self.embeddings._embedder.embed.call_count
        self.index.prepare(progress=None)
        self.assertEqual(calls, self.embeddings._embedder.embed.call_count)
