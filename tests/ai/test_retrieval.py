"""Synthetic retrieval checks never download models or call provider APIs."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from ai_lib.retrieval.indexing import RetrievalIndex
from ai_lib.retrieval.windows import token_windows
from tests.ai.fixtures import documents, fake_embedder


class RetrievalTests(unittest.TestCase):
    def test_lexical_roundtrip(self):
        with TemporaryDirectory() as temp:
            index = RetrievalIndex(documents())
            index.build_lexical(temp)
            self.assertEqual(index.search("blue")[0].metadata["passage_id"], "b")
            restored = RetrievalIndex(documents())
            restored.build_lexical(temp)
            self.assertEqual(index.search("sky"), restored.search("sky"))
            self.assertIsNone(index._embedder)

    def test_local_cache(self):
        with TemporaryDirectory() as temp:
            index = RetrievalIndex(documents())
            index.build_lexical(temp)
            fake_embedder(index)
            first = index.search("blue", method="local", k=2)
            self.assertEqual(first[0].metadata["passage_id"], "b")
            calls = index._embedder.embed.call_count
            self.assertEqual(first, index.search("blue", method="local", k=2))
            self.assertEqual(calls, index._embedder.embed.call_count)
            restored = RetrievalIndex(documents())
            restored.build_lexical(temp)
            fake_embedder(restored)
            self.assertEqual(first, restored.search("blue", method="local", k=2))
            restored._embedder.embed.assert_not_called()

    def test_candidate_boundary(self):
        with TemporaryDirectory() as temp:
            index = RetrievalIndex(documents())
            index.build_lexical(temp)
            lexical = index.search("red", k=1)[0]
            with patch.object(index, "_vectors", return_value=np.ones((1, 384))) as embed:
                found = index.search("red", method="local", k=1, candidate_k=1)
            self.assertEqual(found[0].metadata["passage_id"], lexical.metadata["passage_id"])
            self.assertEqual(embed.call_count, 2)  # query and one candidate only

    def test_window_aggregation(self):
        with TemporaryDirectory() as temp:
            index = RetrievalIndex(documents())
            index.build_lexical(temp)
            def vectors(text, **kwargs):
                if kwargs.get("query"):
                    return np.asarray([[1., 0.]])
                if text == "red cloud":
                    return np.asarray([[0., 1.], [1., 0.]])
                return np.asarray([[0., 1.]])
            with patch.object(index, "_vectors", side_effect=vectors):
                found = index.search("red", method="local", k=3)
            self.assertEqual(found[0].metadata["passage_id"], "a")
            self.assertEqual(len({d.metadata["passage_id"] for d in found}), 3)

    def test_token_windows(self):
        tokenizer = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        self.assertEqual(token_windows("one two three four five", tokenizer, size=3, overlap=1),
                         ["one two three", "three four five"])

    def test_invalidation(self):
        with TemporaryDirectory() as temp:
            index = RetrievalIndex(documents())
            index.build_lexical(temp)
            changed = documents()
            changed[0].page_content = "changed content"
            with self.assertRaisesRegex(ValueError, "settings or corpus changed"):
                RetrievalIndex(changed).build_lexical(temp)

    def test_duplicate_ids(self):
        values = documents()
        values[1].metadata["passage_id"] = "a"
        with self.assertRaisesRegex(ValueError, "unique"):
            RetrievalIndex(values)

    def test_invalid_search(self):
        with TemporaryDirectory() as temp:
            index = RetrievalIndex(documents())
            index.build_lexical(temp)
            with self.assertRaisesRegex(ValueError, "candidate_k"):
                index.search("red", method="local", k=5, candidate_k=2)
            with self.assertRaisesRegex(ValueError, "nonempty query"):
                index.search("")

    def test_corrupt_vectors(self):
        with TemporaryDirectory() as temp:
            index = RetrievalIndex(documents())
            index.build_lexical(temp)
            fake_embedder(index)
            index.embed_query("blue")
            path = next((Path(temp)/"queries").glob("*.npy"))
            for vectors in [np.zeros((1,384)), np.ones((1,384)), np.zeros((0,384)), np.ones((2,384))/np.sqrt(384)]:
                np.save(path, vectors)
                with self.assertRaisesRegex(ValueError,"cached local embeddings"):
                    index.embed_query("blue")

    def test_unsupported_embedding_model(self):
        with self.assertRaisesRegex(ValueError,"supports"):
            RetrievalIndex(documents(),model_name="other-model")
