"""Small documents and deterministic embeddings; no downloads or provider calls."""
from unittest.mock import MagicMock

import numpy as np
from langchain_core.documents import Document
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace


def documents():
    return [Document(page_content=text, metadata={"passage_id": name})
            for name, text in [("a", "red cloud"), ("b", "blue sky"), ("c", "green field")]]


def fake_embedder(index):
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    index._tokenizer = tokenizer
    index._embedder = MagicMock()
    def embed(texts, **kwargs):
        vectors = []
        for text in texts:
            vector = np.zeros(384, dtype=np.float32)
            vector[1 if "blue" in text else 0] = 1.0
            vectors.append(vector)
        return iter(vectors)
    index._embedder.embed.side_effect = embed
