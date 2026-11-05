"""Tokenization and passage subwindows."""
import re


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def token_windows(text, tokenizer, *, size=384, overlap=64):
    """Keep long-passage windows within the local model's input limit."""
    return [text[start:end] for start, end in token_window_spans(
        text, tokenizer, size=size, overlap=overlap)]


def token_window_spans(text, tokenizer, *, size=384, overlap=64):
    """Return character offsets into the original passage, including overlap."""
    if size <= overlap or overlap < 0:
        raise ValueError("Window size must exceed its nonnegative overlap.")
    offsets = tokenizer.encode(text, add_special_tokens=False).offsets
    if len(offsets) <= size:
        return [(0, len(text))]
    windows = []
    for start in range(0, len(offsets), size - overlap):
        end = min(start + size, len(offsets))
        windows.append((offsets[start][0], offsets[end - 1][1]))
        if end == len(offsets):
            break
    return windows
