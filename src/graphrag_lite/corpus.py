"""Corpus ingestion: split raw documents into fixed-size Passage objects."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Passage:
    """A text chunk with a stable id and optional metadata."""

    id: str
    text: str
    metadata: dict = field(default_factory=dict)


def split_corpus(
    docs: list[str],
    passage_size: int = 200,
    overlap: int = 0,
) -> list[Passage]:
    """Split a list of document strings into word-windowed Passage objects.

    Args:
        docs: Raw document strings.
        passage_size: Target word count per passage.
        overlap: Number of words to overlap between adjacent passages (default 0).

    Returns:
        List of Passage objects with ids ``P0``, ``P1``, â€¦
    """
    passages: list[Passage] = []
    pid = 0
    for doc_idx, doc in enumerate(docs):
        words = doc.split()
        if not words:
            continue
        step = max(1, passage_size - overlap)
        for start in range(0, len(words), step):
            chunk = words[start : start + passage_size]
            if not chunk:
                continue
            passages.append(
                Passage(
                    id=f"P{pid}",
                    text=" ".join(chunk),
                    metadata={"doc_idx": doc_idx, "word_start": start},
                )
            )
            pid += 1
    return passages
