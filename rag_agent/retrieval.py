"""Document loading, chunking and retrieval (TF-IDF, no external service needed)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class Chunk:
    source: str
    index: int
    text: str


def chunk_text(text: str, source: str, max_chars: int = 600, overlap: int = 120) -> list[Chunk]:
    """Split text into overlapping chunks, cutting on paragraph/sentence boundaries when possible."""
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []
    pieces = [p.strip() for p in re.split(r"\n\s*\n|(?<=[.!?])\s+", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        while len(piece) > max_chars:  # very long sentence: hard split
            head, piece = piece[:max_chars], piece[max_chars - overlap:]
            if current:
                chunks.append(current)
                current = ""
            chunks.append(head)
        if len(current) + len(piece) + 1 <= max_chars:
            current = f"{current} {piece}".strip()
        else:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = f"{tail} {piece}".strip()
    if current:
        chunks.append(current)
    return [Chunk(source=source, index=i, text=c) for i, c in enumerate(chunks)]


def load_documents(folder: Path, max_chars: int = 600, overlap: int = 120) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(folder.rglob("*")):
        if path.suffix.lower() in {".md", ".txt"} and path.is_file():
            chunks.extend(chunk_text(path.read_text(encoding="utf-8"), path.name, max_chars, overlap))
    return chunks


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class Retriever:
    """TF-IDF retriever with word + character n-grams (robust to plurals and typos)."""

    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("no chunks to index")
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(c.text.lower() for c in chunks)

    def search(self, query: str, k: int = 3, min_score: float = 0.05) -> list[Hit]:
        q = self.vectorizer.transform([query.lower()])
        scores = cosine_similarity(q, self.matrix)[0]
        order = scores.argsort()[::-1][:k]
        return [Hit(self.chunks[i], float(scores[i])) for i in order if scores[i] >= min_score]
