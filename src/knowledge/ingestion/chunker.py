"""Text chunking strategies for document ingestion."""

from __future__ import annotations

import re


class TextChunker:
    """Splits text into overlapping chunks using recursive character splitting."""

    SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str) -> list[str]:
        """Split text into overlapping chunks."""
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        text = text.strip()

        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        chunks = self._recursive_split(text, self.SEPARATORS)

        return self._merge_with_overlap(chunks)

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using different separators."""
        if not separators:
            return [text] if text else []

        separator = separators[0]
        remaining_separators = separators[1:]

        if not separator:
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        parts = text.split(separator)
        result: list[str] = []
        current = ""

        for part in parts:
            candidate = f"{current}{separator}{part}" if current else part

            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    result.append(current)
                if len(part) > self.chunk_size:
                    result.extend(self._recursive_split(part, remaining_separators))
                    current = ""
                else:
                    current = part

        if current:
            result.append(current)

        return result

    def _merge_with_overlap(self, chunks: list[str]) -> list[str]:
        """Merge chunks ensuring overlap between consecutive ones."""
        if len(chunks) <= 1:
            return chunks

        result: list[str] = []

        for i, chunk in enumerate(chunks):
            if i == 0:
                result.append(chunk.strip())
                continue

            prev = chunks[i - 1]
            overlap = prev[-self.chunk_overlap :] if len(prev) > self.chunk_overlap else prev

            merged = f"{overlap} {chunk}".strip()

            if len(merged) > self.chunk_size + self.chunk_overlap:
                merged = merged[: self.chunk_size + self.chunk_overlap]

            result.append(merged)

        return result
