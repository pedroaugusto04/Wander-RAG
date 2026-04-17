"""Markdown-aware text chunker for document ingestion.

Splits Markdown by heading hierarchy, preserves section breadcrumbs in
each chunk's metadata, and treats tables as atomic units.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Public data model
# ---------------------------------------------------------------------------

@dataclass
class ChunkWithContext:
    """A text chunk enriched with structural context from the source document."""

    content: str
    section_breadcrumb: str = ""
    heading_level: int = 0
    chunk_index: int = 0
    metadata: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Chunker implementation
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_TABLE_RE = re.compile(
    r"((?:^\|.+\|[ \t]*$\n?)+)",
    re.MULTILINE,
)


class MarkdownChunker:
    """Splits Markdown into overlapping chunks, respecting headings and tables.

    Key behaviours:
      * Headings (``#`` … ``######``) act as semantic boundaries.
      * Each chunk carries a *breadcrumb* built from its parent headings
        (e.g. ``"Graduação > Eng. Ambiental > Grade Curricular"``).
      * Tables are never split across chunks.
      * Within a section, text is split using recursive character splitting
        with configurable overlap (similar to LangChain's approach).
    """

    SEPARATORS = ["\n\n", "\n", ". ", " "]

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, markdown: str) -> list[ChunkWithContext]:
        """Split *markdown* into contextualised chunks."""
        markdown = self._normalize(markdown)
        if not markdown:
            return []

        sections = self._split_by_headings(markdown)
        chunks: list[ChunkWithContext] = []
        idx = 0

        for section in sections:
            breadcrumb = section["breadcrumb"]
            level = section["level"]
            body = section["body"].strip()

            if not body:
                continue

            # Prefix each chunk with the breadcrumb for retrieval context.
            prefix = f"[{breadcrumb}]\n\n" if breadcrumb else ""

            sub_chunks = self._split_section(body, reserved_prefix_len=len(prefix))

            for text in sub_chunks:
                full_text = f"{prefix}{text}" if prefix else text
                chunks.append(
                    ChunkWithContext(
                        content=full_text,
                        section_breadcrumb=breadcrumb,
                        heading_level=level,
                        chunk_index=idx,
                    )
                )
                idx += 1

        return chunks

    # ------------------------------------------------------------------
    # Heading-based splitting
    # ------------------------------------------------------------------

    def _split_by_headings(self, text: str) -> list[dict[str, object]]:
        """Split text into sections delimited by Markdown headings.

        Returns a list of dicts with keys: ``level``, ``title``,
        ``breadcrumb``, ``body``.
        """
        matches = list(_HEADING_RE.finditer(text))

        if not matches:
            return [{"level": 0, "title": "", "breadcrumb": "", "body": text}]

        sections: list[dict[str, object]] = []
        # Stack tracks the current heading hierarchy for breadcrumb building.
        # Each entry is (level, title).
        heading_stack: list[tuple[int, str]] = []

        # Text before the first heading (if any).
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append(
                {"level": 0, "title": "", "breadcrumb": "", "body": preamble}
            )

        for i, match in enumerate(matches):
            level = len(match.group(1))
            title = match.group(2).strip()

            # Pop headings from the stack that are at the same or deeper level.
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))

            breadcrumb = " > ".join(t for _, t in heading_stack)

            # Body is everything between this heading and the next one.
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()

            sections.append(
                {
                    "level": level,
                    "title": title,
                    "breadcrumb": breadcrumb,
                    "body": body,
                }
            )

        return sections

    # ------------------------------------------------------------------
    # Recursive character splitting within a section
    # ------------------------------------------------------------------

    def _split_section(
        self,
        text: str,
        reserved_prefix_len: int = 0,
    ) -> list[str]:
        """Split a section body into chunks respecting tables and overlap."""
        effective_size = max(64, self.chunk_size - reserved_prefix_len)

        if len(text) <= effective_size:
            return [text]

        # Protect tables: replace them with placeholders, chunk the rest,
        # then re-inject tables as standalone chunks when needed.
        tables: list[str] = []
        protected = text

        for table_match in _TABLE_RE.finditer(text):
            placeholder = f"\x00TABLE_{len(tables)}\x00"
            tables.append(table_match.group(0))
            protected = protected.replace(table_match.group(0), placeholder, 1)

        raw_chunks = self._recursive_split(protected, self.SEPARATORS, effective_size)

        # Expand table placeholders back.
        result: list[str] = []
        for chunk in raw_chunks:
            if "\x00TABLE_" in chunk:
                # A chunk that contains a table placeholder → emit the table
                # as its own chunk(s) and the surrounding text separately.
                parts = re.split(r"\x00TABLE_(\d+)\x00", chunk)
                for j, part in enumerate(parts):
                    if j % 2 == 0:
                        # Normal text
                        stripped = part.strip()
                        if stripped:
                            result.append(stripped)
                    else:
                        # Table index
                        table_idx = int(part)
                        result.append(tables[table_idx].strip())
            else:
                stripped = chunk.strip()
                if stripped:
                    result.append(stripped)

        return self._merge_with_overlap(result, effective_size)

    def _recursive_split(
        self,
        text: str,
        separators: list[str],
        target_size: int,
    ) -> list[str]:
        """Recursively split *text* using progressively finer separators."""
        if not separators:
            # Last resort: hard character split.
            return [text[i: i + target_size] for i in range(0, len(text), target_size)]

        sep = separators[0]
        remaining = separators[1:]

        parts = text.split(sep)
        result: list[str] = []
        current = ""

        for part in parts:
            candidate = f"{current}{sep}{part}" if current else part

            if len(candidate) <= target_size:
                current = candidate
            else:
                if current:
                    result.append(current)
                if len(part) > target_size:
                    result.extend(self._recursive_split(part, remaining, target_size))
                    current = ""
                else:
                    current = part

        if current:
            result.append(current)

        return result

    def _merge_with_overlap(
        self,
        chunks: list[str],
        target_size: int,
    ) -> list[str]:
        """Add overlap between consecutive chunks from the *same section*."""
        if len(chunks) <= 1:
            return chunks

        result: list[str] = [chunks[0].strip()]

        for i in range(1, len(chunks)):
            previous = chunks[i - 1].strip()
            current = chunks[i].strip()

            # Word-overlap helps prose retrieval, but it mangles structured
            # Markdown blocks such as bullet lists by prepending trailing words
            # from the previous chunk into the next list item.
            if self._should_skip_overlap(previous, current):
                result.append(current)
                continue

            overlap = self._word_safe_overlap(previous)

            merged = f"{overlap} {current}".strip()
            if len(merged) > target_size + self.chunk_overlap:
                merged = self._trim_to_boundary(
                    merged,
                    target_size + self.chunk_overlap,
                )

            result.append(merged)

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(text: str) -> str:
        """Basic whitespace normalisation before chunking."""
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        return text.strip()

    def _word_safe_overlap(self, text: str) -> str:
        """Return overlap text using whole trailing words whenever possible."""
        if len(text) <= self.chunk_overlap:
            return text.strip()

        words = text.split()
        if len(words) <= 1:
            return text[-self.chunk_overlap :].strip()

        selected: list[str] = []
        current_len = 0

        for word in reversed(words):
            selected.append(word)
            current_len += len(word) + (1 if len(selected) > 1 else 0)
            if current_len >= self.chunk_overlap:
                break

        return " ".join(reversed(selected)).strip()

    @staticmethod
    def _trim_to_boundary(text: str, max_len: int) -> str:
        """Trim text without cutting the final word when avoidable."""
        if len(text) <= max_len:
            return text

        trimmed = text[:max_len].rstrip()
        boundary = max(trimmed.rfind(" "), trimmed.rfind("\n"))
        if boundary >= max_len // 2:
            return trimmed[:boundary].rstrip()
        return trimmed

    @staticmethod
    def _should_skip_overlap(previous: str, current: str) -> bool:
        """Return True when overlap would damage structured Markdown."""
        return (
            MarkdownChunker._looks_like_markdown_list(previous)
            or MarkdownChunker._looks_like_markdown_list(current)
            or MarkdownChunker._looks_like_table(previous)
            or MarkdownChunker._looks_like_table(current)
        )

    @staticmethod
    def _looks_like_markdown_list(text: str) -> bool:
        """Detect bullet/ordered list chunks that should keep line boundaries."""
        list_lines = re.findall(r"(?m)^\s*(?:[-*+]|\d+[.)])\s+", text)
        return len(list_lines) >= 2

    @staticmethod
    def _looks_like_table(text: str) -> bool:
        """Detect table chunks that should remain untouched by overlap."""
        return bool(_TABLE_RE.search(text))
