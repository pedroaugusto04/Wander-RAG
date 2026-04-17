"""Markdown cleaner — normalises and cleans Markdown extracted from documents.

Removes parsing artefacts (page numbers, repeated headers/footers),
fixes encoding issues, and normalises heading hierarchy so that
downstream chunking receives clean, well-structured Markdown.
"""

from __future__ import annotations

import re
import unicodedata


class MarkdownCleaner:
    """Pipeline of sequential rules that clean raw Markdown."""

    # Common CEFET-MG header/footer patterns.
    _INSTITUTIONAL_NOISE = re.compile(
        r"^.*(?:CEFET[\s-]*MG|Centro Federal|campus\s+Timóteo).*$",
        re.IGNORECASE | re.MULTILINE,
    )

    # Standalone page numbers: a line that is only digits (possibly surrounded
    # by whitespace or a few separator characters like "- 3 -").
    _PAGE_NUMBER = re.compile(r"^\s*[-–—]*\s*\d{1,4}\s*[-–—]*\s*$", re.MULTILINE)

    # Broken Markdown image/link references that carry no useful text.
    _BROKEN_LINK = re.compile(r"!?\[([^\]]*)\]\((?:https?://\S+|data:\S+)\)")

    # Three or more consecutive blank lines → collapse to two.
    _EXCESSIVE_BLANKS = re.compile(r"\n{3,}")

    # Trailing whitespace per line.
    _TRAILING_WS = re.compile(r"[ \t]+$", re.MULTILINE)

    # Horizontal rules used as page separators by LlamaParse.
    _PAGE_SEPARATOR = re.compile(r"^-{3,}$", re.MULTILINE)

    def clean(self, markdown: str, *, source_filename: str = "") -> str:
        """Run the full cleaning pipeline and return normalised Markdown."""
        text = markdown
        text = self._fix_encoding(text)
        if not source_filename.lower().endswith(".md"):
            text = self._remove_page_artifacts(text)
        text = self._normalize_headings(text)
        text = self._clean_broken_links(text)
        text = self._normalize_whitespace(text)
        return text.strip()

    # ------------------------------------------------------------------
    # Individual cleaning steps
    # ------------------------------------------------------------------

    @staticmethod
    def _fix_encoding(text: str) -> str:
        """Normalise Unicode and fix common mojibake from PDF extraction."""
        # NFC normalisation merges combining marks (e.g. a + ◌́  → á).
        text = unicodedata.normalize("NFC", text)

        # Common double-encoding artefacts (latin-1 bytes re-decoded as UTF-8).
        replacements: dict[str, str] = {
            "\xc3\xa1": "á", "\xc3\xa0": "à", "\xc3\xa2": "â", "\xc3\xa3": "ã",
            "\xc3\xa9": "é", "\xc3\xaa": "ê",
            "\xc3\xb3": "ó", "\xc3\xb4": "ô", "\xc3\xb5": "õ",
            "\xc3\xba": "ú", "\xc3\xbc": "ü",
            "\xc3\xa7": "ç", "\xc3\x81": "Á",
            "\xe2\x80\x99": "\u2019",  # right single quote
            "\xe2\x80\x9c": "\u201c",  # left double quote
            "\xe2\x80\x9d": "\u201d",  # right double quote
            "\xe2\x80\x94": "\u2014",  # em dash
            "\xe2\x80\x93": "\u2013",  # en dash
            "\u00ad": "",   # soft hyphen
            "\ufeff": "",   # BOM
        }
        for bad, good in replacements.items():
            text = text.replace(bad, good)

        return text

    def _remove_page_artifacts(self, text: str) -> str:
        """Strip page numbers, repeated institutional headers/footers."""
        text = self._PAGE_NUMBER.sub("", text)
        text = self._PAGE_SEPARATOR.sub("", text)

        # Remove institutional noise lines, but only if they appear multiple
        # times (a single mention is likely real content, not a header).
        # We also limit it to short lines to prevent deleting entire paragraphs
        # that mention CEFET-MG.
        noise_matches = self._INSTITUTIONAL_NOISE.findall(text)
        for match in set(noise_matches):
            if len(match) < 150 and text.count(match) > 2:
                text = text.replace(match, "")

        return text

    @staticmethod
    def _normalize_headings(text: str) -> str:
        """Ensure heading hierarchy starts at ``#`` and never skips levels.

        For example ``### Title`` followed by ``##### Sub`` becomes
        ``# Title`` / ``### Sub`` (shifted so the top-most heading is ``#``).
        """
        heading_re = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
        matches = list(heading_re.finditer(text))
        if not matches:
            return text

        min_level = min(len(m.group(1)) for m in matches)
        if min_level <= 1:
            return text  # already starts at #

        shift = min_level - 1

        def _shift(m: re.Match[str]) -> str:
            new_level = max(1, len(m.group(1)) - shift)
            return f"{'#' * new_level} {m.group(2)}"

        return heading_re.sub(_shift, text)

    def _clean_broken_links(self, text: str) -> str:
        """Replace broken image/link references with their alt text."""
        return self._BROKEN_LINK.sub(r"\1", text)

    def _normalize_whitespace(self, text: str) -> str:
        """Collapse excessive blank lines and strip trailing whitespace."""
        text = self._TRAILING_WS.sub("", text)
        text = self._EXCESSIVE_BLANKS.sub("\n\n", text)
        return text
