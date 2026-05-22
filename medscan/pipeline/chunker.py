"""
Medical-aware document chunker.

Chunks medical text intelligently by:
- Respecting section boundaries (CHIEF COMPLAINT, HPI, LABS, ASSESSMENT, etc.)
- Keeping lab value groups together
- Preserving table rows intact
- Overlapping chunks for better retrieval
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Common section headers in medical documents
MEDICAL_SECTION_HEADERS = re.compile(
    r"^(CHIEF\s+COMPLAINT|HISTORY\s+OF\s+PRESENT\s+ILLNESS|HPI|"
    r"PAST\s+MEDICAL\s+HISTORY|PMH|MEDICATIONS|ALLERGIES|"
    r"REVIEW\s+OF\s+SYSTEMS|ROS|PHYSICAL\s+EXAMINATION|VITALS?|"
    r"LABORATORY|LABS?|IMAGING|RADIOLOGY|ASSESSMENT|PLAN|"
    r"IMPRESSION|DIAGNOSIS|DIAGNOS[EI]S|DISCHARGE\s+SUMMARY|"
    r"DISCHARGE\s+INSTRUCTIONS|FOLLOW[-\s]UP|SIGNATURE)\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class Chunk:
    """A single text chunk ready for embedding."""
    content: str
    chunk_type: str = "text"  # text | table | image_description | structured
    page_number: int | None = None
    chunk_index: int = 0
    section: str | None = None  # medical section header
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def word_count(self) -> int:
        return len(self.content.split())

    @property
    def char_count(self) -> int:
        return len(self.content)


class MedicalAwareChunker:
    """
    Chunks medical documents with awareness of clinical structure.

    Strategy:
    1. Split by medical section headers first
    2. If a section is too long, split by sentence with overlap
    3. Keep tables and structured data as single chunks
    4. Prepend section context to each chunk for better retrieval
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_size: int = 10,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk_text(
        self,
        text: str,
        page_number: int | None = None,
        start_index: int = 0,
    ) -> list[Chunk]:
        """
        Chunk plain text from a document page.

        Args:
            text: Raw text to chunk.
            page_number: Source page number.
            start_index: Starting chunk index offset.

        Returns:
            List of Chunk objects.
        """
        if not text.strip():
            return []

        # Try to split by medical sections first
        sections = self._split_by_sections(text)

        chunks: list[Chunk] = []
        idx = start_index

        for section_name, section_text in sections:
            if len(section_text) <= self.chunk_size:
                # Small enough to be a single chunk
                content = (
                    f"[{section_name}]\n{section_text}"
                    if section_name
                    else section_text
                )
                chunks.append(Chunk(
                    content=content.strip(),
                    chunk_type="text",
                    page_number=page_number,
                    chunk_index=idx,
                    section=section_name,
                ))
                idx += 1
            else:
                # Split large sections by sentences with overlap
                sub_chunks = self._sliding_window_split(
                    section_text,
                    section_name=section_name,
                    page_number=page_number,
                    start_index=idx,
                )
                chunks.extend(sub_chunks)
                idx += len(sub_chunks)

        return [c for c in chunks if c.char_count >= self.min_chunk_size]

    def chunk_table(
        self,
        table_data: dict[str, Any],
        page_number: int | None = None,
        chunk_index: int = 0,
    ) -> Chunk:
        """Represent a table as a single structured chunk."""
        headers = table_data.get("headers", [])
        rows = table_data.get("rows", [])

        # Format table as pipe-delimited text for embedding
        lines = [" | ".join(str(h) for h in headers)]
        lines.append("-" * 40)
        for row in rows:
            lines.append(" | ".join(str(cell) for cell in row))

        content = "\n".join(lines)

        return Chunk(
            content=content,
            chunk_type="table",
            page_number=page_number,
            chunk_index=chunk_index,
            metadata={"headers": headers, "row_count": len(rows)},
        )

    def chunk_image_description(
        self,
        description: str,
        image_type: str = "image",
        page_number: int | None = None,
        chunk_index: int = 0,
    ) -> Chunk:
        """Wrap an image description as a chunk."""
        content = f"[{image_type.upper()} DESCRIPTION]\n{description}"
        return Chunk(
            content=content.strip(),
            chunk_type="image_description",
            page_number=page_number,
            chunk_index=chunk_index,
            metadata={"image_type": image_type},
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_by_sections(self, text: str) -> list[tuple[str | None, str]]:
        """Split text by medical section headers."""
        lines = text.split("\n")
        sections: list[tuple[str | None, str]] = []
        current_header: str | None = None
        current_lines: list[str] = []

        for line in lines:
            if MEDICAL_SECTION_HEADERS.match(line.strip()):
                # Save current section
                if current_lines:
                    sections.append((current_header, "\n".join(current_lines).strip()))
                current_header = line.strip().rstrip(":")
                current_lines = []
            else:
                current_lines.append(line)

        # Save last section
        if current_lines:
            sections.append((current_header, "\n".join(current_lines).strip()))

        if not sections:
            sections = [(None, text)]

        return sections

    def _sliding_window_split(
        self,
        text: str,
        section_name: str | None = None,
        page_number: int | None = None,
        start_index: int = 0,
    ) -> list[Chunk]:
        """Split text into overlapping chunks by character count."""
        words = text.split()
        chunks = []
        idx = start_index

        step = self.chunk_size - self.chunk_overlap
        pos = 0

        while pos < len(words):
            window = words[pos : pos + self.chunk_size]
            content = " ".join(window)
            if section_name:
                content = f"[{section_name}]\n{content}"

            chunks.append(Chunk(
                content=content.strip(),
                chunk_type="text",
                page_number=page_number,
                chunk_index=idx,
                section=section_name,
            ))
            idx += 1
            pos += step

        return chunks
