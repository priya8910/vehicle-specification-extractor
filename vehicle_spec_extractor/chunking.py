from __future__ import annotations

import re
from typing import Any

_FORMAL_SECTION = re.compile(
    r"^SECTION\s+\d{3}-\d{2}[A-Z]?\s*:\s*.+$",
    re.IGNORECASE,
)
_KNOWN_HEADING = re.compile(
    r"^(?:"
    r"Suspension System|Inspection and Verification|Visual Inspection Chart|"
    r"Symptom Chart(?:\s*[—-].*)?|Mechanical|Diagnosis and Testing|"
    r"Component Tests|Ball Joint Inspection|Torque Specifications|"
    r"Specifications|Removal and Installation|Description and Operation|"
    r"General Procedures|Special Tools?|"
    r"A\d+\s+(?:ISOLATE|CHECK|VERIFY|INSPECT|ADJUST)\b.*"
    r")$",
    re.IGNORECASE,
)


def _clean_heading(line: str) -> str:
    return line.strip()


def _section_chunks(
    text: str,
    *,
    formal_section: str,
) -> list[tuple[str, str]]:
    sections: list[tuple[str, list[str]]] = []
    current_heading = ""
    current_lines: list[str] = []

    def flush() -> None:
        if current_lines:
            sections.append((current_heading, current_lines.copy()))
            current_lines.clear()

    for line in text.splitlines():
        normalized = _clean_heading(line)
        if _FORMAL_SECTION.match(normalized):
            formal_section = normalized
            continue
        if _KNOWN_HEADING.match(normalized):
            flush()
            current_heading = normalized
        if line.strip():
            current_lines.append(line.strip())

    flush()
    if not sections and text.strip():
        sections.append(("", [text.strip()]))

    result: list[tuple[str, str]] = []
    for heading, lines in sections:
        section_name = heading or formal_section or "Unlabeled section"
        if formal_section and heading and formal_section.casefold() not in heading.casefold():
            section_name = f"{formal_section} / {heading}"
        result.append(("\n".join(lines), section_name))
    return result


def chunk_manual_pages(
    pages: list[dict[str, Any]],
    *,
    chunk_size: int = 900,
    chunk_overlap: int = 120,
) -> list[dict[str, Any]]:
    """Split each PDF page by headings first, retaining page/section citation data."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be between zero and chunk_size - 1.")

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    output: list[dict[str, Any]] = []

    for page in pages:
        text = str(page.get("text", "")).strip()
        if not text:
            continue
        page_metadata = dict(page.get("metadata", {}))
        page_number = page_metadata.get("page_number")
        formal_section = ""
        for line in text.splitlines():
            normalized = _clean_heading(line)
            if _FORMAL_SECTION.match(normalized):
                formal_section = normalized
                break

        page_chunk_index = 0
        for section_text, section_name in _section_chunks(
            text,
            formal_section=formal_section,
        ):
            for chunk_text in splitter.split_text(section_text):
                metadata = page_metadata.copy()
                metadata.update(
                    {
                        "page_number": page_number,
                        "section": section_name,
                        "chunk_index": page_chunk_index,
                    }
                )
                output.append({"text": chunk_text, "metadata": metadata})
                page_chunk_index += 1
    return output
