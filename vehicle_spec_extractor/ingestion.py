from __future__ import annotations

from pathlib import Path
from typing import Any


def load_pdf_pages(pdf_path: str | Path) -> list[dict[str, Any]]:
    path = Path(pdf_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Manual PDF not found: {path}")
    if not path.is_file():
        raise ValueError(f"Manual path is not a file: {path}")
    if path.suffix.casefold() != ".pdf":
        raise ValueError(f"Expected a PDF manual, got: {path.name}")

    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise RuntimeError(
            "PDF support is unavailable. Install this project's requirements."
        ) from error

    reader = PdfReader(str(path))
    if reader.is_encrypted:
        raise ValueError(f"The PDF is encrypted and cannot be read: {path.name}")

    pages: list[dict[str, Any]] = []
    for page_index, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        pages.append(
            {
                "text": text,
                "metadata": {
                    "page_number": page_index + 1,
                    "pdf_page_index": page_index,
                    "source_file": path.name,
                    "file_name": path.name,
                    "file_path": str(path),
                    "file_type": "pdf",
                },
            }
        )
    return pages
