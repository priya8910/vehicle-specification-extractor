from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from vehicle_spec_extractor.ingestion import load_pdf_pages


def test_load_pdf_pages_preserves_one_based_pdf_page_numbers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manual = tmp_path / "manual.pdf"
    manual.write_bytes(b"test pdf placeholder")

    class FakePdfReader:
        is_encrypted = False
        pages = [
            SimpleNamespace(extract_text=lambda: " First page text "),
            SimpleNamespace(extract_text=lambda: ""),
            SimpleNamespace(extract_text=lambda: "Third page text"),
        ]

        def __init__(self, _path: str) -> None:
            pass

    monkeypatch.setitem(
        sys.modules,
        "pypdf",
        SimpleNamespace(PdfReader=FakePdfReader),
    )

    pages = load_pdf_pages(manual)

    assert [item["metadata"]["page_number"] for item in pages] == [1, 3]
    assert [item["metadata"]["pdf_page_index"] for item in pages] == [0, 2]
    assert pages[0]["text"] == "First page text"
    assert pages[0]["metadata"]["source_file"] == "manual.pdf"


def test_load_pdf_pages_rejects_missing_manual(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Manual PDF not found"):
        load_pdf_pages(tmp_path / "missing.pdf")


def test_load_pdf_pages_rejects_non_pdf_file(tmp_path: Path) -> None:
    text_file = tmp_path / "manual.txt"
    text_file.write_text("not a PDF", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected a PDF"):
        load_pdf_pages(text_file)
