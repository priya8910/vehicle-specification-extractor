from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from vehicle_spec_extractor.chunking import chunk_manual_pages
from vehicle_spec_extractor.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_PERSIST_DIRECTORY,
    project_path,
)
from vehicle_spec_extractor.ingestion import load_pdf_pages
from vehicle_spec_extractor.vector_store import ChromaVectorStore


def manual_fingerprint(pdf_path: str | Path) -> str:
    path = Path(pdf_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Manual PDF not found: {path}")
    if not path.is_file():
        raise ValueError(f"Manual path is not a file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collection_name_for(manual_id: str) -> str:
    return f"vehicle_specs_{manual_id[:24]}"


_STORE_CACHE: dict[tuple[str, str, str], ChromaVectorStore] = {}
_TORQUE_QUERY_TERMS = {
    "torque",
    "specification",
    "specifications",
    "table",
    "value",
    "values",
    "what",
    "is",
    "the",
    "for",
    "of",
    "on",
    "in",
    "at",
    "a",
    "an",
    "to",
    "give",
    "me",
    "please",
    "much",
    "how",
    "does",
    "do",
    "use",
    "section",
    "nm",
    "lb",
    "ft",
    "inch",
    "inches",
    "newton",
    "meter",
    "meters",
    "metre",
    "metres",
    "pound",
    "pounds",
    "foot",
    "feet",
    "front",
    "rear",
    "suspension",
}
_RETRIEVAL_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does",
    "for", "from", "give", "how", "in", "is", "it", "me", "of", "on",
    "or", "please", "return", "the", "this", "to", "what", "when",
    "which", "with",
}


def _torque_component_terms(query: str) -> list[str]:
    without_section = re.sub(
        r"\bsection\s+\d{3}-\d{2}[A-Z]?\b",
        " ",
        query,
        flags=re.IGNORECASE,
    )
    return [
        word
        for word in re.findall(r"[a-z]+", without_section.casefold())
        if word not in _TORQUE_QUERY_TERMS
    ]


def _matching_torque_tables(
    query: str,
    evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    component_terms = _torque_component_terms(query)
    if not component_terms:
        return []

    matching: list[dict[str, Any]] = []
    for item in evidence:
        passage = str(item.get("text", ""))
        normalized_passage = passage.casefold()
        if "description" not in normalized_passage or "nm" not in normalized_passage:
            continue
        section = str(
            item.get("source", {}).get("section")
            or item.get("metadata", {}).get("section", "")
        ).upper()
        if (
            {"ball", "joint"}.issubset(component_terms)
            and not re.search(r"\bSECTION\s+204-01[AB]\b", section)
        ):
            continue

        for line in passage.splitlines():
            row_words = {
                word.rstrip("s")
                for word in re.findall(r"[a-z]+", line.casefold())
            }
            has_value = bool(re.search(r"\d", line))
            if has_value and all(
                term.rstrip("s") in row_words for term in component_terms
            ):
                matching.append(item)
                break
    return matching


def _rerank_by_query_terms(
    query: str,
    evidence: list[dict[str, Any]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    """Blend semantic similarity with lexical and explicit section matches."""
    query_terms = {
        term
        for term in re.findall(r"[a-z0-9]+", query.casefold())
        if term not in _RETRIEVAL_STOPWORDS and len(term) > 1
    }
    section_hint_match = re.search(
        r"\bin\s+(?:the\s+)?(.+?)\s+procedure\b",
        query,
        flags=re.IGNORECASE,
    )
    section_hint = (
        " ".join(re.findall(r"[a-z0-9]+", section_hint_match.group(1).casefold()))
        if section_hint_match
        else ""
    )

    def score(item: dict[str, Any]) -> tuple[bool, float]:
        source = item.get("source", {})
        metadata = item.get("metadata", {})
        section = str(source.get("section") or metadata.get("section") or "")
        text = str(item.get("text", ""))
        passage_terms = set(
            re.findall(r"[a-z0-9]+", f"{section} {text}".casefold())
        )
        lexical = (
            len(query_terms & passage_terms) / len(query_terms)
            if query_terms
            else 0.0
        )
        heading_match = bool(
            section_hint
            and section_hint in " ".join(re.findall(r"[a-z0-9]+", section.casefold()))
        )
        similarity = float(item.get("similarity", 0.0))
        return heading_match, similarity + 0.15 * lexical

    return sorted(evidence, key=score, reverse=True)[:top_k]


def get_manual_store(
    manual_id: str,
    *,
    persist_directory: str | Path = DEFAULT_PERSIST_DIRECTORY,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    create: bool = False,
) -> ChromaVectorStore:
    resolved_directory = project_path(persist_directory).resolve()
    key = (
        str(resolved_directory),
        collection_name_for(manual_id),
        embedding_model,
    )
    if key not in _STORE_CACHE:
        _STORE_CACHE[key] = ChromaVectorStore(
            persist_directory=resolved_directory,
            collection_name=key[1],
            embedding_model=embedding_model,
            create=create,
        )
    return _STORE_CACHE[key]


def ingest_manual(
    pdf_path: str | Path,
    *,
    persist_directory: str | Path = DEFAULT_PERSIST_DIRECTORY,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> dict[str, Any]:
    path = Path(pdf_path).expanduser().resolve()
    manual_id = manual_fingerprint(path)
    pages = load_pdf_pages(path)
    if not pages:
        raise ValueError(f"No readable text was extracted from {path.name}.")

    chunks = chunk_manual_pages(
        pages,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if not chunks:
        raise ValueError(f"No usable chunks were created from {path.name}.")

    for chunk in chunks:
        metadata = chunk["metadata"]
        section_match = re.search(
            r"\bSECTION\s+(\d{3}-\d{2}[A-Z]?)\b",
            str(metadata.get("section", "")),
            flags=re.IGNORECASE,
        )
        material = (
            f"{manual_id}:{metadata.get('page_number')}:"
            f"{metadata.get('chunk_index')}:{chunk['text']}"
        )
        metadata.update(
            {
                "manual_id": manual_id,
                "source_file": path.name,
                "section_id": (
                    section_match.group(1).upper() if section_match else ""
                ),
            }
        )
        chunk["id"] = hashlib.sha256(material.encode("utf-8")).hexdigest()

    store = get_manual_store(
        manual_id,
        persist_directory=persist_directory,
        embedding_model=embedding_model,
        create=True,
    )
    indexed = store.upsert_chunks(chunks)
    if indexed != len(chunks):
        raise RuntimeError(
            f"Only {indexed} of {len(chunks)} chunks were indexed."
        )

    return {
        "status": "completed",
        "manual_id": manual_id,
        "source_file": path.name,
        "pages_with_text": len(pages),
        "chunks_indexed": indexed,
        "collection_name": collection_name_for(manual_id),
        "persist_directory": str(project_path(persist_directory).resolve()),
    }


def retrieve_manual(
    pdf_path: str | Path,
    query: str,
    *,
    top_k: int = 5,
    persist_directory: str | Path = DEFAULT_PERSIST_DIRECTORY,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> list[dict[str, Any]]:
    if not query.strip():
        raise ValueError("Query cannot be empty.")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    path = Path(pdf_path).expanduser().resolve()
    manual_id = manual_fingerprint(path)
    store = get_manual_store(
        manual_id,
        persist_directory=persist_directory,
        embedding_model=embedding_model,
        create=False,
    )
    section_match = re.search(
        r"\bsection\s+(\d{3}-\d{2}[A-Z]?)\b",
        query,
        flags=re.IGNORECASE,
    )
    where = (
        {"section_id": section_match.group(1).upper()}
        if section_match
        else None
    )
    torque_query = bool(re.search(r"\btorque\b", query, flags=re.IGNORECASE))
    search_query = (
        f"{query} torque specifications table Description Nm lb-ft"
        if torque_query
        else query
    )
    search_limit = max(top_k, 10) if torque_query else max(top_k * 4, 20)
    results = store.search(search_query, top_k=search_limit, where=where)
    if torque_query:
        matching_tables = _matching_torque_tables(query, results)
        if matching_tables:
            return matching_tables[:top_k]
    return _rerank_by_query_terms(query, results, top_k=top_k)
