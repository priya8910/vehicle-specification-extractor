from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
DATA_DIRECTORY = PROJECT_ROOT / "data"
DEFAULT_MANUAL_PATH = DATA_DIRECTORY / "sample-service-manual 1.pdf"


def project_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


DEFAULT_PERSIST_DIRECTORY = project_path(
    os.getenv("VEHICLE_SPEC_CHROMA_DIR", "artifacts/chroma_db")
)
DEFAULT_EMBEDDING_MODEL = os.getenv(
    "VEHICLE_SPEC_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
DEFAULT_LLM_MODEL = os.getenv("GROQ_FAST_MODEL", "openai/gpt-oss-20b")
DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 120
EVALUATION_CASES_PATH = (
    PROJECT_ROOT / "evaluation" / "fixtures" / "evaluation_cases.json"
)
