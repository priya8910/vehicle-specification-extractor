import csv
import io
import json
import os
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

from vehicle_spec_extractor.chunking import chunk_manual_pages
from vehicle_spec_extractor.extraction import (
    _call_groq,
    extract_vehicle_specs,
    export_csv,
    export_json,
)
from vehicle_spec_extractor.config import DEFAULT_LLM_MODEL
from vehicle_spec_extractor import pipeline
from vehicle_spec_extractor.evaluate import _case_metrics
from vehicle_spec_extractor.schemas import CandidateResponse


def _evidence(
    evidence_id: str,
    text: str,
    page: int,
    section: str,
) -> dict:
    return {
        "id": evidence_id,
        "text": text,
        "metadata": {
            "page_number": page,
            "section": section,
            "source_file": "sample-service-manual.pdf",
        },
        "similarity": 0.9,
    }


def _mock_llm(payload: dict):
    def respond(system_prompt: str, user_prompt: str) -> str:
        assert "Never use outside knowledge" in system_prompt
        assert "Evidence passages" in user_prompt
        return json.dumps(payload)

    return respond


def test_groq_call_uses_configured_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict = {}

    def create_completion(**kwargs):
        observed.update(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"results": []}')
                )
            ]
        )

    class FakeGroq:
        def __init__(self, *, api_key: str) -> None:
            assert api_key == "test-key"
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=create_completion)
            )

    import groq

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(groq, "Groq", FakeGroq)

    response = _call_groq("system prompt", "user prompt")

    assert response == '{"results": []}'
    assert observed["model"] == DEFAULT_LLM_MODEL
    assert observed["response_format"]["type"] == "json_schema"
    assert observed["response_format"]["json_schema"]["strict"] is True


def test_chunks_preserve_page_and_split_at_manual_headings() -> None:
    pages = [
        {
            "text": (
                "SECTION 204-00: Suspension System — General Information\n"
                "Inspection and Verification\n"
                "Check the tires and inspect the steering components.\n"
                "Ball Joint Inspection\n"
                "The suspension must be in full rebound for measurement."
            ),
            "metadata": {"page_number": 8, "file_name": "manual.pdf"},
        }
    ]

    chunks = chunk_manual_pages(pages, chunk_size=300, chunk_overlap=0)

    assert len(chunks) == 2
    assert all(chunk["metadata"]["page_number"] == 8 for chunk in chunks)
    assert "Inspection and Verification" in chunks[0]["metadata"]["section"]
    assert "Ball Joint Inspection" in chunks[1]["metadata"]["section"]
    assert "full rebound" in chunks[1]["text"]


def test_claim_support_handles_pdf_line_wrapping() -> None:
    from vehicle_spec_extractor.extraction import _claim_is_supported

    evidence = (
        "SECTION 204-00 / Ball Joint Inspection\n"
        "The suspension must be in\nfull rebound for measurement."
    )

    assert _claim_is_supported(
        "full rebound",
        None,
        evidence,
        component="suspension",
        spec_type="Measurement position",
    )


def test_found_value_is_returned_with_page_section_and_source_passage() -> None:
    evidence = _evidence(
        "chunk-24",
        "Description Nm lb-ft lb-in\nUpper ball joint nuts 115 85 -",
        24,
        "SECTION 204-01A / Torque Specifications",
    )
    response = {
        "results": [
            {
                "component": "Upper ball joint nut",
                "spec_type": "Torque",
                "value": "115",
                "unit": "Nm",
                "status": "found",
                "evidence_ids": ["chunk-24"],
                "conflicting_values": [],
            }
        ]
    }

    result = extract_vehicle_specs(
        "Upper ball joint nut torque?",
        [evidence],
        llm=_mock_llm(response),
    )

    item = result.results[0]
    assert (item.status, item.value, item.unit) == ("found", "115", "Nm")
    assert item.citations[0].page == 24
    assert item.citations[0].section == "SECTION 204-01A / Torque Specifications"
    assert "Upper ball joint nuts 115 85" in item.citations[0].passage


def test_found_result_keeps_only_citations_that_support_the_claim() -> None:
    evidence = [
        _evidence("supporting", "Upper ball joint nut torque is 115 Nm.", 24, "Specs"),
        _evidence("unrelated", "Lower ball joint nut torque is 120 Nm.", 25, "Specs"),
    ]
    response = {
        "results": [
            {
                "component": "Upper ball joint nut",
                "spec_type": "Torque",
                "value": "115",
                "unit": "Nm",
                "status": "found",
                "evidence_ids": ["supporting", "unrelated"],
                "conflicting_values": [],
            }
        ]
    }

    item = extract_vehicle_specs(
        "What is the upper ball joint nut torque?",
        evidence,
        llm=_mock_llm(response),
    ).results[0]

    assert item.status == "found"
    assert [citation.evidence_id for citation in item.citations] == ["supporting"]


def test_unsupported_llm_value_is_downgraded_to_not_found() -> None:
    evidence = _evidence(
        "chunk-8",
        "The suspension must be in full rebound.",
        8,
        "Ball Joint Inspection",
    )
    response = {
        "results": [
            {
                "component": "Ball joint",
                "spec_type": "Torque",
                "value": "115",
                "unit": "Nm",
                "status": "found",
                "evidence_ids": ["chunk-8"],
                "conflicting_values": [],
            }
        ]
    }

    item = extract_vehicle_specs(
        "What is the torque?",
        [evidence],
        llm=_mock_llm(response),
    ).results[0]

    assert item.status == "not_found"
    assert item.value is None
    assert item.citations == []


def test_numeric_substring_is_not_accepted_as_supported_value() -> None:
    evidence = _evidence("chunk", "The specified torque is 115 Nm.", 24, "Specs")
    response = {
        "results": [
            {
                "component": "specified torque",
                "spec_type": "Torque",
                "value": "15",
                "unit": "Nm",
                "status": "found",
                "evidence_ids": ["chunk"],
                "conflicting_values": [],
            }
        ]
    }

    item = extract_vehicle_specs(
        "What is the torque?",
        [evidence],
        llm=_mock_llm(response),
    ).results[0]

    assert item.status == "not_found"


def test_torque_table_unit_must_match_its_column() -> None:
    evidence = _evidence(
        "chunk-59",
        "Description Nm lb-ft lb-in\nShock absorber lower nuts 90 66 —",
        59,
        "4WD Torque Specifications",
    )
    response = {
        "results": [
            {
                "component": "Shock absorber lower nut",
                "spec_type": "Torque",
                "value": "66",
                "unit": "lb-in",
                "status": "found",
                "evidence_ids": ["chunk-59"],
                "conflicting_values": [],
            }
        ]
    }

    item = extract_vehicle_specs(
        "What is the shock absorber lower nut torque?",
        [evidence],
        llm=_mock_llm(response),
    ).results[0]

    assert item.status == "not_found"
    assert item.value is None
    assert item.citations == []


def test_ambiguous_values_are_reported_only_with_grounded_conflicts() -> None:
    evidence = [
        _evidence("a", "Ball joint nut torque is 115 Nm.", 24, "Specs"),
        _evidence("b", "Ball joint nut torque is 120 Nm.", 25, "Specs"),
    ]
    response = {
        "results": [
            {
                "component": "Ball joint nut",
                "spec_type": "Torque",
                "value": None,
                "unit": None,
                "status": "ambiguous",
                "evidence_ids": [],
                "conflicting_values": [
                    {"value": "115", "unit": "Nm", "evidence_id": "a"},
                    {"value": "120", "unit": "Nm", "evidence_id": "b"},
                ],
            }
        ]
    }

    item = extract_vehicle_specs(
        "What is the ball joint nut torque?",
        evidence,
        llm=_mock_llm(response),
    ).results[0]

    assert item.status == "ambiguous"
    assert [claim.value for claim in item.conflicting_values] == ["115", "120"]
    assert [citation.page for citation in item.citations] == [24, 25]


def test_unknown_citation_id_fails_loudly() -> None:
    response = {
        "results": [
            {
                "component": "Ball joint",
                "spec_type": "Torque",
                "value": "115",
                "unit": "Nm",
                "status": "found",
                "evidence_ids": ["not-retrieved"],
                "conflicting_values": [],
            }
        ]
    }

    with pytest.raises(ValueError, match="not retrieved"):
        extract_vehicle_specs(
            "Ball joint torque?",
            [_evidence("real", "115 Nm", 24, "Specs")],
            llm=_mock_llm(response),
        )


def test_malformed_llm_json_is_rejected() -> None:
    with pytest.raises(ValueError):
        extract_vehicle_specs(
            "Ball joint torque?",
            [_evidence("chunk", "Torque 115 Nm.", 24, "Specs")],
            llm=lambda _system, _user: "{malformed",
        )


def test_no_evidence_returns_not_found_without_calling_llm() -> None:
    result = extract_vehicle_specs("fluid capacity?", [])

    assert result.results[0].status == "not_found"
    assert result.results[0].value is None
    assert result.results[0].citations == []


def test_empty_llm_result_is_reported_as_invalid() -> None:
    with pytest.raises(ValueError):
        extract_vehicle_specs(
            "Ball joint torque?",
            [_evidence("chunk", "Torque data unavailable.", 24, "Specs")],
            llm=_mock_llm({"results": []}),
        )


def test_csv_export_keeps_page_citations() -> None:
    evidence = _evidence(
        "chunk-8",
        "The suspension is in full rebound.",
        8,
        "Inspection",
    )
    response = {
        "results": [
            {
                "component": "Suspension",
                "spec_type": "Measurement position",
                "value": "full rebound",
                "unit": None,
                "status": "found",
                "evidence_ids": ["chunk-8"],
                "conflicting_values": [],
            }
        ]
    }
    result = extract_vehicle_specs(
        "Measurement position?",
        [evidence],
        llm=_mock_llm(response),
    )

    csv_text = export_csv(result)
    row = next(csv.DictReader(io.StringIO(csv_text)))
    citations = json.loads(row["citations"])
    assert citations[0]["page"] == 8
    assert "full rebound" in row["value"]


def test_json_export_is_valid_and_preserves_citation() -> None:
    evidence = _evidence(
        "chunk-24",
        "Fastener torque is 115 Nm.",
        24,
        "Specs",
    )
    response = {
        "results": [
            {
                "component": "Fastener",
                "spec_type": "Torque",
                "value": "115",
                "unit": "Nm",
                "status": "found",
                "evidence_ids": ["chunk-24"],
                "conflicting_values": [],
            }
        ]
    }
    result = extract_vehicle_specs(
        "Torque?",
        [evidence],
        llm=_mock_llm(response),
    )

    exported = json.loads(export_json(result))
    assert exported["results"][0]["citations"][0]["page"] == 24
    assert exported == result.model_dump(mode="json")


def test_evaluation_scores_conflicting_values_and_their_pages() -> None:
    case = {
        "id": "ambiguous-shock-torque",
        "relevant_pages": [59],
        "expected_status": "ambiguous",
        "expected_values": [
            {"value": "90", "unit": "Nm"},
            {"value": "550", "unit": "Nm"},
        ],
        "expected_citation_pages": [59],
    }
    retrieved = [
        {
            "source": {"page_number": 59},
        }
    ]
    result = {
        "results": [
            {
                "status": "ambiguous",
                "value": None,
                "citations": [{"page": 59}],
                "conflicting_values": [
                    {"value": "90", "unit": "Nm"},
                    {"value": "550", "unit": "Nm"},
                ],
            }
        ]
    }

    scored = _case_metrics(case, retrieved, result)

    assert scored["retrieval_recall_at_k"] == 1.0
    assert scored["extraction_correct"] is True
    assert scored["citation_correct"] is True


def test_invalid_candidate_response_fails_pydantic_validation() -> None:
    with pytest.raises(ValueError):
        CandidateResponse.model_validate_json('{"results":[]}')


def test_retrieval_failures_are_not_silently_returned_as_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenStore:
        def search(self, *_args, **_kwargs):
            raise RuntimeError("Chroma is unavailable")

    monkeypatch.setattr(pipeline, "manual_fingerprint", lambda _path: "fingerprint")
    monkeypatch.setattr(
        pipeline,
        "get_manual_store",
        lambda *_args, **_kwargs: BrokenStore(),
    )
    with pytest.raises(RuntimeError, match="Chroma is unavailable"):
        pipeline.retrieve_manual("manual.pdf", "find a specification")


def test_retrieval_filters_explicit_manual_section(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RecordingStore:
        where: dict | None = None

        def search(self, _query, *, top_k, where=None):
            self.where = where
            return []

    store = RecordingStore()
    monkeypatch.setattr(pipeline, "manual_fingerprint", lambda _path: "fingerprint")
    monkeypatch.setattr(
        pipeline,
        "get_manual_store",
        lambda *_args, **_kwargs: store,
    )

    pipeline.retrieve_manual(
        "manual.pdf",
        "What is the shock torque in Section 204-01B?",
        top_k=5,
    )

    assert store.where == {"section_id": "204-01B"}


def test_retrieval_reranks_an_explicit_procedure_section_hint() -> None:
    evidence = [
        {
            "text": "A ball joint part number is listed in the parts catalog.",
            "metadata": {"section": "Parts Catalog"},
            "source": {"section": "Parts Catalog"},
            "similarity": 0.85,
        },
        {
            "text": "Ball Joint Inspection. The suspension must be at full rebound.",
            "metadata": {"section": "SECTION 204-00 / Ball Joint Inspection"},
            "source": {"section": "SECTION 204-00 / Ball Joint Inspection"},
            "similarity": 0.55,
        },
    ]

    ranked = pipeline._rerank_by_query_terms(
        "What part number is specified in the Ball Joint Inspection procedure?",
        evidence,
        top_k=1,
    )

    assert ranked[0]["source"]["section"].endswith("Ball Joint Inspection")


def test_torque_retrieval_prefers_matching_suspension_spec_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RecordingStore:
        query: str = ""
        top_k: int = 0

        def search(self, query, *, top_k, where=None):
            self.query = query
            self.top_k = top_k
            return [
                _evidence(
                    "procedure",
                    "Ball joint inspection procedure, step 4.",
                    36,
                    "Unlabeled section",
                ),
                _evidence(
                    "wrong-section",
                    "Description Nm lb-ft\nUpper ball joint nut 150 111",
                    578,
                    "SECTION 205-04: Front Drive Halfshafts / Torque Specifications",
                ),
                _evidence(
                    "rwd-spec",
                    "Description Nm lb-ft lb-in\nUpper ball joint nut 115 85 -",
                    24,
                    "SECTION 204-01A: Front Suspension — RWD / Torque Specifications",
                ),
                _evidence(
                    "4wd-spec",
                    "Description Nm lb-ft lb-in\nUpper ball joint nut 115 85 -",
                    59,
                    "SECTION 204-01B: Front Suspension — 4WD / Torque Specifications",
                ),
            ]

    store = RecordingStore()
    monkeypatch.setattr(pipeline, "manual_fingerprint", lambda _path: "fingerprint")
    monkeypatch.setattr(
        pipeline,
        "get_manual_store",
        lambda *_args, **_kwargs: store,
    )

    evidence = pipeline.retrieve_manual(
        "manual.pdf",
        "What is the torque for the upper ball joint nut?",
        top_k=5,
    )

    assert "torque specifications table" in store.query
    assert store.top_k == 10
    assert [item["id"] for item in evidence] == ["rwd-spec", "4wd-spec"]


def test_llm_context_is_limited_to_three_evidence_passages() -> None:
    evidence = [
        _evidence(
            f"chunk-{index}",
            "Upper ball joint clearance is 12 mm.",
            24 + index,
            "Inspection",
        )
        for index in range(4)
    ]
    response = {
        "results": [
            {
                "component": "Upper ball joint",
                "spec_type": "clearance",
                "value": "12",
                "unit": "mm",
                "status": "found",
                "evidence_ids": ["chunk-0"],
                "conflicting_values": [],
            }
        ]
    }

    def check_prompt(system_prompt: str, user_prompt: str) -> str:
        assert "chunk-0" in user_prompt
        assert "chunk-2" in user_prompt
        assert "chunk-3" not in user_prompt
        return json.dumps(response)

    result = extract_vehicle_specs(
        "What is upper ball joint clearance?",
        evidence,
        llm=check_prompt,
    )

    assert result.results[0].status == "found"


def test_package_imports_from_project_root_without_legacy_project_path() -> None:
    project = Path(__file__).resolve().parents[1]
    code = """
import sys
from pathlib import Path
project = Path(sys.argv[1]).resolve()
import vehicle_spec_extractor.pipeline as pipeline
import vehicle_spec_extractor.streamlit_app as ui
assert Path(pipeline.__file__).resolve().is_relative_to(project)
assert Path(ui.__file__).resolve().is_relative_to(project)
assert pipeline.DEFAULT_PERSIST_DIRECTORY == project / "artifacts" / "chroma_db"
assert "app" not in sys.modules
"""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(project),
        ],
        cwd=project,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
