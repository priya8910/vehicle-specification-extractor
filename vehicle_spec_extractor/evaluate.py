from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vehicle_spec_extractor.config import (
    DEFAULT_PERSIST_DIRECTORY,
    EVALUATION_CASES_PATH,
)
from vehicle_spec_extractor.extraction import extract_vehicle_specs
from vehicle_spec_extractor.pipeline import (
    ingest_manual,
    retrieve_manual,
)


def _normalized(value: str) -> str:
    return " ".join(value.casefold().split())


def _case_metrics(
    case: dict[str, Any],
    retrieved: list[dict[str, Any]],
    result: dict[str, Any],
) -> dict[str, Any]:
    retrieved_pages = {
        int(item["source"]["page_number"])
        for item in retrieved
        if item["source"].get("page_number") is not None
    }
    relevant_pages = set(case["relevant_pages"])
    page_recall = (
        len(retrieved_pages & relevant_pages) / len(relevant_pages)
        if relevant_pages
        else 1.0
    )
    results = result["results"]
    expected_status = case["expected_status"]
    statuses = {item["status"] for item in results}
    expected_values = {
        (_normalized(item["value"]), item["unit"])
        for item in case["expected_values"]
    }
    expected_citation_pages = set(case["expected_citation_pages"])

    if expected_status == "not_found":
        extraction_correct = bool(results) and statuses == {"not_found"} and all(
            item["value"] is None and not item["citations"] for item in results
        )
        citation_correct = extraction_correct
        missing_detected = extraction_correct
    elif expected_status == "found":
        actual_values = {
            (_normalized(item["value"]), item["unit"])
            for item in results
            if item["status"] == "found" and item["value"] is not None
        }
        extraction_correct = (
            statuses == {"found"} and actual_values == expected_values
        )
        found_results = [
            item for item in results if item["status"] == "found"
        ]
        citation_correct = bool(found_results) and all(
            item["citations"]
            and all(
                citation["page"] in expected_citation_pages
                for citation in item["citations"]
            )
            for item in found_results
        )
        missing_detected = None
    elif expected_status == "ambiguous":
        actual_values = {
            (_normalized(claim["value"]), claim["unit"])
            for item in results
            for claim in item["conflicting_values"]
        }
        extraction_correct = (
            statuses == {"ambiguous"} and actual_values == expected_values
        )
        ambiguous_results = [
            item for item in results if item["status"] == "ambiguous"
        ]
        citation_correct = bool(ambiguous_results) and all(
            item["citations"]
            and all(
                citation["page"] in expected_citation_pages
                for citation in item["citations"]
            )
            for item in ambiguous_results
        )
        missing_detected = None
    else:
        raise ValueError(f"Unsupported expected status: {expected_status}")

    failed_checks: list[str] = []
    if page_recall != 1.0:
        failed_checks.append("retrieval_recall_at_k")
    if not extraction_correct:
        failed_checks.append("extraction")
    if not citation_correct:
        failed_checks.append("citation")
    if missing_detected is False:
        failed_checks.append("missing_value_detection")

    return {
        "id": case["id"],
        "expected_status": expected_status,
        "expected_values": case["expected_values"],
        "expected_citation_pages": sorted(expected_citation_pages),
        "retrieval_recall_at_k": page_recall,
        "extraction_correct": extraction_correct,
        "citation_correct": citation_correct,
        "missing_value_detected": missing_detected,
        "retrieved_pages": sorted(retrieved_pages),
        "extracted_statuses": sorted(statuses),
        "actual_values": [
            {"value": item.get("value"), "unit": item.get("unit")}
            for item in results
            if item.get("status") == "found"
        ]
        + [
            {"value": claim.get("value"), "unit": claim.get("unit")}
            for item in results
            for claim in item.get("conflicting_values", [])
        ],
        "actual_citation_pages": sorted(
            {
                citation.get("page")
                for item in results
                for citation in item.get("citations", [])
            }
        ),
        "failed_checks": failed_checks,
    }


def run_evaluation(
    pdf_path: str | Path,
    *,
    top_k: int = 5,
    persist_directory: str | Path = DEFAULT_PERSIST_DIRECTORY,
) -> dict[str, Any]:
    cases = json.loads(EVALUATION_CASES_PATH.read_text(encoding="utf-8"))
    ingestion = ingest_manual(
        pdf_path,
        persist_directory=persist_directory,
    )
    scored = []
    for case in cases:
        retrieved = retrieve_manual(
            pdf_path,
            case["query"],
            top_k=top_k,
            persist_directory=persist_directory,
        )
        try:
            extraction = extract_vehicle_specs(case["query"], retrieved)
        except (RuntimeError, ValueError) as error:
            failed_case = _case_metrics(case, retrieved, {"results": []})
            failed_case["failed_checks"].append("extraction_error")
            failed_case["error"] = f"{type(error).__name__}: {error}"
            scored.append(failed_case)
            continue
        scored.append(
            _case_metrics(
                case,
                retrieved,
                extraction.model_dump(mode="json"),
            )
        )

    count = len(scored)
    missing_cases = [
        item for item in scored if item["missing_value_detected"] is not None
    ]
    return {
        "manual": ingestion["source_file"],
        "top_k": top_k,
        "cases": scored,
        "metrics": {
            "retrieval_recall_at_k": (
                sum(item["retrieval_recall_at_k"] for item in scored) / count
                if count
                else 0.0
            ),
            "extraction_accuracy": (
                sum(item["extraction_correct"] for item in scored) / count
                if count
                else 0.0
            ),
            "citation_accuracy": (
                sum(item["citation_correct"] for item in scored) / count
                if count
                else 0.0
            ),
            "missing_value_detection": (
                sum(item["missing_value_detected"] for item in missing_cases)
                / len(missing_cases)
                if missing_cases
                else None
            ),
            "case_count": count,
            "missing_case_count": len(missing_cases),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="vehicle-spec-evaluate",
        description=(
            "Run retrieval and extraction on the manually checked evaluation set."
        ),
    )
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=DEFAULT_PERSIST_DIRECTORY,
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the full JSON report to this path instead of stdout only.",
    )
    args = parser.parse_args()
    try:
        report = run_evaluation(
            args.pdf,
            top_k=args.top_k,
            persist_directory=args.persist_dir,
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(status=1, message=f"Evaluation failed: {error}\n")
    output = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n", encoding="utf-8")
        print(f"Evaluation report written to {output_path}")
    print(output)


if __name__ == "__main__":
    main()
