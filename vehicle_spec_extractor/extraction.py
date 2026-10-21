from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from typing import Any

from vehicle_spec_extractor.config import DEFAULT_LLM_MODEL
from vehicle_spec_extractor.schemas import (
    CandidateResponse,
    EvidenceCitation,
    ExtractionResult,
    VerifiedConflict,
    VehicleSpecification,
)


LLMCallable = Callable[[str, str], str]

SYSTEM_PROMPT = """You extract vehicle information from service-manual evidence.
Use only the supplied passages. Never use outside knowledge or infer missing
numbers, units, part numbers, or procedures. Preserve every value and unit
exactly as printed. Return JSON matching the requested schema.

Return results only for the component/specification asked about. For a supported
answer, set status to "found", include the exact value, and cite only evidence
IDs whose passage supports it. When the question requests all torque table
units, return one found result for every populated numeric unit column; omit
dash/blank columns and do not omit a populated unit column. When the manual
shows different values for different trims and the query leaves the trim
unspecified, set status to "ambiguous", omit value/unit, and include every
distinct value/unit with the evidence ID for its row. Do not resolve trim
differences by choosing one row. For procedures, return the concise step or
instruction asked for, preserving its wording from the evidence. When evidence
is insufficient, return status "not_found", value/unit null, and empty evidence
IDs/conflicting values. Do not treat an instruction to refer to another section
as the specification itself.
"""

_GROQ_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "component": {"type": "string"},
                    "spec_type": {"type": "string"},
                    "value": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "null"},
                        ]
                    },
                    "unit": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "null"},
                        ]
                    },
                    "status": {
                        "type": "string",
                        "enum": ["found", "not_found", "ambiguous"],
                    },
                    "evidence_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "conflicting_values": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "value": {"type": "string"},
                                "unit": {
                                    "anyOf": [
                                        {"type": "string"},
                                        {"type": "null"},
                                    ]
                                },
                                "evidence_id": {"type": "string"},
                            },
                            "required": ["value", "unit", "evidence_id"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": [
                    "component",
                    "spec_type",
                    "value",
                    "unit",
                    "status",
                    "evidence_ids",
                    "conflicting_values",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["results"],
    "additionalProperties": False,
}


def _call_groq(system_prompt: str, user_prompt: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is required for extraction. Configure it in the "
            "project-root .env file or environment; ingestion and retrieval "
            "do not require a key."
        )

    from groq import APIError, Groq

    client = Groq(api_key=api_key)
    response = None
    for attempt in range(2):
        retry_note = (
            "\n\nReturn valid JSON only. Ensure the response matches the exact "
            "JSON object shape requested above."
            if attempt
            else ""
        )
        try:
            response = client.chat.completions.create(
                model=DEFAULT_LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt + retry_note},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "vehicle_specification_extraction",
                        "strict": True,
                        "schema": _GROQ_RESPONSE_SCHEMA,
                    },
                },
                temperature=0,
                max_tokens=1400,
            )
            break
        except APIError as error:
            error_text = str(error).casefold()
            if attempt == 0 and (
                "json_validate_failed" in error_text
                or "connection error" in error_text
            ):
                continue
            raise RuntimeError(f"Groq extraction request failed: {error}") from error
    if response is None:
        raise RuntimeError("Groq extraction request returned no response.")
    if not response.choices:
        raise RuntimeError("The LLM returned no completion choices.")
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("The LLM returned an empty response.")
    return content


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).casefold().strip()


def _component_words_match(component: str, line: str) -> bool:
    component_words = re.findall(r"[a-z0-9]+", _normalized(component))
    line_words = re.findall(r"[a-z0-9]+", _normalized(line))
    if not component_words or len(line_words) < len(component_words):
        return False
    for index, word in enumerate(component_words):
        row_word = line_words[index]
        if word == row_word:
            continue
        if (
            index == len(component_words) - 1
            and word.rstrip("s") == row_word.rstrip("s")
        ):
            continue
        return False
    return True


def _component_is_mentioned(component: str, passage: str) -> bool:
    component_words = re.findall(r"[a-z0-9]+", _normalized(component))
    passage_words = re.findall(r"[a-z0-9]+", _normalized(passage))
    if not component_words or len(passage_words) < len(component_words):
        return False
    for start in range(len(passage_words) - len(component_words) + 1):
        if all(
            word == passage_words[start + index]
            or (
                index == len(component_words) - 1
                and word.rstrip("s") == passage_words[start + index].rstrip("s")
            )
            for index, word in enumerate(component_words)
        ):
            return True
    return False


def _torque_table_supports(
    component: str,
    value: str,
    unit: str,
    passage: str,
) -> bool | None:
    lines = passage.splitlines()
    unit_headers: list[str] = []
    for line in lines:
        if "description" in _normalized(line):
            headers = re.findall(r"\b(?:nm|lb-ft|lb-in)\b", _normalized(line))
            if headers:
                unit_headers = headers
                break
    if not unit_headers:
        return None

    numeric_pattern = re.compile(r"(?<![\w.])[+-]?\d+(?:\.\d+)?(?![\w.])")
    for line in lines:
        if not _component_words_match(component, line):
            continue
        numbers = list(numeric_pattern.finditer(line))
        if not numbers:
            continue
        if len(numbers) >= 3:
            pairs = zip(numbers, unit_headers)
        elif len(numbers) == 2:
            between = line[numbers[0].end() : numbers[1].start()]
            unit_indexes = (0, 2) if re.search(r"[—–-]", between) else (0, 1)
            pairs = (
                (numbers[0], unit_headers[unit_indexes[0]]),
                (numbers[1], unit_headers[unit_indexes[1]]),
            )
        else:
            pairs = ((numbers[0], unit_headers[0]),)
        if any(
            match.group() == value and expected_unit == _normalized(unit)
            for match, expected_unit in pairs
        ):
            return True
    return False


def _claim_is_supported(
    value: str,
    unit: str | None,
    passage: str,
    *,
    component: str,
    spec_type: str,
) -> bool:
    if not _component_is_mentioned(component, passage):
        return False

    if "torque" in spec_type.casefold() and unit:
        table_support = _torque_table_supports(
            component,
            value,
            unit,
            passage,
        )
        if table_support is not None:
            return table_support

    # PDF extraction may wrap a sentence or table heading across lines. Keep
    # the component and claim in the same retrieved chunk, but do not require
    # them to occupy the same extracted line.
    normalized_passage = _normalized(passage)
    normalized_value = _normalized(value)
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", normalized_value):
        value_pattern = (
            r"(?<![\w.+-])"
            + re.escape(normalized_value)
            + r"(?!\w|\.\d)"
        )
    else:
        value_pattern = (
            r"(?<!\w)"
            + re.escape(normalized_value)
            + r"(?!\w)"
        )
    if not re.search(value_pattern, normalized_passage):
        return False
    if not unit:
        return True

    normalized_unit = _normalized(unit)
    unit_pattern = (
        r"(?<!\w)"
        + re.escape(normalized_unit)
        + r"(?!\w)"
    )
    return bool(re.search(unit_pattern, normalized_passage))


def _citation(
    evidence_id: str,
    evidence: dict[str, Any],
) -> EvidenceCitation:
    metadata = evidence.get("metadata", {})
    source = evidence.get("source", {})
    raw_page = metadata.get("page_number", source.get("page_number"))
    page = int(raw_page) if raw_page not in (None, "") else None
    section = str(
        metadata.get("section")
        or source.get("section")
        or "Unlabeled section"
    )
    source_file = str(
        metadata.get("source_file")
        or metadata.get("file_name")
        or source.get("file_name")
        or "Unknown PDF"
    )
    return EvidenceCitation(
        evidence_id=evidence_id,
        source_file=source_file,
        page=page,
        section=section,
        passage=str(evidence.get("text", "")),
    )


def _normalize_candidate_payload(payload: Any) -> Any:
    """Discard claims whose status and evidence fields contradict each other."""
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        return payload

    for item in payload["results"]:
        if not isinstance(item, dict):
            continue
        status = item.get("status")
        invalid_found = status == "found" and (
            not item.get("value")
            or not isinstance(item.get("evidence_ids"), list)
            or not item.get("evidence_ids")
            or item.get("conflicting_values")
        )
        conflicts = item.get("conflicting_values")
        invalid_ambiguous = status == "ambiguous" and (
            item.get("value") is not None
            or item.get("unit") is not None
            or not isinstance(conflicts, list)
            or len(conflicts) < 2
        )
        if status == "ambiguous" and not invalid_ambiguous:
            invalid_ambiguous = any(
                not isinstance(conflict, dict)
                or not conflict.get("value")
                or not conflict.get("evidence_id")
                for conflict in conflicts
            ) or len(
                {
                    (str(conflict.get("value")), str(conflict.get("unit")))
                    for conflict in conflicts
                }
            ) < 2

        if status == "not_found" or invalid_found or invalid_ambiguous:
            if status != "not_found":
                item["status"] = "not_found"
            item["value"] = None
            item["unit"] = None
            item["evidence_ids"] = []
            item["conflicting_values"] = []
    return payload


def extract_vehicle_specs(
    query: str,
    evidence: list[dict[str, Any]],
    *,
    llm: LLMCallable | None = None,
) -> ExtractionResult:
    if not query.strip():
        raise ValueError("Query cannot be empty.")
    if not evidence:
        return ExtractionResult(
            query=query,
            results=[
                VehicleSpecification(
                    component=query.strip(),
                    spec_type="unspecified",
                    value=None,
                    unit=None,
                    status="not_found",
                )
            ],
        )

    evidence = evidence[:3]
    evidence_by_id = {str(item["id"]): item for item in evidence}
    evidence_payload = []
    for evidence_id, item in evidence_by_id.items():
        citation = _citation(evidence_id, item)
        evidence_payload.append(
            {
                "evidence_id": evidence_id,
                "page": citation.page,
                "section": citation.section,
                "passage": item.get("text", ""),
            }
        )
    user_prompt = (
        "Question:\n"
        f"{query.strip()}\n\n"
        "Evidence passages (cite only the exact evidence_id values shown):\n"
        f"{json.dumps(evidence_payload, ensure_ascii=False)}\n\n"
        "Return exactly one JSON object with this shape:\n"
        '{"results":[{"component":"...","spec_type":"...","value":"... or null",'
        '"unit":"... or null","status":"found|not_found|ambiguous",'
        '"evidence_ids":["..."],"conflicting_values":['
        '{"value":"...","unit":"... or null","evidence_id":"..."}]}]}'
    )
    response_text = (llm or _call_groq)(SYSTEM_PROMPT, user_prompt)
    # Model responses sometimes contradict their status fields. Repair these
    # cases conservatively before applying the strict schema: discard malformed
    # claims rather than accepting a value that cannot be fully cited.
    candidate_payload = _normalize_candidate_payload(json.loads(response_text))
    candidates = CandidateResponse.model_validate(candidate_payload)
    verified: list[VehicleSpecification] = []

    for candidate in candidates.results:
        referenced_ids = set(candidate.evidence_ids)
        referenced_ids.update(
            item.evidence_id for item in candidate.conflicting_values
        )
        unknown_ids = referenced_ids.difference(evidence_by_id)
        if unknown_ids:
            raise ValueError(
                "LLM cited evidence IDs that were not retrieved: "
                + ", ".join(sorted(unknown_ids))
            )

        if candidate.status == "not_found":
            verified.append(
                VehicleSpecification(
                    component=candidate.component,
                    spec_type=candidate.spec_type,
                    value=None,
                    unit=None,
                    status="not_found",
                )
            )
            continue

        if candidate.status == "found":
            cited = [
                _citation(evidence_id, evidence_by_id[evidence_id])
                for evidence_id in candidate.evidence_ids
            ]
            supported_citations = [
                citation
                for citation in cited
                if candidate.value
                and _claim_is_supported(
                    candidate.value,
                    candidate.unit,
                    citation.passage,
                    component=candidate.component,
                    spec_type=candidate.spec_type,
                )
            ]
            if supported_citations:
                verified.append(
                    VehicleSpecification(
                        component=candidate.component,
                        spec_type=candidate.spec_type,
                        value=candidate.value,
                        unit=candidate.unit,
                        status="found",
                        citations=supported_citations,
                    )
                )
            else:
                verified.append(
                    VehicleSpecification(
                        component=candidate.component,
                        spec_type=candidate.spec_type,
                        value=None,
                        unit=None,
                        status="not_found",
                    )
                )
            continue

        conflicts: list[VerifiedConflict] = []
        for conflict in candidate.conflicting_values:
            citation = _citation(
                conflict.evidence_id,
                evidence_by_id[conflict.evidence_id],
            )
            if not _claim_is_supported(
                conflict.value,
                conflict.unit,
                citation.passage,
                component=candidate.component,
                spec_type=candidate.spec_type,
            ):
                conflicts = []
                break
            conflicts.append(
                VerifiedConflict(
                    value=conflict.value,
                    unit=conflict.unit,
                    citation=citation,
                )
            )
        if len(conflicts) >= 2:
            verified.append(
                VehicleSpecification(
                    component=candidate.component,
                    spec_type=candidate.spec_type,
                    value=None,
                    unit=None,
                    status="ambiguous",
                    citations=[item.citation for item in conflicts],
                    conflicting_values=conflicts,
                )
            )
        else:
            verified.append(
                VehicleSpecification(
                    component=candidate.component,
                    spec_type=candidate.spec_type,
                    value=None,
                    unit=None,
                    status="not_found",
                )
            )

    return ExtractionResult(query=query, results=verified)


def export_csv(result: ExtractionResult) -> str:
    import csv
    import io

    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "query",
            "component",
            "spec_type",
            "value",
            "unit",
            "status",
            "citations",
            "conflicting_values",
        ],
    )
    writer.writeheader()
    for item in result.results:
        row = item.model_dump(mode="json")
        row["query"] = result.query
        row["citations"] = json.dumps(row["citations"], ensure_ascii=False)
        row["conflicting_values"] = json.dumps(
            row["conflicting_values"],
            ensure_ascii=False,
        )
        writer.writerow(row)
    return output.getvalue()


def export_json(result: ExtractionResult) -> str:
    return result.model_dump_json(indent=2)
