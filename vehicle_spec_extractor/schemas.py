from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


Status = Literal["found", "not_found", "ambiguous"]


class ConflictingValue(BaseModel):
    value: str = Field(min_length=1)
    unit: str | None = None
    evidence_id: str = Field(min_length=1)


class CandidateSpec(BaseModel):
    component: str = Field(min_length=1)
    spec_type: str = Field(min_length=1)
    value: str | None = None
    unit: str | None = None
    status: Status
    evidence_ids: list[str] = Field(default_factory=list)
    conflicting_values: list[ConflictingValue] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_status_fields(self) -> CandidateSpec:
        if self.status == "found" and (
            not self.value or not self.evidence_ids or self.conflicting_values
        ):
            raise ValueError(
                "A found specification needs a value and evidence IDs only."
            )
        if self.status == "not_found" and (
            self.value is not None
            or self.unit is not None
            or self.evidence_ids
            or self.conflicting_values
        ):
            raise ValueError(
                "A not_found specification cannot contain a value or citations."
            )
        if self.status == "ambiguous":
            values = {
                (item.value, item.unit) for item in self.conflicting_values
            }
            if (
                self.value is not None
                or self.unit is not None
                or len(self.conflicting_values) < 2
                or len(values) < 2
            ):
                raise ValueError(
                    "An ambiguous specification needs at least two distinct, "
                    "evidence-linked values and no single value."
                )
        return self


class CandidateResponse(BaseModel):
    results: list[CandidateSpec] = Field(min_length=1)


class EvidenceCitation(BaseModel):
    evidence_id: str
    source_file: str
    page: int | None
    section: str
    passage: str


class VerifiedConflict(BaseModel):
    value: str
    unit: str | None
    citation: EvidenceCitation


class VehicleSpecification(BaseModel):
    component: str
    spec_type: str
    value: str | None
    unit: str | None
    status: Status
    citations: list[EvidenceCitation] = Field(default_factory=list)
    conflicting_values: list[VerifiedConflict] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    query: str
    results: list[VehicleSpecification]
