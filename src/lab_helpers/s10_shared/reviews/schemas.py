"""Review requests, evidence and action-specific outputs."""
from typing import Annotated, Literal
from pydantic import Field, model_validator
from ..contracts import Contract


class ReviewRequest(Contract):
    request_id: Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{8,80}$",
        description="Client idempotency key. Reuse for network retries; choose a new key for a new review attempt.",
        examples=["review-attempt-001"])]


class ReviewContext(Contract):
    amenities: list[str] | None = None
    bathroom_description: str | None = None
    minimum_nights: Annotated[int, Field(ge=1, strict=True)] | None = None


class EvidenceItem(Contract):
    claim: Annotated[str, Field(min_length=1, max_length=400)]
    source_id: str
    excerpt: Annotated[str, Field(min_length=1, max_length=600)]


class Evidence(Contract):
    items: list[EvidenceItem] = Field(max_length=10)


class Finding(Contract):
    kind: Literal["conflict", "clarification", "guest_condition"]
    observation: Annotated[str, Field(min_length=1, max_length=500)]
    evidence_ids: list[str] = Field(min_length=1, max_length=12)

    @model_validator(mode="after")
    def conflict_has_two_facts(self):
        if self.kind == "conflict" and len(set(self.evidence_ids)) < 2:
            raise ValueError("A conflict must identify two different facts.")
        return self


class Assessment(Contract):
    findings: list[Finding] = Field(max_length=6)
    summary: Annotated[str, Field(min_length=1, max_length=700)]


class Proposal(Contract):
    action: Literal["confirm_facts", "clarify_condition", "guest_note", "no_change"]
    reason: Annotated[str, Field(min_length=1, max_length=700)]
    next_step: Annotated[str, Field(min_length=1, max_length=500)]
    evidence_ids: list[str] = Field(max_length=12)
    owner_question: Annotated[str, Field(min_length=1, max_length=700)] | None
    guest_note: Annotated[str, Field(min_length=1, max_length=1000)] | None
    confirm_before_use: list[str] = Field(max_length=4)

    @model_validator(mode="after")
    def appropriate_draft(self):
        if self.action in {"confirm_facts", "clarify_condition"}:
            if not self.owner_question or self.guest_note or not self.confirm_before_use:
                raise ValueError("Confirmation actions need an owner question and facts to confirm, without a guest draft.")
        elif self.action == "guest_note":
            if not self.guest_note or self.owner_question or self.confirm_before_use:
                raise ValueError("A guest note uses established facts only, without an owner question.")
        elif self.owner_question or self.guest_note or self.confirm_before_use:
            raise ValueError("No change must not invent a draft or confirmation requirement.")
        if self.action != "no_change" and not self.evidence_ids:
            raise ValueError("An action must cite supporting evidence.")
        return self
