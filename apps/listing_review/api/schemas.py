"""Public responses; versioned historical AI payloads remain readable."""
from typing import Any, Literal
from pydantic import Field
from lab_helpers.s10_shared.contracts import Contract
from lab_helpers.s10_shared.prediction.schemas import Listing, Prediction

JobStatus = Literal["queued", "running", "completed", "failed"]


class ValidationIssue(Contract):
    field: str
    type: str


class ErrorResponse(Contract):
    code: str
    message: str
    request_id: str
    details: list[ValidationIssue] = Field(default_factory=list)


class LiveResponse(Contract):
    status: Literal["ok"]


class ReadyResponse(Contract):
    status: Literal["ready"]
    model_version: str


class StoredListing(Contract):
    batch_id: str
    listing: Listing
    prediction: Prediction


class WorkspaceItem(StoredListing):
    review_status: Literal["not_requested", "queued", "running", "completed", "failed"]


class WorkspaceSummary(Contract):
    listings: int
    in_progress: int
    completed: int


class FilterOptions(Contract):
    neighbourhood: list[str]
    room_type: list[str]
    batch_id: list[str]


class WorkspaceResponse(Contract):
    summary: WorkspaceSummary
    options: FilterOptions
    total: int
    items: list[WorkspaceItem]


class BatchResponse(Contract):
    id: str
    checksum: str
    status: Literal["running", "completed", "failed"]
    total: int
    accepted: int
    rejected: int
    seconds: float
    error: str | None
    updated: str = Field(description="UTC timestamp saved by SQLite.")


class RejectionResponse(Contract):
    batch_id: str
    row_number: int
    listing_id: str | None
    reason: str


class ReviewAccepted(Contract):
    job_id: str
    status_url: str = Field(description="Poll this URL for status and saved outputs.")


class ReviewSummary(Contract):
    id: str
    status: JobStatus
    error: str | None
    created: str = Field(description="UTC timestamp saved by SQLite.")


class JobResponse(ReviewSummary):
    request_id: str = Field(description="Client idempotency key, not the HTTP tracing ID.")
    listing_id: str
    updated: str
    snapshot: dict[str, Any] = Field(description="Immutable inputs and instructions; historical formats preserved.")
    stages: dict[str, Any] = Field(description="Versioned stage outputs, timings and usage.")
    result: dict[str, Any] | None = Field(description="Versioned proposal or null. Historical results are not rewritten.")
