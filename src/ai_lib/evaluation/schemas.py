"""Explicit answer and evaluator output contracts."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class SupportAnswer(BaseModel):
    """A response and its evidence IDs; valid structure does not establish truth."""
    model_config = ConfigDict(extra="forbid")
    answer: str = Field(min_length=1)
    source_ids: list[str]
    disposition: Literal["answer", "clarify"]


class OrdinalJudgment(BaseModel):
    """A strict rubric anchor with a short justification, not a probability."""
    model_config = ConfigDict(extra="forbid")
    reasoning: str = Field(min_length=1)
    score: int = Field(ge=0, le=3, strict=True)


def evaluator_schema(evaluator_type):
    """Explicit score contract used in evaluation and cache identities."""
    if evaluator_type == "ordinal":
        return OrdinalJudgment.model_json_schema()
    if evaluator_type == "binary":
        return "binary-v1"
    raise ValueError("Evaluator type must be binary or ordinal.")
