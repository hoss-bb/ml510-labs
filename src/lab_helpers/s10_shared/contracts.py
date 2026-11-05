"""Shared strict input and output contract."""
from pydantic import BaseModel, ConfigDict


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
