"""Listing inputs and numerical prediction responses."""
from typing import Annotated, Literal
from pydantic import Field
from ..contracts import Contract

Text = Annotated[str, Field(min_length=1, max_length=200)]
Number = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class Features(Contract):
    accommodates: Annotated[int, Field(ge=1, le=100, strict=True)]
    bedrooms: Number | None = None
    bathrooms: Number | None = None
    beds: Number | None = None
    property_type: Text
    room_type: Literal["Entire home/apt", "Private room", "Shared room", "Hotel room"]
    neighbourhood: Text


class Listing(Features):
    listing_id: Annotated[str, Field(pattern=r"^[0-9]{1,25}$")]
    description: Annotated[str, Field(max_length=20000)] = ""
    advertised_price: Annotated[float, Field(gt=0, allow_inf_nan=False)]


class Prediction(Contract):
    estimate: Annotated[float, Field(allow_inf_nan=False)]
    currency: Literal["CAD"] = "CAD"
    model_version: Annotated[str, Field(min_length=1)]
    status: Literal["ok", "review_required"]


class PredictionResponse(Prediction):
    request_id: Annotated[str, Field(min_length=1)]
