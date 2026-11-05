"""One fitted pipeline for direct, API and batch prediction."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ml_lib.data.loading import sha256_file
from ml_lib.models.classical import load_regressor
from .schemas import Features, Prediction


def frame(records):
    return pd.DataFrame([Features.model_validate(
        {key: record.model_dump()[key] for key in Features.model_fields}
    ).model_dump() for record in records]).replace({None: np.nan})


class Predictor:
    def __init__(self, directory: Path):
        self.metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if self.metadata["model_sha256"] != sha256_file(directory / "model.pkl"):
            raise ValueError("Model checksum mismatch; restore the complete trusted bundle.")
        self.model = load_regressor(directory / "model.pkl", trusted=True)
        self.version = self.metadata["version"]

    def predict(self, records: list[Features]) -> list[Prediction]:
        if not records:
            return []
        values = self.model.predict(frame(records))
        if len(values) != len(records) or not np.isfinite(values).all():
            raise RuntimeError("The model returned invalid predictions.")
        return [Prediction(estimate=float(value), model_version=self.version,
                           status="ok" if value > 0 else "review_required") for value in values]
