"""Poll the upstream platform and atomically publish each batch's results."""
import hashlib
import io
import time
import pandas as pd
from pydantic import ValidationError
from lab_helpers.s10_shared.prediction.schemas import Listing
from lab_helpers.s10_shared.runtime.logging import event


def normalize_listing(record):
    """Translate CSV blanks and integer-looking capacity into the input contract."""
    record = dict(record)
    for key in ["bedrooms", "bathrooms", "beds"]:
        if record.get(key) == "":
            record[key] = None
    if "accommodates" in record:
        value = float(record["accommodates"])
        if not value.is_integer():
            raise ValueError("Capacity must be an integer")
        record["accommodates"] = int(value)
    return Listing.model_validate(record)


def process(store, predictor, batch_id, checksum, content):
    if hashlib.sha256(content).hexdigest() != checksum:
        raise ValueError("Batch checksum mismatch; retry discovery before processing.")
    if not store.begin_batch(batch_id, checksum):
        return None
    started = time.perf_counter()
    try:
        table = pd.read_csv(io.BytesIO(content), dtype={"listing_id": str}, keep_default_na=False)
        accepted, rejected = [], []
        for start in range(0, len(table), 128):
            valid, positions = [], []
            for position, row in table.iloc[start:start + 128].iterrows():
                record = row.to_dict()
                try:
                    listing = normalize_listing(record)
                    valid.append(listing)
                    positions.append(int(position) + 1)
                except (ValidationError, ValueError, TypeError, OverflowError):
                    rejected.append((int(position) + 1, str(record.get("listing_id", "")), "invalid_input"))
            predictions = predictor.predict(valid)
            accepted.extend(zip(positions, valid, predictions))
        elapsed = time.perf_counter() - started
        counts = store.finish_batch(batch_id, accepted, rejected, elapsed)
        event("batch", batch_id=batch_id, outcome="completed", seconds=elapsed,
              model_version=predictor.version, **counts)
        return counts
    except Exception as error:
        # Record state, then preserve the original failure for the caller.
        store.fail_batch(batch_id, type(error).__name__)
        raise


def poll(store, predictor, client):
    response = client.get("/batches")
    response.raise_for_status()
    for batch in response.json():
        previous = store.batch(batch["id"])
        if previous and previous["checksum"] != batch["sha256"]:
            event("batch", batch_id=batch["id"], outcome="conflict", error_type="ChangedBatch")
            continue
        if previous and previous["status"] == "completed":
            continue
        result = client.get(f"/batches/{batch['id']}")
        result.raise_for_status()
        process(store, predictor, batch["id"], batch["sha256"], result.content)
