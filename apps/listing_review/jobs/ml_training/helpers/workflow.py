"""Prepare missing inputs, verify reusable data, then save a new candidate model."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import uuid
import pandas as pd
from ml_lib.data.loading import sha256_file
from lab_helpers.s10_shared.runtime.settings import ROOT, settings
from lab_helpers.s10_shared.prediction.schemas import Listing
from lab_helpers.s10_shared.reviews.context import ContextCatalog
from . import prepare as preparation
from .context import prepare_context
from lab_helpers.s10_shared.evaluation import check_partitions
from .train import train


def ensure_prepared(data, simulator):
    prepared = data / "prepared"
    if not prepared.exists():
        preparation.prepare(data, simulator_data=simulator)
    required = [prepared / name for name in ["manifest.json", "listings.csv", "excluded.csv"]]
    if not all(p.is_file() for p in required) or list(prepared.glob("*.tmp")):
        raise ValueError("Incomplete preparation. Restore the complete prepared folder or use a new data folder.")
    manifest = json.loads(required[0].read_text(encoding="utf-8"))
    if manifest.get("raw_sha256") != preparation.SOURCE_SHA256 or manifest.get("source") != preparation.SOURCE:
        raise ValueError("Prepared data belongs to a different snapshot; use the recorded source.")
    if sha256_file(required[1]) != manifest.get("prepared_sha256"):
        raise ValueError("Prepared data fingerprint differs. Restore it or prepare a new data folder.")
    table = pd.read_csv(required[1], dtype={"listing_id": str, "host_id": str})
    check_partitions(table)
    if table.split.value_counts().to_dict() != manifest.get("split_counts") or len(table) != manifest.get("accepted_rows"):
        raise ValueError("Prepared counts disagree with the manifest.")
    raw = data / "raw/listings-2026-06-15.csv.gz"
    preparation.download(raw)
    if sha256_file(raw) != manifest["raw_sha256"]:
        raise ValueError("Raw snapshot fingerprint differs; restore the recorded source.")

    context_file = prepared / "review_context.json"
    if not context_file.exists():
        prepare_context(data)
    catalog = ContextCatalog(context_file)
    if not catalog.provenance or catalog.provenance.get("raw_sha256") != manifest["raw_sha256"]:
        raise ValueError("AI context belongs to a different snapshot.")
    if set(catalog.entries) != set(table.listing_id):
        raise ValueError("AI context has incomplete or unexpected listing coverage.")
    for record in table.to_dict("records"):
        fields = {key: record[key] for key in Listing.model_fields}
        for key, value in fields.items():
            if pd.isna(value):
                fields[key] = "" if key == "description" else None
        if catalog.for_listing(Listing.model_validate(fields).model_dump())["provenance"] is None:
            raise ValueError("AI context does not match the prepared listing records.")

    # Check every existing example before writing missing ones. Published incoming files are untouched.
    arrivals = simulator / "arrival_examples"
    held_out = table[table.split == "test"].sort_values("listing_id")
    expected = {f"day_{i:03d}.csv": held_out.iloc[start:start+250].drop(columns=["host_id", "split"])
                for i, start in enumerate(range(0, len(held_out), 250), 1)}
    if {p.name for p in arrivals.glob("day_*.csv")} - set(expected):
        raise ValueError("Unexpected arrival batches; use a matching simulator data folder.")
    for name, batch in expected.items():
        path = arrivals / name
        if path.exists():
            actual = pd.read_csv(path, dtype={"listing_id": str})
            try:
                pd.testing.assert_frame_equal(actual.reset_index(drop=True), batch.reset_index(drop=True),
                                              check_dtype=False, rtol=1e-12, atol=1e-12)
            except AssertionError as error:
                raise ValueError(f"Arrival example {name} differs from held-out data; it was not overwritten.") from error
    arrivals.mkdir(parents=True, exist_ok=True)
    for name, batch in expected.items():
        path = arrivals / name
        if not path.exists():
            temporary = path.with_suffix(".tmp")
            batch.to_csv(temporary, index=False)
            temporary.replace(path)
    print("Prepared data verified; existing files reused. Missing supporting files prepared if needed.")


def run_job(ridge_alpha=10.0):
    config = settings()
    simulator = Path(os.environ.get("LISTINGS_SIMULATOR_DATA", str(ROOT / "local/listings_simulator")))
    ensure_prepared(config.data, simulator)
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    candidate = config.data / "candidates" / name
    evidence = train(ridge_alpha, data_directory=config.data, output_directory=candidate)
    print(f"Candidate: {candidate}\nEvaluation: {evidence}\nThe deployed model was not replaced.")
    return candidate, evidence
