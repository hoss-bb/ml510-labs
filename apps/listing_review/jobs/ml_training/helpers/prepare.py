"""Download one immutable snapshot and separate hosts before training."""
import html
import json
import os
import re
from pathlib import Path
import httpx
import pandas as pd
from pydantic import ValidationError
from sklearn.model_selection import GroupShuffleSplit
from ml_lib.data.loading import sha256_file
from ml_lib.experiment.artifacts import write_json
from lab_helpers.s10_shared.runtime.settings import ROOT
from lab_helpers.s10_shared.prediction.schemas import Listing

SOURCE = "https://data.insideairbnb.com/canada/qc/montreal/2026-06-15/data/listings.csv.gz"
SOURCE_SHA256 = "413f42dc60a8d778093bef340f161801d4d936c392e9aff9917946fc8260ad49"


def download(destination):
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".part")
    try:
        with httpx.stream("GET", SOURCE, follow_redirects=True, timeout=120) as response:
            response.raise_for_status()
            with partial.open("wb") as stream:
                for chunk in response.iter_bytes():
                    stream.write(chunk)
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def prepare(data: Path, *, simulator_data=None):
    destination = data / "prepared"
    if destination.exists():
        raise FileExistsError("Prepared data already exists. Use a new S10_DATA folder for a new preparation.")
    simulator_data = Path(simulator_data) if simulator_data is not None else Path(os.environ.get("LISTINGS_SIMULATOR_DATA", str(ROOT / "local/listings_simulator")))
    arrivals = simulator_data / "arrival_examples"
    if any(arrivals.glob("day_*.csv")):
        raise FileExistsError("Arrival examples already exist. Use a new LISTINGS_SIMULATOR_DATA folder for a new preparation.")
    raw = data / "raw" / "listings-2026-06-15.csv.gz"
    download(raw)
    if sha256_file(raw) != SOURCE_SHA256:
        raise ValueError("Snapshot checksum differs from the recorded source; investigate before preparation.")
    source = pd.read_csv(raw, dtype={"id": str, "host_id": str})
    accepted, rejected = [], []
    for position, row in source.iterrows():
        try:
            price = float(str(row["price"]).replace("$", "").replace(",", ""))
            numeric = {}
            for name in ["bedrooms", "bathrooms", "beds"]:
                value = row.get(name)
                numeric[name] = None if pd.isna(value) else float(value)
            capacity = float(row["accommodates"])
            if not capacity.is_integer():
                raise ValueError("Capacity must be an integer")
            record = Listing(
                listing_id=row["id"], accommodates=int(capacity), **numeric,
                property_type=row["property_type"], room_type=row["room_type"],
                neighbourhood=row["neighbourhood_cleansed"], advertised_price=price,
                description=html.unescape(re.sub(r"<[^>]+>", " ", str(row.get("description", ""))))
                    if pd.notna(row.get("description")) else "")
            if pd.isna(row["host_id"]):
                raise ValueError("Missing host ID")
            accepted.append({**record.model_dump(), "host_id": row["host_id"]})
        except (ValueError, TypeError, ValidationError, OverflowError):
            rejected.append({"source_row": position, "listing_id": row.get("id"),
                             "reason": "invalid price, required field, or input contract"})
    clean = pd.DataFrame(accepted)
    if clean.empty or clean.listing_id.duplicated().any():
        raise ValueError("No usable listings or duplicate listing IDs.")
    split = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42)
    development, test = next(split.split(clean, groups=clean.host_id))
    dev = clean.iloc[development]
    train, validation = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=42)
                             .split(dev, groups=dev.host_id))
    clean["split"] = "test"
    clean.loc[dev.iloc[train].index, "split"] = "train"
    clean.loc[dev.iloc[validation].index, "split"] = "validation"
    groups = {name: set(part.host_id) for name, part in clean.groupby("split")}
    if any(groups[a] & groups[b] for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]):
        raise RuntimeError("Host leakage across partitions.")
    destination.mkdir(parents=True)
    clean.to_csv(destination / "listings.csv", index=False)
    pd.DataFrame(rejected, columns=["source_row", "listing_id", "reason"]).to_csv(destination / "excluded.csv", index=False)
    arrivals.mkdir(parents=True, exist_ok=True)
    held_out = clean[clean.split == "test"].sort_values("listing_id")
    for i, start in enumerate(range(0, len(held_out), 250), 1):
        batch = held_out.iloc[start:start + 250].drop(columns=["host_id", "split"])
        batch.to_csv(arrivals / f"day_{i:03d}.csv", index=False)
    write_json(destination / "manifest.json", {
        "source": SOURCE, "snapshot": "2026-06-15", "currency": "CAD",
        "license": "CC BY 4.0", "attribution": "Inside Airbnb, Montreal",
        "raw_sha256": sha256_file(raw), "prepared_sha256": sha256_file(destination / "listings.csv"),
        "raw_rows": len(source), "accepted_rows": len(clean), "excluded_rows": len(rejected),
        "split_counts": clean.split.value_counts().to_dict(), "group": "host_id", "seed": 42,
        "arrival_note": "Held-out listings; day numbers are simulated arrival batches, not dates."})
    print(json.dumps({"prepared_rows": len(clean), "split_counts": clean.split.value_counts().to_dict()}))
