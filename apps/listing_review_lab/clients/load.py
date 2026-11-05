"""Measure the complete fixed-input workload, including invalid results and failures."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import math
from time import perf_counter
import httpx
import numpy as np
from pydantic import ValidationError
from lab_helpers.s10_shared.examples import load_example_listing
from lab_helpers.s10_shared.prediction.schemas import PredictionResponse
from lab_helpers.s10_shared.runtime.settings import settings


def request_prediction(client, example):
    started = perf_counter()
    try:
        response = client.post("/predict", json=example)
    except httpx.HTTPError as error:
        return perf_counter() - started, "network:" + type(error).__name__, None
    if response.status_code != 200:
        return perf_counter() - started, f"http:{response.status_code}", None
    try:
        result = PredictionResponse.model_validate(response.json())
    except (ValidationError, ValueError):
        return perf_counter() - started, "invalid_response", None
    return perf_counter() - started, "valid", result


def measure(client, requests, concurrency):
    example = load_example_listing()
    # A reference call establishes consistency, not model accuracy. It is reported separately.
    _, reference_outcome, reference = request_prediction(client, example)
    started = perf_counter()
    with ThreadPoolExecutor(concurrency) as pool:
        results = list(pool.map(lambda _: request_prediction(client, example), range(requests)))
    elapsed = perf_counter() - started
    durations, failures = [], Counter()
    for seconds, outcome, result in results:
        if outcome != "valid":
            failures[outcome] += 1
        elif reference is None:
            failures["reference_unavailable"] += 1
        elif result.model_version != reference.model_version:
            failures["model_mismatch"] += 1
        elif (not math.isclose(result.estimate, reference.estimate, rel_tol=1e-9, abs_tol=1e-6)
              or result.status != reference.status or result.currency != reference.currency):
            failures["prediction_mismatch"] += 1
        else:
            durations.append(seconds)
    return {"requests": len(results), "concurrency": concurrency,
            "reference_call": {"outcome": reference_outcome,
                               "model_version": reference.model_version if reference else None},
            "successes": len(durations), "failures": sum(failures.values()),
            "failure_counts": dict(failures), "elapsed_seconds": elapsed,
            "successful_requests_per_second": len(durations) / elapsed,
            "successful_latency_seconds": dict(zip(["p50", "p95", "p99"],
                np.quantile(durations, [.5, .95, .99]))) if durations else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.concurrency <= 64 or not 1 <= args.requests <= 10000:
        parser.error("Use 1..64 clients and 1..10000 requests.")
    with httpx.Client(base_url=settings().api_url, timeout=15) as client:
        print(json.dumps(measure(client, args.requests, args.concurrency), indent=2))


if __name__ == "__main__":
    main()
