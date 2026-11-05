# Listing-review lab tools

Follow the [S10 guide](../../docs/s10_listings.md) for the six exercises. This folder contains clients and automated checks; the [application](../listing_review/README.md) does not depend on it. Run commands from the repository root in the course Python environment.

## Student experiments

**Compare predictions.** Direct inference needs the deployed model in `local/listing_review/model/`. API inference needs the Docker API running. Both commands use the same [example listing](../listing_review/resources/example_listing.json):

```bash
python -m apps.listing_review_lab.clients.predict
python -m apps.listing_review_lab.clients.predict --api
```

After ingesting an arrival batch, compare a listing's saved batch estimate with direct and API predictions. Replace `LISTING_ID` with its ID from the interface:

```bash
python -m apps.listing_review_lab.clients.predict --listing-id LISTING_ID
```

The comparison requires the same deployed model locally and in the API. A mismatch stops the command; inspect inputs and model identity before continuing.

**Measure concurrency.** Keep the API running and use the same request count and fixed example for both measurements:

```bash
python -m apps.listing_review_lab.clients.load --requests 100 --concurrency 1
python -m apps.listing_review_lab.clients.load --requests 100 --concurrency 4
```

Compare throughput, p95 latency and failures. The client also checks response structure and prediction consistency. More concurrency need not improve latency.

## Maintainer checks

These tests are separate from the student walkthrough. Install the application's [development requirements](../listing_review/requirements.txt) in the course environment, then run:

```bash
python -m unittest discover -s apps/listing_review_lab/tests -v
```

Tests use temporary state, small model fixtures and mocked cloud responses. They cover prediction agreement, duplicate-safe arrivals, saved review instructions, legacy jobs, prompt selection and UI interactions. They require no running Docker services or live provider calls and do not certify container deployment or live answer quality.

General library tests remain in the repository's `tests/` folder. Training and prompt engineering remain operational jobs in the application.
