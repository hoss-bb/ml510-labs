# Listings simulator

An independent HTTP source of incoming listing batches. Python code lives in `simulator/`; deployment files live in `deploy/`. It uses no models, credentials or consumer-application code.

## Code map

- `simulator/api.py`: HTTP entry point and batch routes.
- `simulator/__main__.py`: publication command and argument handling.
- `simulator/helpers/publication.py`: single-day and periodic publication.
- `simulator/helpers/settings.py`: data paths and publication configuration.

The entry points use `helpers/`; helpers do not import the entry points. Docker copies the complete `simulator/` package, including its helpers.

## Configure and start

Copy `.env.example` to `.env` in this application folder, preserving an existing file:

```dotenv
SIMULATOR_MODE=once
SIMULATOR_DAY=1
SIMULATOR_INTERVAL_SECONDS=60
SIMULATOR_PORT=8001
SIMULATOR_BIND_ADDRESS=127.0.0.1
```

From the repository root:

```bash
cd apps/listings_simulator
docker compose --env-file .env -f deploy/compose.yaml up --build -d
docker compose --env-file .env -f deploy/compose.yaml logs publish
```

This starts the HTTP server and publishes day 1. The publisher then exits with code zero; that is normal. The server stays available.

For another day, change `SIMULATOR_DAY` and run `docker compose --env-file .env -f deploy/compose.yaml up -d`. Use `up`, not `restart`, to apply changed settings. Repeating an unchanged day is safe.

For timed arrivals, set `SIMULATOR_MODE=periodic` and the interval in seconds, then run the same command. The first pending day is published immediately. Remaining prepared days follow in order; the publisher exits when exhausted. Restarting skips unchanged published days and rejects changed contents under an existing ID.

## Data and API

Place `day_NNN.csv` files in repository-root `local/listings_simulator/arrival_examples/`. Publication copies a complete file atomically into `incoming/`. The lab training job can supply example files, but any producer may supply them. Publication does not mean a consumer has processed the batch.

- `GET /batches`: IDs and checksums.
- `GET /batches/{batch_id}`: original CSV contents.
- `GET /health/live`: process health.

Consumers configure their own URL and polling schedule. Docker Desktop consumers use `http://host.docker.internal:8001` locally. Bind address and port are configurable for other deployments; choose the interface deliberately.

## Native commands

From this application folder, with requirements already installed:

```bash
python -m simulator --day 3
python -m simulator --interval 60
```

Explicit arguments override publication environment settings. Without arguments, the command reads process environment variables, defaulting to day 1. Compose loads `.env`; native Python does not automatically load it. `LISTINGS_SIMULATOR_DATA` overrides the native data directory.

```bash
docker compose --env-file .env -f deploy/compose.yaml down
```

Stopping containers preserves published files. See the [S10 guide](../../docs/s10_listings.md) for the separate consuming application.
