# Listing-review application

A Python workspace for receiving property listings, estimating advertised nightly prices and requesting structured listing-quality reviews. Scikit-learn runs on CPU; the review chain calls Gemini. Reviews propose wording for a manager and never publish changes or send messages.

For the connected listing walkthrough and six exercises, see the [S10 lab guide](../../docs/s10_listings.md).

## Requirements

Python 3.12, Docker Desktop with Linux containers, and a Gemini API key for AI reviews. Run Python clients locally; run training, the API, ingestion and UI through Docker. Run all commands from the repository root. The price API works without an AI key.

```bash
python -m pip install -r apps/listing_review/requirements.txt
python -m pip install -e . --no-deps
```

## Configuration

Copy [.env.example](.env.example) to `apps/listing_review/.env` and enter `GOOGLE_API_KEY` locally. Preserve an existing file. The default Gemini review model is `gemini-2.5-flash`; change `AI_MODEL` only to a model available to your account. `AI_MODEL` does not select the numerical price model; that model comes from the active fitted bundle. Numerical training and prediction do not need a Gemini key.

Local Python commands use process variables first, the application `.env` second, and the root `.env` as a fallback. Compose uses the explicit `--env-file` below. The API and explicitly launched prompt-engineering job receive the key. Real `.env` files are ignored and excluded from images.

## Prepare and train a candidate

```bash
docker compose --env-file apps/listing_review/.env -f apps/listing_review/deploy/compose.yaml build ml_training
docker compose --env-file apps/listing_review/.env -f apps/listing_review/deploy/compose.yaml run --rm ml_training
```

The job prepares missing inputs from the fixed [Inside Airbnb](https://insideairbnb.com/get-the-data/) Montréal snapshot ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)), then selects a model by validation MAE. Later runs verify and reuse prepared data. Each run saves a new bundle in `local/listing_review/candidates/` and evidence in the repository's `artifacts/s10_training/`. Mounted outputs survive container removal. Normal `up` does not start this job.

### Train again and compare

Run a new candidate with a different Ridge setting:

```bash
docker compose --env-file apps/listing_review/.env -f apps/listing_review/deploy/compose.yaml run --rm ml_training --ridge-alpha 1
```

Default alpha is 10. Each run fits a median baseline, Ridge and gradient boosting, selects by overall validation MAE, and saves the winning fitted pipeline. Alpha only affects Ridge; the winner may stay the same. Compare validation rows in `tables/metrics.csv` and inspect `tables/validation-groups.csv` with group counts. Test results assess the frozen choice and must not drive candidate selection. This is numerical ML training; it does not retrain Gemini or alter active review instructions.

The job prints separate **Candidate** and **Evaluation** paths. Translate `/work/local/candidates/<id>` to `local/listing_review/candidates/<id>` and `/work/artifacts/s10_training/<id>` to `artifacts/s10_training/<id>`. Keep the actual IDs from each printed path. Rebuilding the training image is needed after changing training code/dependencies, not just to pass a different alpha.

## Activate and verify a model

These file-management examples use **PowerShell from the repository root**. Run one step at a time in the same terminal. They do not train a model. Use the candidate produced by your own training run; `model.pkl` contains executable Python serialization.

### Select a candidate

Replace `REPLACE_WITH_CANDIDATE_ID` below with the folder name printed by training. Validate both files before changing the active folder:

```powershell
$ErrorActionPreference = 'Stop'
$s10Candidate = Join-Path (Get-Location) 'local/listing_review/candidates/REPLACE_WITH_CANDIDATE_ID'
$s10Active = Join-Path (Get-Location) 'local/listing_review/model'
$s10Files = @('model.pkl', 'metadata.json')
foreach ($s10File in $s10Files) {
    if (-not (Test-Path -LiteralPath (Join-Path $s10Candidate $s10File) -PathType Leaf)) {
        throw "Missing candidate file: $s10File. Check the candidate folder path."
    }
}
$s10Metadata = Get-Content -LiteralPath (Join-Path $s10Candidate 'metadata.json') -Raw | ConvertFrom-Json
$s10Hash = (Get-FileHash -LiteralPath (Join-Path $s10Candidate 'model.pkl') -Algorithm SHA256).Hash
if ($s10Hash -ne $s10Metadata.model_sha256) { throw 'Candidate files do not match.' }
$s10Metadata | Select-Object version, selected_model, currency
```

**Observe:** the candidate's version and selected model. Keep this version for the verification step. File existence and matching hashes do not replace evaluation of its validation results.

The service commands below use these arguments for the **standard lab deployment**:

```powershell
$s10Compose = @('--env-file', 'apps/listing_review/.env', '-f', 'apps/listing_review/deploy/compose.yaml')
$s10ApiUrl = 'http://127.0.0.1:8000'
```

If working on an existing isolated/custom deployment, use its same `-p PROJECT` and additional `-f OVERRIDE` arguments in `$s10Compose`, its actual API URL, and its mounted active-model path in `$s10Active`. Do not accidentally start the default project beside it. The examples assume direct Python clients also point to that same active bundle and API (`S10_MODEL` and `S10_API_URL` when customized).

### First activation

Use this only when the active folder does not exist and API/ingestion have not been started. Run **Select a candidate** first:

```powershell
if (Test-Path -LiteralPath $s10Active) {
    throw 'An active folder already exists. Keep it, or use Replace an active model.'
}
New-Item -ItemType Directory -Path $s10Active | Out-Null
foreach ($s10File in $s10Files) {
    Copy-Item -LiteralPath (Join-Path $s10Candidate $s10File) -Destination (Join-Path $s10Active $s10File)
}
Get-ChildItem -LiteralPath $s10Active
```

**Observe:** `model.pkl` and `metadata.json` directly inside the active folder. Do not copy the candidate folder inside `model/`. Continue with **Start the API, then the application** below, then **Verify the active model** before processing arrivals.

### Replace an active model

Run **Select a candidate** first. Both API and ingestion keep a fitted model in memory, so both must reload it. Let active AI reviews finish before stopping the API; unfinished reviews are marked interrupted on restart and need an explicit retry.

**1. Stop both model consumers.** Keep the UI and simulator data intact:

```powershell
docker compose @s10Compose stop api data_ingest
if ($LASTEXITCODE -ne 0) { throw 'Could not stop model consumers; do not replace files.' }
```

**2. Back up the active bundle into a new folder.** If either file is missing, stop and investigate before copying the candidate:

```powershell
foreach ($s10File in $s10Files) {
    if (-not (Test-Path -LiteralPath (Join-Path $s10Active $s10File) -PathType Leaf)) {
        throw "Missing active file: $s10File. Restore the existing bundle before replacing it."
    }
}
$s10Backup = Join-Path (Get-Location) ('local/listing_review/model_backups/' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
if (Test-Path -LiteralPath $s10Backup) { throw 'Backup folder already exists; choose a new path.' }
New-Item -ItemType Directory -Path $s10Backup | Out-Null
foreach ($s10File in $s10Files) {
    Copy-Item -LiteralPath (Join-Path $s10Active $s10File) -Destination (Join-Path $s10Backup $s10File)
}
Write-Output "Previous bundle saved at: $s10Backup"
```

Keep the printed backup path for rollback. Candidates and evaluation artifacts remain in their original locations.

**3. Install the candidate files and restart both consumers.** Keep the active directory itself in place:

```powershell
foreach ($s10File in $s10Files) {
    Copy-Item -LiteralPath (Join-Path $s10Candidate $s10File) -Destination (Join-Path $s10Active $s10File) -Force
}
docker compose @s10Compose up -d --force-recreate api data_ingest
if ($LASTEXITCODE -ne 0) { throw 'Startup failed. Inspect logs and use rollback if needed.' }
```

A bundle-only switch needs no image rebuild. If you also changed model-loading code or dependencies, rebuild the affected images before starting them.

### Verify the active model

Wait for `/health/ready` to respond successfully, then compare the loaded version with the active metadata:

```powershell
$s10Expected = Get-Content -LiteralPath (Join-Path $s10Active 'metadata.json') -Raw | ConvertFrom-Json
$s10Ready = Invoke-RestMethod -Uri "$s10ApiUrl/health/ready"
if ($s10Ready.status -ne 'ready' -or $s10Ready.model_version -ne $s10Expected.version) {
    throw 'API did not load the expected model version.'
}
$s10Ready
python -m apps.listing_review_lab.clients.predict
python -m apps.listing_review_lab.clients.predict --api
```

**Observe:** direct and API predictions on the same example have matching estimate, currency, status and model version. Request IDs are expected to differ or be absent from direct inference. For a replacement, the version must also equal the candidate version recorded earlier. Check `docker compose @s10Compose ps` and `docker compose @s10Compose logs --tail 30 api data_ingest` if startup fails.

After ingestion processes a genuinely new batch, open a new listing's **Overview → Estimate details** to confirm its model version. Older saved batch predictions retain their original version and value. Switching models does not re-estimate completed batches or rewrite historical AI reviews. Therefore `clients.predict --listing-id OLD_LISTING_ID` can correctly report a mismatch after a model switch; use the two example commands above to check the current model, or compare a newly processed listing. Do not delete historical state to make old estimates match.

### Roll back to the previous model

Use the backup path printed in replacement step 2. In a new terminal, first restore the deployment variables from **Select a candidate**, then set `$s10Backup` to that exact backup folder. This restores the previous model files; it does not undo batches processed since the switch.

```powershell
foreach ($s10File in $s10Files) {
    if (-not (Test-Path -LiteralPath (Join-Path $s10Backup $s10File) -PathType Leaf)) {
        throw "Missing backup file: $s10File. Check the backup path."
    }
}
docker compose @s10Compose stop api data_ingest
if ($LASTEXITCODE -ne 0) { throw 'Could not stop model consumers; do not restore files.' }
foreach ($s10File in $s10Files) {
    Copy-Item -LiteralPath (Join-Path $s10Backup $s10File) -Destination (Join-Path $s10Active $s10File) -Force
}
docker compose @s10Compose up -d --force-recreate api data_ingest
if ($LASTEXITCODE -ne 0) { throw 'Startup failed. Inspect the service logs.' }
```

Repeat **Verify the active model**. Expect the backup's version, not the candidate's. Keep backups, candidates, evaluation results and application state for comparison.

## Start the API, then the application

```bash
docker compose --env-file apps/listing_review/.env -f apps/listing_review/deploy/compose.yaml up --build -d api
```

Once [readiness](http://localhost:8000/health/ready) returns ready, compare local and HTTP inference:

```bash
python -m apps.listing_review_lab.clients.predict
python -m apps.listing_review_lab.clients.predict --api
```

Both use `local/listing_review/model/`; the API mounts it read-only. Copy the simulator `.env.example` to `.env` in `apps/listings_simulator/`, preserving an existing file. Start the independent upstream simulator, then the remaining review services:

```bash
docker compose --env-file apps/listings_simulator/.env -f apps/listings_simulator/deploy/compose.yaml up --build -d
docker compose --env-file apps/listing_review/.env -f apps/listing_review/deploy/compose.yaml up --build -d
```

Open [Streamlit](http://localhost:8501):

- **Listings:** filter arrivals, inspect saved estimates and request quality reviews.
- **Price scenario:** calculate with edited inputs without changing stored listings.
- **Arrivals:** inspect completed batches and rejected records.

Publish another prepared batch by setting `SIMULATOR_DAY=2` there and rerunning its Compose `up -d` command. Publication makes it available; the ingestion worker processes it on its next poll. Repeating a day does not duplicate completed work.

Read each service in `deploy/compose.yaml` from top to bottom: **image/build** supplies Python and dependencies, **command** selects the process, **environment** supplies configuration, **ports** exposes an HTTP service, and **volumes** connects persistent files. Training and ingestion share the ML image; prompt engineering uses the AI image. The API image includes AI dependencies for background reviews. The UI and API have their own images; the simulator has a separate deployment. Job profiles keep training and prompt comparison out of ordinary startup.

Image recipes and build exclusions live in `deploy/dockerfiles/`. Role-specific dependencies live in `requirements/`; `requirements.txt` combines them for local development. Keep `.env` at the application root and persistent files under repository-root `local/`. The API snapshots active instructions into each review job, and runs the chain in a FastAPI background task.

Set `S10_UPSTREAM_URL` in the application `.env` for another upstream address. Docker Desktop defaults to `http://host.docker.internal:8001`. Ingestion can wait for an unavailable upstream while existing results remain accessible.

The separate [listings simulator](../listings_simulator/README.md) owns prepared arrival examples and published incoming files. Training exports held-out batches to it; the ingestion worker reads its HTTP API.

## Services

| Service | Purpose |
|---|---|
| Application API, port 8000 | Predictions, stored listings and background AI reviews |
| Ingestion worker | Polls arrivals and saves batch predictions |
| Streamlit, port 8501 | API-based manager interface |

[API documentation](http://localhost:8000/docs) describes `POST /predict`, `GET /listings`, `POST /listings/{id}/reviews`, and `GET /jobs/{id}`. `/health/live` checks the process; `/health/ready` checks model and storage availability.

For one arrived listing, `python -m apps.listing_review_lab.clients.predict --listing-id LISTING_ID` compares its saved batch estimate with direct and API inference. It does not change the listing.

## API contracts and client workflow

The API is a small HTTP boundary around the shared prediction, review and storage code. Read the grouped routes and request examples in `/docs`; `/openapi.json` includes success and error schemas. It is intended for localhost and one API process.

| Operation | Response | Meaning |
|---|---|---|
| `POST /predict` | `200` | Immediate CAD estimate, model version and prediction status; nothing is stored |
| `GET /listings/{id}` | `200` or `404` | Saved property facts and its original batch estimate |
| `POST /listings/{id}/reviews` | `202` | Job accepted or previously accepted; body contains `job_id` and `status_url`, and `Location` points to the job |
| `GET /jobs/{id}` | `200` or `404` | Saved job status, snapshot, completed stages and result; reading a failed job is still a successful HTTP read |
| `GET /listings/{id}/reviews` | `200` or `404` | Review history, newest attempt first; an existing listing without reviews returns `[]` |
| `GET /batches/{id}/rejections` | `200` or `404` | Rejected source rows; an existing batch without rejections returns `[]` |

### Request a review, then poll it

1. Choose a new client `request_id`, for example `review-attempt-001`, and submit `{"request_id":"review-attempt-001"}` to the listing's review endpoint.
2. Poll the returned `status_url`. `queued` and `running` mean work is pending; `completed` has a proposal; `failed` has a failure code. Polling never generates another review.
3. If a submission's network response is lost, repeat it with the **same** key and listing. It returns the same job, even when that job has completed or failed. Reusing the key for another listing returns `409`.
4. To explicitly retry a failed review or generate a new one, use a **new** key. New jobs snapshot current inputs and instructions; old jobs remain unchanged.

The client `request_id` prevents duplicate work. The server-generated `X-Request-ID` identifies one HTTP call for troubleshooting. They serve different purposes: repeated submissions reuse the client key but each HTTP call gets a new tracing ID. A successful prediction also includes that tracing ID in its body.

Background execution uses FastAPI `BackgroundTasks`, not a durable queue. One API process claims each queued job once. After a restart, unfinished jobs become `failed` with `interrupted`; retry explicitly with a new key. Completed results remain readable. The response schemas validate the job envelope; saved AI snapshots, stages and results retain their historical version instead of being forced into the newest proposal schema.

### Read collections in pages

`/listings`, `/batches`, `/batches/{id}/rejections` and `/listings/{id}/reviews` return arrays with `offset=0` and `limit=50` by default. `limit` must be 1–250 and `offset` nonnegative. Request the next page by increasing offset by the number received; a short or empty page ends traversal. For example:

```text
GET /batches?offset=0&limit=50
GET /batches?offset=50&limit=50
```

Batches sort by ID; rejected rows by source row number; review history by newest attempt; listings by batch ID then listing ID. `/listings/workspace` additionally supplies a matching `total`, full-inventory counts and filter options, sorting newest batch first. Offset pages are not a frozen snapshot: refresh from the start when new arrivals/reviews change a collection during traversal. The UI follows bounded pages for counts and history rather than silently showing only the first page.

### Handle errors consistently

All handled API errors use this JSON structure and include `X-Request-ID`:

```json
{
  "code": "validation_error",
  "message": "Check the indicated fields.",
  "request_id": "server-generated-reference",
  "details": [{"field": "body.accommodates", "type": "greater_than_equal"}]
}
```

Use the HTTP status and stable `code`, not message wording, for client decisions: `404/not_found`, `405/method_not_allowed`, `409/conflict`, `422/validation_error`, `503/unavailable`, or `500/internal_error`. `details` is an empty array except for input-validation failures. Submitted values, provider responses and internal exception messages are not echoed. Unexpected errors are logged and returned as a generic JSON `500`; they are not converted into successful results. The tracing ID connects the response to the logs.

These contracts improve integration and debugging; they do not establish prediction accuracy or factual correctness of an AI proposal. Continue inspecting model errors and source evidence.

## Develop and compare prompts

Use the [student exercise](../../docs/s10_listings.md#4-improve-the-listing-quality-review) for the learning sequence. From the repository root, edit only `apps/listing_review/jobs/prompt_eng/templates/candidate_examples.txt` for the exercise, then rebuild and run:

```bash
docker compose --env-file apps/listing_review/.env -f apps/listing_review/deploy/compose.yaml build prompt_eng
docker compose --env-file apps/listing_review/.env -f apps/listing_review/deploy/compose.yaml run --rm prompt_eng
```

Each stage's instructions are assembled as shared `policy.txt` + variant examples + its stage file (`evidence.txt`, `assessment.txt`, `proposal.txt`). Starting uses `examples.txt`; candidate uses `candidate_examples.txt`. Changing a shared policy/stage file affects both variants, so keep those files unchanged in this exercise. The experiment compares repository templates; it does not load the current active prompt folder as its baseline. Templates live inside the image and require a rebuild after edits.

Both variants run the same three-stage chain on the same two fixed cases and generation settings. Completed comparisons select candidate only with more action-check passes and no regression; otherwise they select starting. Incomplete execution exports no selected prompts. Exact instructions, stage results and failures are retained. Inspect the wording and its factual support, not just the action-check score.

| Location on your computer | Role |
|---|---|
| `apps/listing_review/jobs/prompt_eng/templates/` | Editable experiment source; not read by live API reviews |
| `artifacts/s10_prompt_eng/<run-id>/` | `tables/comparison.csv`, `selection.json` and per-case JSON evidence |
| `local/listing_review/prompt_candidates/<run-id>/` | Three fully assembled instruction files exported after a complete comparison |
| `local/listing_review/prompts/` | Active files read when the API accepts a new review job |

The job prints **Evaluation** and **Selected prompts** paths. Translate `/work/local/` to `local/listing_review/` and `/work/artifacts/` to `artifacts/`. A selected folder may contain the starting variant: read `selection.json`. No output folder is activated automatically. This changes instructions, not Gemini weights or the numerical fitted bundle.

### Activate selected prompts

For the standard deployment, follow these steps from the repository root. For a custom deployment, use its existing Compose project/override arguments and actual mounted prompt directory (`S10_PROMPTS` inside the API), rather than starting a different project.

1. Choose the printed selected folder from a complete experiment. Check that it contains nonempty `evidence.txt`, `assessment.txt` and `proposal.txt`. Inspect those assembled files and their saved comparison evidence.
2. Let queued/running AI reviews finish, then stop the API with `docker compose --env-file apps/listing_review/.env -f apps/listing_review/deploy/compose.yaml stop api`. Stopping it interrupts unfinished background reviews. Numerical ingestion is independent and need not stop for a prompt-only change.
3. If `local/listing_review/prompts/` already has active files, copy all three into a **new** dated folder under `local/listing_review/prompt_backups/`. Keep the backup path. On a first installation, create the active `prompts/` folder.
4. Copy all three files from the **selected output folder** directly into `local/listing_review/prompts/`, replacing the active files together. Keep the mounted directory in place; do not nest the selected folder and do not copy only `candidate_examples.txt`.
5. Start the API with `docker compose --env-file apps/listing_review/.env -f apps/listing_review/deploy/compose.yaml start api`. If the API was never created, follow the earlier API startup instructions instead. A prompt-file-only change does not require rebuilding the API image.
6. Request one new review in Streamlit. Under **Technical details**, obtain its job ID, then inspect `/jobs/JOB_ID`. Compare `snapshot.instructions.evidence`, `.assessment` and `.proposal` with the exact contents of the active files. Check the completed recommendation and evidence as well. API readiness alone does not verify prompt availability.

Existing review jobs retain their snapshotted inputs, instructions and provider settings; activation does not change their results. Refreshing an old result does not generate a new review. If active instructions are missing or invalid, new review requests fail; numerical prediction remains available. The Gemini key/model settings must also be valid for a live review to complete.

**Rollback:** wait for current reviews to finish, stop the same API, copy the three backed-up files into the active folder, and start the API again. Verify a new job's instruction snapshot against the restored files. Rollback affects future requests; it does not erase reviews produced with the candidate. Keep unsuccessful experiment evidence and historical jobs.

## Operation and troubleshooting

```bash
docker compose --env-file apps/listing_review/.env -f apps/listing_review/deploy/compose.yaml ps
docker compose --env-file apps/listing_review/.env -f apps/listing_review/deploy/compose.yaml logs --tail 30 data_ingest api
docker compose --env-file apps/listing_review/.env -f apps/listing_review/deploy/compose.yaml down
```

- **API not ready:** confirm preparation/training produced `local/listing_review/model/` and inspect API logs.
- **No new listings:** confirm the day was published and the ingestion worker is running.
- **Review failed:** inspect its job status and API logs using the job ID. Check provider configuration or quota for provider failures, then explicitly request a retry.
- **Ports occupied:** stop any other process using ports 8000, 8001 or 8501.

SQLite state persists in the bind-mounted `local/listing_review/state/` folder. Removing containers does not remove this folder. Use one ingestion worker and one API process. Ports bind to localhost; the application has no authentication layer.

Reviews use FastAPI `BackgroundTasks`: the response returns a job ID before the chain runs. Tasks from separate requests may run concurrently. Saved results survive restarts, but pending execution does not; API startup marks unfinished jobs interrupted for an explicit retry. Keep one API process for this lab's startup recovery. After upgrading from the separate-worker version, remove its stopped container with `docker compose --env-file apps/listing_review/.env -f apps/listing_review/deploy/compose.yaml up --build -d --remove-orphans`.

## Code and checks

| Component | Entry point | Supporting code |
|---|---|---|
| API | [api/app.py](api/app.py) | [api/helpers/](api/helpers/) |
| UI | [ui/app.py](ui/app.py) | [ui/helpers/](ui/helpers/) |
| Training | [jobs/ml_training/__main__.py](jobs/ml_training/__main__.py) | [jobs/ml_training/helpers/](jobs/ml_training/helpers/) |
| Ingestion | [jobs/data_ingest/__main__.py](jobs/data_ingest/__main__.py) | [jobs/data_ingest/helpers/](jobs/data_ingest/helpers/) |
| Prompt engineering | [jobs/prompt_eng/__main__.py](jobs/prompt_eng/__main__.py) | [jobs/prompt_eng/helpers/](jobs/prompt_eng/helpers/) |

The training command is `python -m apps.listing_review.jobs.ml_training` (also used by Compose). Preparation, context creation and fitting helpers are callable functions, not separate command-line entry points.

Ingestion uses `python -m apps.listing_review.jobs.data_ingest`; `--once` processes currently available arrivals and exits. Its entry point owns configuration and the polling loop; `helpers/batches.py` handles CSV normalization and batch processing. The `api/`, `ui/` and `jobs/` trees use Python namespace packages without empty `__init__.py` files. Job `__main__.py` files remain the executable entry points.

`api/app.py` handles HTTP; `jobs/` orchestrates background work; `ui/` renders the workspace using HTTP only. Shared prediction, review and runtime code supports the API and jobs, and never imports their entry points. SQL stays in the concrete SQLite store; evidence checks stay beside the review chain.

The shared implementation lives in [lab_helpers.s10_shared](../../src/lab_helpers/s10_shared/), installed through the existing course package. `ml_lib` and `ai_lib` provide general capabilities; `s10_shared` supplies listing-specific prediction, review and persistence logic; `apps/` contains executable entry points. Each image installs its own copy, so rebuild affected images after shared-code changes.

Run native commands from the repository root. When launching elsewhere, set the process environment variable `S10_ROOT` to that root before starting Python. Containers use `/work`. Existing data, model, prompt and database overrides still apply. The editable example stays in `resources/example_listing.json`.

Each component keeps its visible entry point at the folder root and its supporting Python modules in `helpers/`. The UI dispatches to listing, review, scenario and arrival helpers. Training and prompt engineering call `helpers/workflow.py`; fitting and selection remain visible there. API routes and lifecycle remain in `api/app.py`, with request logging and validation responses in `api/helpers/http.py`. Editable prompt text stays in the job's `templates/` folder. [Lab tools](../listing_review_lab/README.md) contain clients and automated tests. The [lab guide](../../docs/s10_listings.md) explains the Docker job and serving sequence.

General partition validation and aligned regression group arithmetic live in `ml_lib`. `ai_lib.evaluation.reports.paired_outcomes` counts matched successes, improvements and regressions without choosing a winner. `lab_helpers.s10_shared.evaluation` defines the listing groups and their interpretation; the jobs retain the model and prompt replacement rules.

```bash
python -m unittest discover -s apps/listing_review_lab/tests -v
```
