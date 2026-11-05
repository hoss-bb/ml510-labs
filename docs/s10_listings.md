# S10 - From a trained model to a working application

[All labs](../README.md)

## Context and purpose

A manager receives new property listings. How can we provide price estimates, process new arrivals and prepare useful listing-quality reviews?

Two applications have distinct roles: `apps/listings_simulator/` represents the external platform; `apps/listing_review/` predicts prices and prepares reviews. Each application has its own Compose setup.

| Operation | What happens | When it runs |
|---|---|---|
| Training | Produces a candidate price model for inspection | A one-off job you launch |
| Online prediction | Returns one price estimate through HTTP | On each API request |
| Batch inference | Ingestion fetches published listings, predicts and saves results | When a new batch is available |
| AI review | Extracts facts, assesses them and proposes an action | A background task requested for a listing |

## How to use this guide

Follow the six sections in order. Each separates **Context**, **Tasks**, **Check** and **Explain**. Keep the listing ID selected in Exercise 3 for Exercise 5. An open challenge after Exercise 6 asks you to investigate quality limits and test an ML or AI improvement.

**Working directory:** run every command below from the repository root, `ml510-labs/`. Keep that directory throughout the walkthrough. Run Python clients in your activated course environment; Docker runs the services and one-off jobs.

**Run a training job, then serve the selected model with Docker.** Follow the phases below from the repository root. You will change one thing at a time and inspect its effect. There are no notebooks or submissions.

## 1. Prepare and choose a model

### Context

Training teaches the numerical price model to estimate an advertised nightly price from property facts. It is separate from Gemini reviews: no cloud-model weights are trained here, and Streamlit has no training button.

| Step | What you do | What changes |
|---|---|---|
| Train | Run the one-off Docker job | A new candidate bundle and evaluation results are saved |
| Compare | Inspect validation errors | You decide whether the candidate is worth deploying |
| Activate | Explicitly install the chosen bundle and start/restart its consumers | Future predictions use that model |

A **candidate** is a fitted model you can evaluate. The **active model** is the bundle the API and ingestion service load. Training never activates a candidate automatically.

### Tasks — configure and train

**1. Prepare your environment.**

Use Python 3.12 and Docker Desktop with Linux containers. Activate your course environment. For a first installation, use the [environment instructions](s4_regression.md#1-prepare-the-environment), then install the S10 requirements:

```bash
python -m pip install -r apps/listing_review/requirements.txt
python -m pip install -e . --no-deps
python -m pip check
```

**2. Configure the application.** Copy `apps/listing_review/.env.example` to `.env` in the same folder and enter your Gemini key locally. Keep an existing `.env`. The key is used later for AI reviews and prompt comparison; numerical training and price prediction do not need it. The [application README](../apps/listing_review/README.md#configuration) explains configuration.

**3. Train the first candidate.** These commands work before the API or UI has been started. Build the training image, then run the one-off job. Keep its printed **Candidate** and **Evaluation** paths:

```bash
docker compose --env-file apps/listing_review/.env -f apps/listing_review/deploy/compose.yaml build ml_training
docker compose --env-file apps/listing_review/.env -f apps/listing_review/deploy/compose.yaml run --rm ml_training
```

### Check — inspect the outputs

The job prepares missing data and AI context, exports held-out simulator batches, then trains. Later runs verify and reuse that preparation. Invalid or incomplete data stops the job rather than being silently overwritten.

Mounted folders keep the outputs after the container exits:

| Output on your computer | What to inspect |
|---|---|
| `local/listing_review/candidates/<candidate-id>/` | `model.pkl` (fitted preprocessing + model) and `metadata.json` (version, selected model and metrics) |
| `artifacts/s10_training/<evaluation-id>/tables/metrics.csv` | Training/validation errors and the frozen test assessment |
| `artifacts/s10_training/<evaluation-id>/tables/validation-groups.csv` | Validation performance for room-type and capacity groups |
| `local/listings_simulator/arrival_examples/` | Held-out listings exported as simulated arrival batches |

Printed container paths starting `/work/local/` correspond to `local/listing_review/` on your computer; `/work/artifacts/` corresponds to `artifacts/`. Candidate and evaluation IDs need not be identical—use both printed paths. Ordinary application startup does not launch training.

The job compares a median-price baseline, Ridge regression and gradient boosting. **MAE** is the average absolute error in CAD per night; lower is better. Overall validation MAE chooses the saved model. Group results help you understand limitations; they do not select different models for different groups.

**Judge usefulness, not just the winner.** Compare the winner with the median baseline, read the absolute error in CAD, and check group sizes and nonpositive predictions in the evaluation tables. Winning validation does not make every estimate accurate. A large advertised-versus-estimated gap is a question to investigate, not evidence that a listing is overpriced. Sparse groups need more evidence; scenario changes describe model behaviour, not guaranteed market effects.

The fixed Montreal snapshot comes from [Inside Airbnb](https://insideairbnb.com/get-the-data/) under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Prices are advertised nightly prices in CAD, not optimal prices or achieved revenue. See the [data assumptions](https://insideairbnb.com/data-assumptions/).

Inspect [jobs/ml_training/helpers/train.py](../apps/listing_review/jobs/ml_training/helpers/train.py): hosts are separated across splits, preprocessing fits on training only, validation MAE selects the model, then the frozen winner and median reference are assessed on test. The job never replaces `local/listing_review/model/`, the bundle used for serving.

### Exercise 1 - Choose what to deploy

**Task — train another candidate:** run the same job with Ridge alpha changed from 10 to 1. Alpha controls how strongly Ridge penalizes large coefficients. This changes one training setting, not the dataset:


```bash
docker compose --env-file apps/listing_review/.env -f apps/listing_review/deploy/compose.yaml run --rm ml_training --ridge-alpha 1
```

**Check:** a new candidate folder is created; existing candidates and the active bundle remain untouched. Compare the validation rows in each run's `tables/metrics.csv`, then the private-room row in `tables/validation-groups.csv`. Use group counts alongside their errors. Changing Ridge alpha does not guarantee a different winner: the job still compares all three models.

Use validation evidence to choose between candidates. Test results describe the already-selected model; do not use them to keep tuning or choose a run. Reusing the same test set is not new independent confirmation.

**Explain:** would you keep or reject the candidate, and why? Why should the test results not determine your selection?

### Before continuing — choose the correct activation path

- **First installation, no active model:** follow [First activation](../apps/listing_review/README.md#first-activation) to copy your chosen candidate's two files into `local/listing_review/model/`. Then continue with Section 2 to start the API.
- **An active model already exists:** keep it for the main walkthrough. Training a candidate does not require replacing it.
- **You deliberately want to switch models later:** follow [Replace an active model](../apps/listing_review/README.md#replace-an-active-model). It covers stopping the API and ingestion, backing up the old bundle, installing the candidate, restarting and checking the version. [Rollback](../apps/listing_review/README.md#roll-back-to-the-previous-model) restores the saved bundle if needed.

**Check:** the active folder contains `model.pkl` and `metadata.json` directly, not another nested candidate folder. Keep these files together; the loader checks that they match. A new bundle requires the API and ingestion to reload it, but does not require rebuilding images when code and dependencies are unchanged.

**What changes after activation?** New online/scenario predictions and newly processed batches use the new model. Existing saved batch estimates and AI reviews remain unchanged. Their old model versions explain why an older listing's saved estimate may differ from a new calculation. Completed batches are not reprocessed automatically.

## 2. Serve the same prediction

### Context

Does moving a prediction behind HTTP change its result? Compose runs the same image with a different Python command for each service. Environment settings connect the services, ports expose the API and UI, and mounted folders or volumes keep data outside the containers.

### Tasks — start the API and compare predictions

**1. Start only the API:**

```bash
docker compose --env-file apps/listing_review/.env -f apps/listing_review/deploy/compose.yaml up --build -d api
```

**2. Wait for [readiness](http://localhost:8000/health/ready) to report `ready`.** Then run both clients:

```bash
python -m apps.listing_review_lab.clients.predict
python -m apps.listing_review_lab.clients.predict --api
```

### Check

The estimate, currency, status and model version should agree. Both use the same fitted bundle: Python loads it locally; the API loads its read-only Docker mount. This is **online prediction**: one request returns an estimate immediately. Explore the [API documentation](http://localhost:8000/docs). The grouped routes document response fields and errors. See [API contracts and client workflow](../apps/listing_review/README.md#api-contracts-and-client-workflow) for pagination, polling and the difference between a review request key and an HTTP tracing ID.

### Exercise 2 - Follow one prediction

**Task:** open the [API documentation](http://localhost:8000/docs), expand `POST /predict` and choose **Try it out**. Submit the [example listing](../apps/listing_review/resources/example_listing.json) with `accommodates` changed to `0`.

**Check:** expect HTTP **422** with input-validation details rather than a price estimate.

**Explain:** why is this an input error? What changed when the valid prediction moved behind HTTP?

## 3. Receive new listings

### Context

**Publication** makes a CSV file available through the simulator's API. **Ingestion** is a separate process: the review application fetches that file, predicts in chunks and saves results. This is **batch inference**. Simulated days describe arrival order, not original publication dates.

### Tasks — publish day 1, then receive it

**1. Configure the simulator.** Copy the simulator's `.env.example` to `.env` in `apps/listings_simulator/`, preserving an existing file. Set `SIMULATOR_MODE=once` and `SIMULATOR_DAY=1`.

**2. Start the simulator and inspect publication.** Stay at the repository root:

```bash
docker compose --env-file apps/listings_simulator/.env -f apps/listings_simulator/deploy/compose.yaml up --build -d
docker compose --env-file apps/listings_simulator/.env -f apps/listings_simulator/deploy/compose.yaml logs publish
```

**3. Start the review application's remaining services.**

```bash
docker compose --env-file apps/listing_review/.env -f apps/listing_review/deploy/compose.yaml up --build -d
```

**4. Open [Streamlit](http://localhost:8501) and select Arrivals.** Refresh until the batch has been processed.

### Check

The publisher exits successfully after publishing the selected day; the simulator API keeps running. Review ingestion connects to `S10_UPSTREAM_URL`, defaulting to `http://host.docker.internal:8001` in Docker Desktop.

Inspect the batch status and accepted/rejected counts. A published batch may not be completed until ingestion has fetched and processed it.

**Optional extension — timed arrivals:** set `SIMULATOR_MODE=periodic` and `SIMULATOR_INTERVAL_SECONDS=60` in the simulator `.env`. From the repository root, apply it with `docker compose --env-file apps/listings_simulator/.env -f apps/listings_simulator/deploy/compose.yaml up -d`. The first pending day appears immediately; publication stops when exhausted. Use `up`, not `restart`, after changing `.env`.

### Exercise 3 - Receive another day

**Task 1 — publish and repeat:** predict what repeating a day will do. Set `SIMULATOR_DAY=2` with `SIMULATOR_MODE=once` in the simulator `.env`. From the repository root, publish it and repeat the same command:

```bash
docker compose --env-file apps/listings_simulator/.env -f apps/listings_simulator/deploy/compose.yaml up -d
```

**Check:** in Arrivals, check that the source batch's record count equals accepted + rejected and that listing counts do not grow twice. Restart ingestion and inspect again:

```bash
docker compose --env-file apps/listing_review/.env -f apps/listing_review/deploy/compose.yaml restart data_ingest
```

**Explain:** what makes completed work survive a restart? If days 1 and 2 were already processed, unchanged counts are expected. Inspect [ingestion entry point](../apps/listing_review/jobs/data_ingest/__main__.py) and its [batch helpers](../apps/listing_review/jobs/data_ingest/helpers/batches.py); storage internals are supporting code.

**Task 2 — keep a listing for the next exercises:** in **Listings**, filter to the arrival batch and select one listing. Keep its ID for the remaining walkthrough. Compare its saved batch estimate with direct and API inference:

```bash
python -m apps.listing_review_lab.clients.predict --listing-id LISTING_ID
```

**Check:** replace `LISTING_ID` with that listing's ID before running the command. The estimates and model versions should match. No stored record is changed.

## 4. Improve the listing-quality review

### Context

Prompt engineering means improving the instructions and examples given to Gemini. It changes how the review is requested; it does not train Gemini's weights or change the numerical price model.

The [review chain](../src/lab_helpers/s10_shared/reviews/chain.py) makes three successive structured calls:

| Stage | What it does | What the next stage receives |
|---|---|---|
| Evidence | Extracts facts and exact quotations from the supplied listing/context | Original source data plus validated evidence |
| Assessment | Checks the facts for conflicts or conditions worth clarifying | Original source data, evidence and assessment |
| Proposal | Chooses one permitted action and prepares any useful draft | The final recommendation displayed to the manager |

Each stage is validated before continuing. [Validation](../src/lab_helpers/s10_shared/reviews/validation.py) checks quotations, references and output consistency; the manager still checks whether the recommendation is useful and factually justified.

**The development pipeline:** edit candidate examples → rebuild the experiment image → compare starting/candidate chains → inspect evidence and selection → explicitly activate the selected files → request a new review.

The four possible outcomes are: confirm conflicting facts, clarify an important condition, prepare a guest-information note, or indicate no change. A price difference alone does not justify an action. The manager reviews drafts; nothing is published or sent automatically.

### Exercise 4 - Improve one AI decision

**Task 1 — understand the starting point.** Read [examples.txt](../apps/listing_review/jobs/prompt_eng/templates/examples.txt) and [candidate_examples.txt](../apps/listing_review/jobs/prompt_eng/templates/candidate_examples.txt). Both variants combine the same `policy.txt` and stage-specific instruction files with their own examples. The candidate examples are included in all three stages. “Starting” means the repository's examples, not whichever prompts happen to be deployed.

**Task 2 — make one focused change.** Edit only `candidate_examples.txt`. Write down the behaviour you expect to improve—for example, distinguishing a genuine minimum-stay contradiction from two compatible statements. Keep examples grounded in supplied facts, preserve the four permitted actions and avoid inventing details. Leave shared policy/stage files and evaluation cases unchanged so the comparison isolates your change.

**Task 3 — rebuild and compare.** Templates are copied into the experiment image, so rebuild after editing. From the repository root, run the one-off job. It uses the local Gemini credentials and makes real provider calls:

```bash
docker compose --env-file apps/listing_review/.env -f apps/listing_review/deploy/compose.yaml build prompt_eng
docker compose --env-file apps/listing_review/.env -f apps/listing_review/deploy/compose.yaml run --rm prompt_eng
```

### Check — inspect the comparison

**Task 4 — inspect the results before choosing to use them.** The job runs both variants on the same two constructed cases with the same generation settings: four review-chain executions, normally three stage calls each if all stages succeed. This is a small development comparison, not a broad evaluation.

Both cases supply a structured minimum stay of 14 nights. “More than 14 nights” is expected to produce `confirm_facts`; “at least 14 nights” accepts `no_change` or `guest_note`.

| Result | What the job exports |
|---|---|
| Candidate passes more cases, with no previously passing case becoming a failure | Candidate instructions |
| Tie, no improvement, or any regression | Starting instructions |
| Provider, parsing or validation failure leaves execution incomplete | No selected bundle; failures remain saved |

Open `tables/comparison.csv` in the evaluation folder to see each case/action/error, then `selection.json` for the decision. Inspect `starting-conflict.json`, `starting-legitimate.json`, `candidate-conflict.json` and `candidate-legitimate.json` for exact inputs, instructions, saved stage outputs, drafts and failures. An unexpected early error may leave only the cases reached before it stopped. Verify the quotations and recommended action yourself: a matching action code alone is not proof of a good review.

If a change fails to improve the comparison, understand the failure before making another focused edit. Preserve each run's evidence; do not rerun unchanged prompts just to obtain a favourable outcome.

The job prints **Evaluation** and **Selected prompts** paths. Evaluation evidence is under `artifacts/s10_prompt_eng/`; selected instruction files are under `local/listing_review/prompt_candidates/`. Container paths beginning `/work/local/` correspond to `local/listing_review/` on your computer. The selected folder contains only `evidence.txt`, `assessment.txt` and `proposal.txt`. This changes instructions, not cloud-model weights.

### Task 5 — activate the selected instructions for Exercise 5

If the comparison reports incomplete execution, inspect its saved error and resolve it before continuing; it deliberately exports no selected bundle. Price prediction remains available. If it selected **starting**, the exported folder contains starting instructions—not your rejected candidate. Editing templates or running the experiment has not changed the running app.

Use the printed **Selected prompts** folder only after inspecting its results. The [prompt activation reference](../apps/listing_review/README.md#activate-selected-prompts) explains backup, activation, verification and rollback. For the standard deployment:

1. Stop the API:

   ```bash
   docker compose --env-file apps/listing_review/.env -f apps/listing_review/deploy/compose.yaml stop api
   ```

2. Preserve any existing files in `local/listing_review/prompts/`. On a first installation, create this folder.
3. Copy the selected `evidence.txt`, `assessment.txt` and `proposal.txt` directly into it. Do not nest their candidate folder.
4. Start the API again:

   ```bash
   docker compose --env-file apps/listing_review/.env -f apps/listing_review/deploy/compose.yaml start api
   ```

**Check — use the new instructions:** in Exercise 5, request a **new** review (or **Generate another review** for a listing with a completed review). Refreshing a historical result does not rerun it. Open the new job at `/jobs/JOB_ID`: its `snapshot.instructions` contains `evidence`, `assessment` and `proposal`. Compare those texts with the three active files to confirm which instructions were used.

Each job saves its actual instructions with its inputs and uses that snapshot throughout the chain. Later activation does not rewrite existing jobs or results. Wait for active reviews to finish before stopping the API; interrupted work requires an explicit retry. Older queued jobs without saved instructions also require a new attempt.

### Explain

Did the candidate improve a case without making another worse? Are its quotations and drafts supported by the inputs? Why do two passing cases not establish general quality?

## 5. Follow the work through the application

### Context

AI review is **asynchronous**: the API creates a job, FastAPI runs the cloud chain in a background task, and Streamlit follows its status. Numerical estimates remain available while it runs.

The task runs inside the API process, not a separate worker. Status and results are stored in SQLite, but background execution does not survive an API interruption. On restart, unfinished jobs are marked failed; use **Retry review** for a new attempt. This lab runs one API process; separate requests can run background reviews concurrently.

### Exercise 5 - Follow a listing

**Tasks:** use the listing ID kept in Exercise 3.

1. Return to the same listing used in the batch/API comparison. Use the arrow button beside its listing reference to open the property workspace. In **Overview**, inspect its saved estimate and model version under **Estimate details**. **Back to listings** restores your filters and page.
2. Open **Quality review**, choose **Generate review** and follow its status. If a completed review already exists, inspect it before deciding whether another is needed. Short jobs may finish before every intermediate state appears.
3. Open **Technical details** for the job ID and stage timings. Look up that ID at `http://localhost:8000/jobs/JOB_ID` and in the API logs:

   ```bash
   docker compose --env-file apps/listing_review/.env -f apps/listing_review/deploy/compose.yaml logs api
   ```

4. Read the recommendation and supporting evidence. Compare the quotations with the description or structured context, and inspect any draft.
5. Open the **Price scenario** tab, change capacity and choose **Calculate estimate**. Inspect **Last calculated estimate** and its submitted inputs, then return to **Overview**. Further edits take effect only when you calculate again. The sidebar **Price scenario** is a separate simulation: it starts with empty inputs for a property of your choice, independent of the selected listing.

### Check

The stored listing facts and saved batch estimate should be unchanged. Draft edits affect the download only; nothing is sent or published. A structurally valid review can still be unhelpful or factually unsupported.

**Check the meaning of the draft:** trace every claim back to the description or structured context. Preserve amounts, minimum stays, optional versus mandatory fees, and shared versus private facilities. For example, “two bathrooms” does not mean “two private bathrooms.” An owner question can introduce an unsupported assumption too. If the draft adds one, record the failure and edit or reject the draft; do not rerun until you obtain a nicer answer. Retain the failed example for the next prompt comparison.

### Explain

Which process performed each operation, and which results were persisted? Is the proposed action useful and supported by the supplied facts? Follow [API background review](../apps/listing_review/api/helpers/reviews.py) to connect the job ID with saved stages and logs; the UI and database internals need not be read for this exercise.

## 6. Measure and explain

### Context

More concurrent requests may improve throughput while increasing the time each request takes. Keep inputs and request counts fixed so the comparison is meaningful.

### Tasks — run two measurements

Keep the API running. Run 100 requests at concurrency one, then repeat at concurrency four:

```bash
python -m apps.listing_review_lab.clients.load --requests 100 --concurrency 1
python -m apps.listing_review_lab.clients.load --requests 100 --concurrency 4
```

### Exercise 6 - Measure service behaviour

**Task:** compare successful requests per second, p95 latency, successes, failures and `failure_counts`.

**Check:** verify **successes + failures = requests**. Each run makes one separately reported reference call before the measured workload. A successful measured response must have valid fields and match that reference prediction and model version. Latency percentiles cover successful calls only; these local measurements are not a capacity guarantee.

**Explain:** did throughput and p95 latency change in the same direction? What trade-off do you observe?

## Open challenge — improve the quality, then decide what to deploy

### Context — a working application can still give weak answers

The earlier exercises establish that the system runs consistently. They do not establish that its answers are always useful.

- **ML limitation:** a model can beat the median baseline and still make large errors for individual properties. Sparse groups and nonpositive estimates deserve attention. It predicts advertised prices from a limited set of facts, not an optimal price or booking revenue.
- **AI limitation:** a review can contain valid JSON and exact quotations but draw an unsupported conclusion. For example, “two bathrooms” does not establish “two private bathrooms.” A correct action label does not guarantee a good draft. The two-case prompt experiment is too small to establish general quality.
- **Efficiency limitation:** more training, a larger cloud model or an extra AI stage can increase time and cost without improving useful answers.

**Your challenge:** choose the **ML track or AI track**, implement an alternative, compare it fairly with the current application, and demonstrate whether it should replace the current solution. You may complete both tracks. Keeping the current solution is a valid conclusion when the alternative does not improve it.

### Tasks — start with a testable hypothesis

1. Inspect the saved evaluation results and a few concrete failures. State what is wrong, who it affects and what evidence supports that conclusion.
2. Propose one change and explain why it could address that failure. Write down your acceptance criteria **before** running the comparison: quality improvement, acceptable regressions, and time/cost limits.
3. Use separate candidate/output folders and isolated application state. Preserve the active model, prompts and historical results while experimenting. Keep workloads small and record unsuccessful attempts too.

### Track A — train and try another price model

**Possible direction:** compare a small neural network for tabular regression with the existing Ridge/boosting models. Alternatively, try another supported regressor, improve features available at prediction time, or test a justified target transformation. Deep learning is a hypothesis to test, not an automatic upgrade.

1. **Implement and train the candidate.** Start from [the training comparison](../apps/listing_review/jobs/ml_training/helpers/train.py) and reuse `ml_lib` preprocessing, modelling and evaluation capabilities where appropriate. Keep the same recorded data and host-separated partitions. Fit preprocessing on training data only; use validation for tuning and early stopping.
2. **Make the comparison fair.** Compare the candidate with both the median baseline and the current model on identical validation rows. Report MAE in CAD, group errors with sample counts, nonpositive predictions, training time and inference latency. Inspect individual mistakes as well as averages. Do not hide invalid prices by silently clipping them.
3. **Freeze the decision before test evaluation.** Select settings using validation, then assess the frozen candidate on test. Do not tune again to improve that test score. If you have already used the test results to develop the candidate, acknowledge that the test is no longer an independent final assessment; obtain a separate held-out evaluation before claiming generalization.
4. **Integrate the fitted candidate.** The serving path expects saved fitted preprocessing plus a model that accepts the existing feature contract and returns one finite CAD estimate per record. A neural-network file is not automatically a drop-in replacement for the current bundle: implement and verify its loading/prediction adapter if needed. Preserve model identity, metadata and checksum checks; startup must not train. Rebuild affected images if code or runtime requirements change.
5. **Demonstrate substitution.** First load the candidate in an isolated deployment. Verify direct, API and new-batch predictions agree; test invalid inputs and a property scenario. If the evidence meets your criteria, follow [explicit model replacement and rollback](../apps/listing_review/README.md#replace-an-active-model). Otherwise retain the existing model and explain why. Older saved estimates must remain unchanged.

**Explain:** is the quality gain large enough to justify the added complexity and computation? Which property groups still lack convincing evidence?

### Track B — improve the review's reasoning and wording

**Possible directions:** improve instructions or contrasting examples; add a focused check of whether each proposed claim follows from its cited facts; or compare the current cloud model with an available model you expect to be more capable. Change one factor at a time. An extra stage or a larger model is not proof of better reasoning.

1. **Build a small evaluation set before editing.** Include all four outcomes, legitimate listings that should remain unchanged, missing information, numerical boundaries and the bathroom-count/privacy failure. For each case, record acceptable actions, supported facts and claims the answer must not invent. Keep some cases separate from the examples used to develop prompts.
2. **Implement your candidate.** Start with [prompt templates](../apps/listing_review/jobs/prompt_eng/templates/), [the comparison harness](../apps/listing_review/jobs/prompt_eng/helpers/compare_prompts.py) or [the review chain](../src/lab_helpers/s10_shared/reviews/chain.py), according to your hypothesis. Editing shared policy/stage templates changes both assembled prompt variants: freeze the original instructions separately when testing those edits. The supplied harness compares two prompt variants with one model configuration; extend it explicitly for a model or pipeline comparison.
3. **Compare matched cases and inspect meaning.** Use the same inputs and evaluation rubric for baseline and candidate. Keep generation settings matched where supported and record any unavoidable differences. Assess action appropriateness, quotation accuracy, factual support of every draft claim and usefulness of the next step. Retain provider/format failures as failures. Report regressions, latency, token usage and estimated cost; do not select solely from action-label matches or rerun until an answer looks better.
4. **Check integration.** A new stage must preserve original inputs, save its outputs/timing and handle failure without disabling numerical prediction. A model switch must support the required structured outputs through the provider adapter. Record the model and generation settings for each run. Update snapshots, validation and focused tests if the pipeline changes, while keeping historical jobs readable. An AI checking stage can also make mistakes; inspect its decisions against your rubric.
5. **Demonstrate substitution.** Test the candidate in isolated state first. Activate improved instructions using [the prompt activation procedure](../apps/listing_review/README.md#activate-selected-prompts) only when your evidence supports it. For another model with the current provider, configure `AI_MODEL` and recreate the API container so it receives the new environment; a simple restart does not apply changed Compose environment values. Switching providers additionally requires checking the provider adapter, credentials and Compose configuration. Rebuild images for chain-code changes. Verify a new job's saved instructions/settings and result; keep previous reviews unchanged and retain a rollback path.

**Explain:** did you reduce unsupported claims and unnecessary interventions, or merely produce more fluent text? Is the improvement worth its extra latency and cost?

### Check — show your evidence and defend the decision

Prepare a short demonstration and a comparison table containing:

| Include | What it should establish |
|---|---|
| Failure and hypothesis | A concrete quality problem and a plausible reason for the change |
| Reproducible experiment | Data/cases, model or prompt versions, settings and saved run locations |
| Baseline versus candidate | Quality measures, failures/regressions, time and cost on matched inputs |
| Before/after examples | At least one improvement and one remaining failure or difficult case |
| Integration check | The candidate works through the application and preserves existing results |
| Deployment decision | Replace or retain, with evidence, limitations and a rollback plan |

A well-supported decision **not** to replace the baseline is a successful outcome. The objective is to improve and evaluate the system responsibly, not to make it more complex. When finished, stop your isolated experiment services as well as the standard applications below, retaining your evidence and candidate files.

## Finish — stop both applications

From the repository root:

```bash
docker compose --env-file apps/listing_review/.env -f apps/listing_review/deploy/compose.yaml down
docker compose --env-file apps/listings_simulator/.env -f apps/listings_simulator/deploy/compose.yaml down
```

Stored results remain in `local/listing_review/state/`; removing containers does not delete these files. For service details and troubleshooting, use the [application README](../apps/listing_review/README.md#operation-and-troubleshooting).

The API and jobs reuse `lab_helpers.s10_shared`; general modeling capabilities remain in `ml_lib` and `ai_lib`. See the [lab tools README](../apps/listing_review_lab/README.md) for client and test references.
