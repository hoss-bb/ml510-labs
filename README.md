# ML510 Labs

Intro to AI for Enterprise — Polytechnique Montréal

Practical labs for learning to frame business problems, prepare data, build models, evaluate evidence, and make engineering decisions. This repository grows with the course; each available lab has its own setup instructions, notebook sequence, and exercises.

## Available labs

| Lab | Focus | Start here |
|---|---|---|
| S4 — Regression and forecasting | Contextual estimation, model comparison, and forecasting from history | [Setup and lab guide](docs/s4_regression.md) |
| S5 — Classification and review decisions | Fraud detection under an error constraint, model comparison, calibration, and review workload | [Setup and lab guide](docs/s5_classification.md) |
| S6 — Unsupervised operating-pattern discovery | Compressor regimes, reconstruction, anomaly evidence, and maintenance inspection | [Setup and lab guide](docs/s6_unsupervised.md) |
| S7 - Evaluation and experimentation | Five practical checks: validation, learning curves, group errors, behavioral expectations and decision guardrails | [Setup and lab guide](docs/s7_evaluation.md) |
| S8 - AI application capabilities | Prompts, RAG, conversation, tools, and graphs | [Setup and lab guide](docs/s8_assistant.md) |
| S9 - AI evaluation and controls | Answer quality, RAG diagnosis, guardrails, and middleware | [Setup and lab guide](docs/s9_evaluation.md) |
| S10 - Model serving and applications | CPU prediction, daily arrivals, AI review chains, and Docker Compose | [Setup and lab guide](docs/s10_listings.md) |

Notebook assignments are provided through the course Moodle, not GitHub. Download the lab folder (`s4_regression`, `s5_classification`, `s6_unsupervised`, `s7_evaluation`, `s8_assistant`, or `s9_evaluation`), extract it if zipped, and copy it into `notebooks/`. See the [notebook placement instructions](notebooks/README.md), then follow the relevant guide for installation and data preparation.

## Repository map

```text
docs/              Setup, sequence, and exercises for each lab
notebooks/         Instructions; place Moodle notebooks here locally
src/ml_lib/        Shared ML teaching library
src/ai_lib/        Shared AI application components
src/lab_helpers/   Dataset and session-specific helpers
configs/           Lab data and execution settings
data/              Local datasets and explicitly generated data
artifacts/         Saved experiment results and models
scripts/           Environment, data, and execution tools
apps/              Python applications; S10 runs without notebooks
tests/             Shared-library unit tests
requirements.txt   Shared pinned course dependencies
```

## Shared environment and library

Reusable evaluation helpers accept application data explicitly: `ml_lib.data.validation.validate_partitions` checks declared assignments, `ml_lib.evaluation.groups.regression_group_comparisons` compares indexed predictions within supplied groups, and `ai_lib.evaluation.reports.paired_outcomes` summarizes two methods on an expected case roster. Domain groups and replacement decisions belong to the caller.

`ai_lib` follows the same explicit-submodule approach as `ml_lib`: `models/` handles provider configuration and construction, `retrieval/` handles indexes and passage formatting, `evaluation/` handles judges and metrics, `guardrails/` handles checks, `tools/` supplies integrations, and `experiment/` handles caching and local traces. Import the capability directly, for example `from ai_lib.models.providers import build_chat_model`.

Use the updated S8/S9 notebook copies with this layout; older copies using flat imports such as `from ai_lib.models import ...` need their imports updated. Prepared indexes and saved results do not need rebuilding for this change.

The notebook labs use the course environment defined in `requirements.txt`, which also installs the local packages. S10 has a smaller application environment described in its guide. Follow the available lab guide to create the environment and select its notebook kernel.

`ml_lib` provides reusable preparation, fitting, evaluation, and artifact handling. `lab_helpers` contains lab-specific data rules. Keep modeling choices visible in the notebooks and inspect detailed results in their linked artifact folders.

When changing reusable library behavior, run:

```bash
python -m unittest discover -s tests -t . -v
```

Tests are grouped under `tests/ml/` and `tests/ai/`. See the [test guide](tests/README.md) for running one library, responsibility or module. Review notebook and lab-helper changes by running the affected notebooks.

## License

Copyright (c) 2026 Houssem Ben Braiek. Original code, notebooks, and course materials in this repository are available under the [MIT License](LICENSE).

Third-party materials retain their own licenses. See the separate attribution and licenses for [Bike Sharing](docs/s4_regression.md#2-obtain-and-verify-the-data), [BAF Base](docs/s5_classification.md#2-obtain-and-verify-baf-base), and [MetroPT-3](docs/s6_unsupervised.md#2-obtain-and-verify-metropt-3).
