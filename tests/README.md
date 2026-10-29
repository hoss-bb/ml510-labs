# Shared-library tests

Run commands from the repository root with the course environment active.

```bash
# Everything
python -m unittest discover -s tests -t . -v

# One library
python -m unittest discover -s tests/ml -t . -v
python -m unittest discover -s tests/ai -t . -v

# One responsibility or module
python -m unittest discover -s tests/ml/evaluation -t . -v
python -m unittest tests.ai.test_retrieval -v
```

| Folder | What it checks |
|---|---|
| `ml/data/` | Input data and split boundaries |
| `ml/features/` | Preprocessing, temporal features and windows |
| `ml/models/` | Baselines, classical and neural adapters, calibration and ensembles |
| `ml/evaluation/` | Metrics, reports, selection, policies and behavioral comparisons |
| `ml/experiment/` | Saved artifacts and source fingerprints |
| `ai/` | Providers, retrieval, recording, evaluation and controls |

Fixtures live beside their library's tests. Tests should import fixtures, not other
test modules. Keep package `__init__.py` files so `unittest` discovers every folder;
`-t .` keeps module names rooted at `tests`.

The suite uses small synthetic inputs and mocked cloud responses. It does not need
API keys, real datasets or a corpus index. Neural tests use tiny TensorFlow models;
tests requiring an unavailable TensorFlow runtime report a skip. Review notebook
and lab-helper changes by executing the affected notebooks separately.
