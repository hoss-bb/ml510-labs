# Course notebooks

Notebook assignments are provided through the course Moodle, separately from this GitHub repository.

Download the provided `s4_regression`, `s5_classification`, `s6_unsupervised`, `s7_evaluation`, or `s8_assistant` materials for your session. If they arrive as a ZIP, extract it and copy the resulting lab folder into this `notebooks/` folder. Copying the ZIP alone is not enough.

The first notebook must be at:

```text
notebooks/s4_regression/00_start_here.ipynb
notebooks/s5_classification/00_start_here.ipynb
notebooks/s6_unsupervised/00_start_here.ipynb
notebooks/s7_evaluation/00_validation.ipynb
notebooks/s8_assistant/00_start_here.ipynb
```

Avoid an extra nested folder such as `s4_regression/s4_regression/` or `s5_classification/s5_classification/`. Keep each session's notebook files together so their navigation and documentation links work.

Follow the [S4 guide](../docs/s4_regression.md) or [S5 guide](../docs/s5_classification.md) for the environment, dataset, and notebook sequence. Start Jupyter from the repository root and select **Python (ML510)**.

Downloaded notebooks, local results, and instructor corrections are ignored by Git. This README is the public entry point for the folder.

For Session 6, obtain `s6_unsupervised` from Moodle and follow the [S6 guide](../docs/s6_unsupervised.md). The same extraction instructions apply.

For Session 7, follow the [S7 guide](../docs/s7_evaluation.md). Also copy the Moodle `s7_evidence` folder into `data/processed/`. The five notebooks reuse earlier results and perform small training exercises.

## S8 - Cloud-support assistant

Download `s8_assistant` from Moodle and copy it into this folder. Open
`s8_assistant/00_start_here.ipynb` locally after following the [S8 guide](../docs/s8_assistant.md).
Keep the root `.env` and its keys out of notebook delivery archives.

## S9

Download `s9_evaluation` from Moodle and place it directly in this folder. Open `s9_evaluation/00_start_here.ipynb` locally and follow the [S9 guide](../docs/s9_evaluation.md).
