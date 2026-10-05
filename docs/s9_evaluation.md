# S9: From a working assistant to a dependable assistant

Our cloud-support assistant can answer questions. How do we know its answers are useful and supported, and what should happen when it cannot answer well?

Follow the investigation: **evaluate answers - prepare cases - diagnose RAG - improve it - understand guardrails - assemble the controlled assistant**.

## 1. Prepare

Reuse the S8 environment, MTRAG Cloud data and prepared indexes. Follow the [S8 guide](s8_assistant.md) if you have not prepared them.

From the repository root, activate the environment and install the shared requirements:

```powershell
# Windows
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

```bash
# Linux/macOS
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The existing root `.env` supplies the model and Tavily keys. Never put keys in notebook cells. The judge and case generator use the application model unless you explicitly set `S9_JUDGE_PROVIDER`, `S9_JUDGE_MODEL`, `S9_SYNTHETIC_PROVIDER` or `S9_SYNTHETIC_MODEL`. Only configure models available to your account. Before comparing application models, set the judge provider and model explicitly so the evaluator stays fixed.

Evaluation runs locally with OpenEvals and the LangSmith SDK. You do not need a LangSmith account. Results and traces stay in your artifact folders; calls to your chosen model provider and Tavily still use their quotas.

Notebook 02 opens the prepared Qdrant collection. Import the Moodle index using the [S8 index instructions](s8_assistant.md) before running it. Close another notebook using that local collection first. The lab does not rebuild the corpus index.

## 2. Open the notebooks

Download `s9_evaluation` from Moodle, extract it if zipped, and put it inside `notebooks/`. Avoid a doubled `s9_evaluation/s9_evaluation/` folder.

Start Jupyter from the repository root:

```bash
python -m jupyter lab
```

Open `notebooks/s9_evaluation/00_start_here.ipynb` and select the course kernel. Each notebook runs from its own fresh kernel. At the end, copy the printed `source_run` path into the next notebook's clearly marked setup cell. This identifies your actual preceding experiment, rather than an arbitrary latest run.

## 3. Follow the investigation

| Notebook | Question | Your change |
|---|---|---|
| 00 - Evaluate answers | Is it correct, relevant and properly structured? | Change one instruction. |
| 01 - Prepare cases | Do generated questions test something meaningful? | Improve one generation instruction. |
| 02 - Diagnose RAG | Did retrieval find evidence, and did the answer use it? | Diagnose a failure before proposing a fix. |
| 03 - Improve RAG | Does rewriting or a prompt change help? | Keep evidence fixed and change one instruction. |
| 04 - Understand guardrails | What should happen when search fails or returns misleading instructions? | Test a control on both problematic and legitimate requests. |
| 05 - Assemble the assistant | Do integrated controls preserve usefulness? | Change one middleware bound. |

Make a prediction before each comparison. Keep failed changes visible: they help explain which engineering choices matter. Notebook 03 carries forward direct or rewritten queries using development correctness; ties retain the simpler direct query. This small comparison is not a general ranking. There is no submission.

### Read the measures correctly

- **Correctness** compares important facts and conditions with a reference answer.
- **Relevance** asks whether the response addresses the user's request.
- **Groundedness** asks whether supplied evidence supports the answer's claims.
- **Retrieval hit/recall** uses the benchmark's passage judgments. Model-judged retrieval relevance is a different check.
- **Structure and source IDs** can be checked in code. An existing citation is not proof that it supports a claim.

Model judges can disagree or make mistakes. Inspect their reasons and the evidence. Their scores are not probabilities of truth.

A high score among answered requests can conceal unnecessary refusal. Notebook 05 also reports accepted answers among all eligible requests, clarifications and errors.

### Keep comparisons fair

Development and reserved cases come from different conversations. Follow-up replay uses the benchmark's preceding turns. Synthetic cases expand development evidence; they do not replace the original benchmark.

Generation and filtering run automatically, retaining sources and rejection reasons. Inspect a few cases before trusting the set. A separate checking call is still automated judgment, even when it agrees with the generator.

The final reserved evaluation uses local documentation. Current-web questions and controlled challenges are reported separately. Once you inspect reserved results, they are no longer fresh evidence for another change.

### Understand middleware scope

Notebook 04 makes checks explicit. Notebook 05 attaches reusable controls to a LangChain agent inside the graph's web node. Request-wide checks still belong before routing; final checks must cover every answer path.

The web-agent comparison replays notebook 04's saved search evidence to keep comparisons fair. The application itself uses live search by default, and notebook 05 executes one separate live check. No real business action is performed.

## 4. Inspect saved evidence

Open the artifact overview linked at the end of each notebook. It points to compact summaries, complete case results, answers, supporting passages and evaluator explanations.

Successful calls can be reused when their inputs and settings match; results label that reuse. Deliberate repetition experiments make fresh calls. Quota errors are recorded as errors, not poor answers. Correct the access problem and rerun; failed calls are not reused.

To execute all stages with automatic handoffs:

```bash
python scripts/execute_notebooks.py --lab s9 --scope all
```

This creates local output copies in `notebooks/s9_corrections/`. It stops before overwriting annotated instructor corrections. Preserve or rename those copies before another scripted execution. Instructor versions include additional worked TODO cells; students' starting experiments remain separate.


Saved calls are reused only for matching inputs and settings. Search comparisons
may use a saved snapshot; inspect its live/reused label. Provider failures and
incomplete structured responses are not successful answers and are not reused.
Unexpected Python errors stop execution so their traceback can be fixed. A run
is complete only when its expected cases and requested judgments have completed;
a disappointing score does not make an otherwise complete experiment invalid.

## References

- Session 9 course slides - use the deck provided in the Moodle course space.
- [MTRAG benchmark](https://github.com/IBM/mt-rag-benchmark) - original benchmark, attribution and dataset terms; see also the S8 guide.
- [OpenEvals](https://github.com/langchain-ai/openevals) - editable evaluators.
- [LangChain RAG evaluation](https://docs.langchain.com/langsmith/evaluate-rag-tutorial).
- [Local LangSmith evaluation](https://docs.langchain.com/langsmith/local).
- [LangChain middleware](https://docs.langchain.com/oss/python/langchain/middleware/overview).
