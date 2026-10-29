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

| Notebook | Question | Main exercise |
|---|---|---|
| 00 - Answer and judge | What does each check establish, and can the judge be wrong? | Review four teaching answers, then challenge one judgment. |
| 01 - Cases | Which generated cases are supported and usable? | Improve one generation instruction on the same source. |
| 02 - RAG metrics | Which evidence relationship failed? | Inspect statement-level verdicts and diagnose one failure. |
| 03 - Regression | Does a rewrite preserve the required answer? | Verify an apparent gain, check neighbouring cases and justify which method to keep. |
| 04 - Validation | Should a failed check cause repair, clarification or a hold? | Test a control against both a problematic and legitimate request. |
| 05 - Complete application | Do controls preserve useful completion? | Change one call limit and inspect the trade-off. |

Make a prediction before running each comparison. One principal TODO appears in
each notebook; nothing is submitted. Results need not improve for an experiment
to teach something useful.

### Read the measures correctly

Notebook 00 compares exact match and token F1 with a four-level correctness rubric:
**0 incorrect, 1 partly correct, 2 nearly correct, 3 correct and complete**.
The separate strict pass flag requires anchor 3. A rubric rating is not a probability.
Compare semantic scores on the same judged cases. Report invalid answers separately,
and calculate end-to-end pass rates over every eligible case. The targeted prompt
experiment keeps all six development cases and the same evaluator.
Review the proposed human labels yourself; automatic judgments are not independent
expert adjudication.

Notebook 02 keeps the metric inputs and denominators visible:

| Measure | What is compared |
|---|---|
| Passage-ID hit/recall@5 | Retrieved IDs against benchmark relevance judgments |
| Contextual precision | Useful retrieved nodes and their ranked positions |
| Contextual recall | Required reference statements against supplied evidence |
| Contextual relevancy | Context statements against the user's request |
| Faithfulness | Answer claims against the actual supplied context |
| Answer relevancy | Emitted answer statements against the user's request |
| Correctness | The complete answer against required facts and conditions |

These implement the course definitions with explicit rubrics and Python arithmetic;
they are not claims of reproducing a particular framework's default metrics.
Calculate the ranked-precision example by hand, then inspect the saved units, reasons and denominators. Small, clearly constructed variants use known labels to isolate the arithmetic; real-case verdicts remain automated judgments that can be wrong. No applicable units means an
unavailable score, not a perfect answer. Missing judgments are evaluation errors.
A retrieved passage without a benchmark label can still be useful.

Notebook 02 also challenges a judge that rejects useful explanations as irrelevant.
Declare the reviewed development labels first, then rejudge unchanged statements
with one clearer criterion. Higher scores alone do not establish a better evaluator.

Notebook 03 distinguishes a confirmed regression from an existing weakness and
separates answer repetition from judging an unchanged answer repeatedly. Compare
direct retrieval, the original rewrite and one product-preserving revision on the
same six follow-ups. Distinguish score deterioration from a pass-to-fail regression. Cache
reuse is labelled and is not measured generation latency.

Notebook 04 separates **PASS, FAIL and ERROR** from what the application does next.
A bounded citation repair rechecks the new candidate. Passing structure and source-ID
checks does not establish claim support. Injection probes check both a marker and a
false claim of an account action; the assistant has no account-write capability.

Notebook 05 uses the same web-agent architecture, prompt and saved evidence on both
sides of the control comparison. It checks tool-call/result association, task completion,
false acceptance and false rejection on labelled fixtures. Report accepted answers
among all eligible cases and among answered cases, alongside clarifications and errors.
The illustrative release rule checks whether any case previously rated correct
and complete (3/3) falls below that correctness anchor; this is not a separate fact-by-fact verification. Inspect the actual controlled
outputs as well as the fixture rates.


### Keep comparisons fair

Development and reserved cases come from different conversations. Follow-up replay uses the benchmark's preceding turns. Synthetic cases expand development evidence; they do not replace the original benchmark.

Generation and filtering run automatically, retaining sources and rejection reasons. Inspect a few cases before trusting the set. A separate checking call is still automated judgment, even when it agrees with the generator.

The final reserved evaluation uses local documentation. Some reserved cases were
already inspected during lab development; they are not fresh confirmation. Current-web questions and controlled challenges are reported separately. Once you inspect reserved results, they are no longer fresh evidence for another change.

### Understand middleware scope

Notebook 04 makes checks and one repair attempt explicit. Notebook 05 attaches reusable controls to a LangChain agent inside the graph's web node. Request-wide checks still belong before routing; final checks must cover every answer path. The web agent gathers evidence and writes a draft, then one tool-free call produces the answer schema. The agent call limit excludes this final call; a limit stop skips it and is reported separately.

The web-agent comparison replays notebook 04's saved search evidence to keep comparisons fair. The application itself uses live search by default, and notebook 05 executes one separate live check. No real business action is performed.

## 4. Inspect saved evidence

Open the artifact overview linked at the end of each notebook. It points to compact summaries, complete case results, answers, supporting passages and evaluator explanations.

Successful calls can be reused when their inputs and settings match; results label that reuse. Deliberate repetition experiments make fresh calls. Quota errors are recorded as errors, not poor answers. Correct the access problem and rerun; failed calls are not reused. Unexpected Python errors stop execution with their traceback. A run is complete only when its expected cases and judgments have completed; a disappointing score is still a valid result.

To execute all stages with automatic handoffs:

```bash
python scripts/execute_notebooks.py --lab s9 --scope all
```

This creates local output copies in `notebooks/s9_corrections/`. It stops before overwriting annotated instructor corrections. Preserve or rename those copies before another scripted execution. Instructor versions include additional worked TODO cells; students' starting experiments remain separate.

## References

- Session 9 course slides - use the deck provided in the Moodle course space.
- [MTRAG benchmark](https://github.com/IBM/mt-rag-benchmark) - original benchmark, attribution and dataset terms; see also the S8 guide.
- [OpenEvals](https://github.com/langchain-ai/openevals) - editable evaluators.
- [LangChain RAG evaluation](https://docs.langchain.com/langsmith/evaluate-rag-tutorial).
- [Local LangSmith evaluation](https://docs.langchain.com/langsmith/local).
- [LangChain middleware](https://docs.langchain.com/oss/python/langchain/middleware/overview).

## Course coverage

The core exercises target 24 of the 37 substantive teaching topics in Session 9.
The mapping uses slide positions in the updated deck, excluding section dividers,
activities, the lab recap and research bonuses. Coverage means running a check,
calculating a measure or explicitly reviewing evidence, not merely mentioning it.

| Notebook | Core slide positions |
|---|---|
| 00 | 7, 8, 12, 13, 15 |
| 01 | 14 |
| 02 | 18-25 |
| 03 | 49-50 |
| 04 | 39, 40, 43, 46 |
| 05 | 30, 31, 47, 51 |

Lexical overlap, repeated-run effort and privacy receive focused demonstrations.
Full lexical-metric catalogs, perplexity, separate embedding evaluators, professional
advice, account-write approval systems and the research bonuses stay outside this lab.
