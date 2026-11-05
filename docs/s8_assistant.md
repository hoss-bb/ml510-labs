# S8 — Build a cloud-support assistant

Help an IBM Cloud user find and understand documentation. Build one application
through prompts, chains, retrieval, conversation history, tools, and a graph.
S8 makes the components work together. S9 will examine their quality and controls.

## 1. Prepare the environment

Use the existing Python 3.12 course environment. From the repository root:

```bash
python -m pip install -r requirements.txt
python -m pip check
python scripts/check_environment.py --strict-pins
```

For a new installation, follow the shared Windows or Linux/macOS environment and
kernel instructions in [S4](s4_regression.md). The new packages retain the existing
course pins. Retrieval and candidate embeddings run locally on CPU.

Copy `.env.example` to `.env` at the repository root. On Windows use
`Copy-Item .env.example .env`; on Linux/macOS use `cp .env.example .env`.
Preserve an existing `.env` rather than overwriting its keys.

```dotenv
AI_PROVIDER=gemini
AI_MODEL=gemini-2.5-flash
GOOGLE_API_KEY=your_local_key
TAVILY_API_KEY=your_local_key
```

Get a model key from [Google AI Studio](https://aistudio.google.com/apikey),
[Groq](https://console.groq.com/keys), or [Cohere](https://dashboard.cohere.com/api-keys).
Choose one chat provider: `gemini`, `groq`, or `cohere`. Set `AI_MODEL` to a model available
to that account; for example, Cohere `command-a-03-2025`. Groq uses `GROQ_API_KEY`;
Cohere uses `COHERE_API_KEY`. Only the selected chat provider needs a key.
Gemini is the measured starting provider for this lab.
Gemini 2.5 Flash runs with extra reasoning disabled and a 1,024-token output budget
for these short examples. Both settings are recorded. Cohere's current integration uses its native output budget; artifacts record this
as unspecified rather than claiming the Gemini/Groq limit applies.

Get a [Tavily key](https://app.tavily.com/) for notebooks 04–05. Trial access and
quotas depend on the provider and account. Tavily currently offers 1,000 monthly
free credits; basic search consumes one. No provider is substituted automatically.

Enter keys only in `.env`, which Git ignores. Existing environment variables take
precedence. Restart the kernel after changing keys or model settings. Never include
keys in notebook cells, screenshots, Moodle archives, or artifact records.

## 2. Prepare the evidence

```bash
python scripts/prepare_s8.py
python scripts/verify_data.py --lab s8
```

The first command downloads the pinned official MTRAG Cloud corpus, builds BM25,
and loads the small local `BAAI/bge-small-en-v1.5` embedding model. It does not
embed the full corpus. The model download requires Hugging Face access the first
time; no embedding API key or vector database is needed.

At query time BM25 finds 30 candidates across the full corpus. Local embeddings
rank only those candidates; the answer chain receives the top five original
passages. Candidate and query embeddings are cached. The first query can take
longer than a repeated query. Semantic ranking cannot recover passages that BM25
did not retrieve. We compare this pipeline with BM25 alone.

Notebook 02 also has a second phase: **direct vector search with Qdrant**. It uses
the same local embedding model, but searches every indexed passage instead of
BM25's candidates. You can skip this phase and continue through notebooks 03–05.

### Use the prepared index from Moodle

After preparing the environment and dataset, download `s8_qdrant.zip` from Moodle
and import it from the repository root:

```bash
python scripts/s8_qdrant.py import --archive path/to/s8_qdrant.zip
```

Use the path where you saved the ZIP; you do not need to unzip it yourself.
This restores the computed vectors without embedding the corpus again. Keep the
local embedding model: it is still used to turn each question into a search vector.
Import before the lab: writing the full local collection can take several minutes.

Set `run_vector_search = True` for notebook 02's second phase, or
`retrieval_method = "qdrant"` in notebooks 03 and 05. You can keep using BM25 plus
local ranking without importing anything.

Developed on Windows. The portable archive and Qdrant client should also work on
Linux and macOS. Close notebooks using Qdrant before importing. An existing
collection is left unchanged; move it aside yourself if you want to replace it.
No Docker or Qdrant API key is needed.

### Prepare an index to share

To build your own collection, run `python scripts/prepare_s8.py --build-qdrant`.
This can take hours on CPU; rerun the command to resume an interrupted build.
Once it finishes, close notebooks using the collection and export it:

```bash
python scripts/s8_qdrant.py export --output artifacts/s8_delivery/s8_qdrant.zip
```

Share that ZIP through Moodle. It contains vectors, passage metadata, and source
attribution, not notebooks, API keys, or model caches. Students still prepare the
original dataset with the command above.

The pinned archive contains **72,442 unique passage records**, including three
empty records excluded from search. This differs from the upstream README's
61,022 count. The manifest records the actual files and counts. Long passages use
overlapping token windows within the local model limit; retrieval returns their original passage IDs.

There are **205 Cloud tasks in 26 conversations**. Six fixed conversations are
teaching examples, selected before model execution; the others remain reserved
for S9. Human-reviewed answers and relevance judgments support inspection.
These are documentation conversations, not logs of a deployed support service.

Keep reference answers out of ordinary prompts, query rewriting, and routing.
Notebook 01 deliberately supplies reference passages to separate generation from
retrieval. A retrieved passage without a relevance judgment may still be useful.

Dataset attribution and license: [raw-data instructions](../data/raw/mtrag_cloud/README.md).

## 3. Open the notebooks

Download `s8_assistant` from Moodle, extract it if zipped, and place the folder in
`notebooks/`. Start Jupyter from the repository root:

```bash
python -m jupyter lab
```

Open `notebooks/s8_assistant/00_start_here.ipynb` and choose the course kernel.
Every notebook prepares its own state and reuses the saved retrieval indexes.

| Notebook | Build | Main exploration |
|---|---|---|
| 00 — Start here | Direct and instructed model calls | Change one instruction |
| 01 — Chains | Prompt, model, parser, structured response | Add a useful response field |
| 02 — Retrieval | BM25, embeddings, RAG chain | Investigate a query and retrieval miss |
| 03 — Conversation | Recover the missing product, retrieve evidence, answer the original request | Apply the same steps to the Python SDK follow-up |
| 04 — Tools | Tavily call, model-requested tool, returned evidence | Clarify when the tool should be used |
| 05 — Graphs | State, nodes, conditional edges, execution | Continue after clarification and stop explicitly |

Predict before running each check. Complete the embedded TODOs; nothing is submitted.
Notebook 02 distinguishes the six first-turn comparisons from its optional Cloudant
follow-up example. State which query you change and use only information supplied
by the user or preceding conversation. In notebook 04, keep the complete user message
unchanged when comparing tool descriptions.
In notebooks 03 and 05, set `retrieval_method = "bm25_local"` (the default) or
`"qdrant"` in the setup cell. Choose separately in each notebook; no earlier kernel
state is needed. Qdrant requires the completed collection imported from Moodle or
built with the preparation command. Missing or incomplete collections produce a preparation message, with no
silent fallback. Rerun setup after changing the choice. Each run records the method.
Notebook 04 uses Tavily independently; in notebook 05 this setting changes only
the local-document node. Run the closing cell before opening Qdrant in another
notebook, or shut down the old kernel if execution stopped early.

The notebooks are the application's readable entry points. `ai_lib` supplies shared
components; the S8 helper supplies MTRAG-specific examples. Prompts and application
logic remain visible in notebook cells.

Shared code uses explicit submodules such as `ai_lib.retrieval.indexing`,
`ai_lib.retrieval.qdrant` and `ai_lib.experiment.recording`. Use the updated notebook
copies with these imports; the existing prepared indexes remain the same.

## 4. Inspect what ran

Open the artifact overview linked at the end of each notebook. Inspect answers,
retrieved passage IDs, tool calls/results, graph paths, and reported token usage.
Details stay in `artifacts/s8_<notebook>/<run-id>/`. Hosted tracing is disabled.

Benchmark replay uses the supplied preceding turns. The generated-history example
uses the application's own prior answers and is recorded separately. Web search
asks about current documentation; it is a course extension, not a score against
historical MTRAG answers. Record sources and retrieval dates, not an invented
overall answer-quality score.

Notebook 03 starts with a question about adding an image or PDF to a document.
The conversation identifies IBM Cloudant; the search query makes that product
explicit. Compare the retrieved evidence and both saved answers, then continue
using the application's own answer in the message history. These are selected
teaching examples, not a claim that rewriting always improves retrieval.

Notebook 05 adds a conversation-loop exercise: ask for missing information, pass the
user reply and history into a new graph invocation, and stop after an answer, `quit`,
or two clarification replies. Prepared replies keep Run All non-interactive.

The graph shows application logic. Timeouts, limited retries, and a step limit keep
an exercise from running indefinitely. Systematic routing evaluation, adversarial
tests, and guardrails belong to S9.

Command-line execution is also available:

```bash
python scripts/execute_notebooks.py --lab s8 --scope all --kernel ml510
```

It makes live API calls and writes local output copies to `notebooks/s8_corrections/`.
It stops before overwriting annotated instructor solutions. Preserve or rename
those copies first. Corrections supplied privately contain starting experiments
plus separately marked, runnable instructor solutions.

## 5. If execution stops

- **Missing key:** edit the root `.env` and restart the kernel.
- **Unavailable model:** check the provider's model list and account access; change
  `AI_MODEL` explicitly. Changing a model creates a different experiment.
- **Quota reached:** preserve the partial run and retry when access is available.
  A partially executed notebook is not a completed result.
- **Missing or mismatched index:** rerun preparation. If corpus/settings changed,
  prepare a matching cache directory.
- **Slow first query:** local embeddings are calculated only for the BM25 candidates
  and cached. Inspect cold and repeated-query timing; neither needs cloud embeddings.


A rewritten query is only a retrieval aid. The answer still addresses the original
question and conversation. If search fails, fix that execution failure before
interpreting the answer; a successful search with no evidence is a different outcome.
Close the Qdrant connection using the final cleanup cell, or shut down the kernel
before opening the same local collection elsewhere.

## References

- [MTRAG benchmark and evidence](https://github.com/IBM/mt-rag-benchmark)
- [LangChain model interfaces](https://docs.langchain.com/oss/python/langchain/models)
- [LangGraph graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)
- [FastEmbed](https://qdrant.github.io/fastembed/)
- [Qdrant local mode](https://github.com/qdrant/qdrant-client#local-mode)
- [Tavily search](https://docs.tavily.com/documentation/api-reference/endpoint/search)

API availability and web results can change. If a call fails, check your account access and quota.
