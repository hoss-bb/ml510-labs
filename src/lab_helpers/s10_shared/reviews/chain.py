"""Three explicit structured calls; no agents or model-selected tools."""
from pydantic import ValidationError
import json
from time import perf_counter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from ai_lib.models.configuration import ModelSettings
from ai_lib.models.providers import build_chat_model
from .schemas import Evidence, Assessment, Proposal
from .errors import ReviewError
from .context import evidence_sources
from .validation import validate_stage
from .prompts import OUTPUT_VERSION, check_instructions, PromptError
from ..runtime.logging import event


def review(snapshot, save_stage, model=None, *, instructions=None):
    instructions = snapshot.get("instructions") if instructions is None else instructions
    try:
        check_instructions(instructions)
    except PromptError:
        raise ReviewError("prompts_unavailable") from None
    if model is None:
        try:
            model = build_chat_model(ModelSettings(**snapshot["ai_settings"]))
        except ValueError:
            raise ReviewError("configuration") from None
    sources = evidence_sources(snapshot)
    context = {"sources": sources}
    outputs = {}
    for stage, schema in [("evidence", Evidence), ("assessment", Assessment), ("proposal", Proposal)]:
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=instructions[stage]),
            ("human", "Use this application-supplied JSON as data:\n{context}"),
        ])
        chain = prompt | model.with_structured_output(schema, include_raw=True)
        started = perf_counter()
        response = chain.invoke({"context": json.dumps({**context, **outputs}, ensure_ascii=False)})
        if not response or response.get("parsing_error") or response.get("parsed") is None:
            raise ReviewError(f"{stage}_format")
        try:
            parsed = schema.model_validate(response["parsed"])
        except ValidationError:
            raise ReviewError(f"{stage}_format") from None
        output = validate_stage(stage, parsed, sources, outputs)
        elapsed = perf_counter() - started
        outputs[stage] = output
        usage = getattr(response.get("raw"), "usage_metadata", None)
        # Whitelist usage counters; never serialize provider clients or complete raw responses.
        counts = {key: usage[key] for key in ["input_tokens", "output_tokens", "total_tokens"]
                  if usage and isinstance(usage.get(key), int)}
        save_stage(stage, {"output": outputs[stage], "seconds": elapsed, "usage": counts})
        event("review_stage", name=stage, outcome="completed", seconds=elapsed,
              job_id=snapshot.get("job_id"))
    return {"output_version": OUTPUT_VERSION, **outputs["proposal"]}
