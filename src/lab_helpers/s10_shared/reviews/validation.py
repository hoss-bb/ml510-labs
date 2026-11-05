"""Evidence and action checks, independent of model calls and persistence."""
from .errors import ReviewError


def validate_stage(stage, parsed, sources, outputs):
    """Return a checked stage output with stable evidence identifiers."""
    output = parsed.model_dump()
    if stage == "evidence":
        for index, item in enumerate(parsed.items, 1):
            source = sources.get(item.source_id)
            if source is None:
                raise ReviewError("evidence_source")
            valid = item.excerpt in source if item.source_id == "listing.description" else item.excerpt == source
            if not valid:
                raise ReviewError("unsupported_excerpt")
            output["items"][index - 1]["id"] = f"e{index}"
    else:
        valid_ids = {item["id"] for item in outputs["evidence"]["items"]}
        if stage == "assessment":
            cited = [identifier for finding in parsed.findings for identifier in finding.evidence_ids]
        else:
            cited = parsed.evidence_ids
            kinds = {finding["kind"] for finding in outputs["assessment"]["findings"]}
            required = None
            if "conflict" in kinds:
                required = "confirm_facts"
            elif "clarification" in kinds:
                required = "clarify_condition"
            if required is not None and parsed.action != required:
                raise ReviewError("proposal_support")
        if any(identifier not in valid_ids for identifier in cited):
            raise ReviewError(f"{stage}_citation")
        if stage == "proposal" and parsed.action != "no_change":
            kind = {"confirm_facts": "conflict", "clarify_condition": "clarification", "guest_note": "guest_condition"}[parsed.action]
            findings = [finding for finding in outputs["assessment"]["findings"] if finding["kind"] == kind]
            if not any(set(finding["evidence_ids"]).issubset(parsed.evidence_ids) for finding in findings):
                raise ReviewError("proposal_support")
    return output
