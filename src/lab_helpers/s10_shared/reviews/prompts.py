"""Read and save the three editable stage instructions; no prompt registry."""
import os
from pathlib import Path
from ..runtime.settings import settings

OUTPUT_VERSION = 2
STAGES = ("evidence", "assessment", "proposal")


class PromptError(ValueError):
    pass


def check_instructions(instructions):
    if not isinstance(instructions, dict) or set(instructions) != set(STAGES) or any(
        not isinstance(value, str) or not value.strip() for value in instructions.values()
    ):
        raise PromptError("Expected nonempty evidence, assessment and proposal instructions.")
    return instructions


def save_prompts(directory, instructions):
    check_instructions(instructions)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    for stage in STAGES:
        (directory / f"{stage}.txt").write_bytes(instructions[stage].encode("utf-8"))
    return directory


def load_prompts(directory=None):
    if directory is None:
        directory = Path(os.environ.get("S10_PROMPTS", str(settings().data / "prompts")))
    directory = Path(directory)
    try:
        instructions = {stage: (directory / f"{stage}.txt").read_text(encoding="utf-8") for stage in STAGES}
        return check_instructions(instructions)
    except (OSError, ValueError) as error:
        raise PromptError(f"Cannot read prompts at {directory}. Supply nonempty evidence.txt, assessment.txt and proposal.txt.") from error
