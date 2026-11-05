"""Temporary active prompts keep tests independent of ignored local instruction files."""
import os
from pathlib import Path
import tempfile
from unittest.mock import patch
from lab_helpers.s10_shared.reviews.prompts import save_prompts
from apps.listing_review.jobs.prompt_eng.helpers.templates import assemble


def install_prompts(test):
    temporary = tempfile.TemporaryDirectory()
    test.addCleanup(temporary.cleanup)
    path = Path(temporary.name) / "prompts"
    save_prompts(path, assemble("starting"))
    environment = patch.dict(os.environ, {"S10_PROMPTS": str(path)})
    environment.start()
    test.addCleanup(environment.stop)
