"""Snapshot generation settings without importing the review chain."""
from dataclasses import asdict
from ai_lib.models.configuration import ModelSettings, load_settings
from ..runtime.settings import ROOT, load_environment


def generation_settings():
    load_environment()
    settings = load_settings(ROOT)
    return asdict(ModelSettings(settings.provider, settings.model, temperature=0,
                               max_tokens=2048, thinking_budget=settings.thinking_budget))
