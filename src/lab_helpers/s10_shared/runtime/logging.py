"""Log operational metadata, not listing text or provider error bodies."""
import json
import logging
from datetime import datetime, timezone

LOG = logging.getLogger("s10")


def configure():
    if not LOG.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        LOG.addHandler(handler)
    LOG.setLevel(logging.INFO)
    LOG.propagate = False


def event(stage, **fields):
    configure()
    LOG.info(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "stage": stage,
                         **fields}, allow_nan=False))
