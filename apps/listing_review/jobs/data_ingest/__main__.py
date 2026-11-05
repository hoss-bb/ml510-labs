"""Poll for incoming batches, or process available arrivals once with --once."""
import argparse
import time
import httpx
from .helpers.batches import poll
from lab_helpers.s10_shared.prediction.service import Predictor
from lab_helpers.s10_shared.runtime.storage import Store
from lab_helpers.s10_shared.runtime.settings import settings
from lab_helpers.s10_shared.runtime.logging import event


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="Process currently available arrivals, then exit.")
    args = parser.parse_args()
    config = settings()
    store = Store(config.database)
    store.initialize()
    predictor = Predictor(config.model)
    with httpx.Client(base_url=config.upstream_url, timeout=30) as client:
        while True:
            try:
                poll(store, predictor, client)
            except httpx.HTTPError as error:
                event("upstream", outcome="unavailable", error_type=type(error).__name__)
                if args.once:
                    raise SystemExit(1) from None
            if args.once:
                break
            time.sleep(config.poll_seconds)


if __name__ == "__main__":
    main()
