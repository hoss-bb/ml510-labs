"""Compare embedded and HTTP inference on exactly the same inputs."""
import argparse
import json
import math
import httpx
from lab_helpers.s10_shared.prediction.service import Predictor
from lab_helpers.s10_shared.prediction.schemas import Features, PredictionResponse
from lab_helpers.s10_shared.runtime.settings import settings

from lab_helpers.s10_shared.examples import load_example_listing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", action="store_true")
    parser.add_argument("--listing-id", help="Compare an arrived listing with its saved batch estimate.")
    args = parser.parse_args()
    config = settings()
    if args.listing_id:
        with httpx.Client(base_url=config.api_url, timeout=15) as client:
            response = client.get(f"/listings/{args.listing_id}")
            response.raise_for_status()
            saved = response.json()
            inputs = {key: saved["listing"][key] for key in Features.model_fields}
            response = client.post("/predict", json=inputs)
            response.raise_for_status()
            online = PredictionResponse.model_validate(response.json()).model_dump()
        direct = Predictor(config.model).predict([Features(**inputs)])[0].model_dump()
        batch = saved["prediction"]
        matches = all(value["model_version"] == batch["model_version"] and
                      value["currency"] == batch["currency"] and
                      value["status"] == batch["status"] and
                      math.isclose(value["estimate"], batch["estimate"], rel_tol=1e-9, abs_tol=1e-6)
                      for value in [online, direct])
        print(json.dumps({"listing_id": args.listing_id, "batch": batch, "direct": direct,
                          "api": online, "matches": matches}, indent=2))
        if not matches:
            raise SystemExit("Predictions differ: inspect inputs and model versions before continuing.")
        return
    if args.api:
        response = httpx.post(config.api_url + "/predict", json=load_example_listing(), timeout=15)
        response.raise_for_status()
        result = PredictionResponse.model_validate(response.json()).model_dump()
    else:
        result = Predictor(config.model).predict([Features(**load_example_listing())])[0].model_dump()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
