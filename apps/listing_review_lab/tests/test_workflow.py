"""Exercise service boundaries with a small fitted model and temporary state."""
from dataclasses import replace
import hashlib
import io
import os
import json
from pathlib import Path
import tempfile
import unittest
from apps.listing_review_lab.tests.prompt_fixtures import install_prompts
from unittest.mock import patch
import httpx
import pandas as pd
from fastapi.testclient import TestClient
from pydantic import ValidationError
from ai_lib.models.errors import ExternalCallError
from ml_lib.models.classical import build_regressor
from ml_lib.models.interfaces import ModelSpec
from ml_lib.features.schema import FeatureSchema
from ml_lib.features.preprocessing import PreprocessingConfig
from ml_lib.data.loading import sha256_file
from lab_helpers.s10_shared.runtime.settings import settings
from lab_helpers.s10_shared.prediction.schemas import Listing, Features
from lab_helpers.s10_shared.prediction.service import Predictor
from lab_helpers.s10_shared.runtime.storage import Store, Conflict
from apps.listing_review.api.app import create_app
from lab_helpers.s10_shared.reviews.prompts import load_prompts
from apps.listing_review.jobs.data_ingest.helpers.batches import process, poll
from apps.listing_review.api.helpers.reviews import process_review
from lab_helpers.s10_shared.examples import load_example_listing
from apps.listings_simulator.simulator.api import create_app as upstream_app


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        install_prompts(self)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        model_dir = self.root / "model"
        model_dir.mkdir()
        schema = FeatureSchema(numeric=("accommodates", "bedrooms", "bathrooms", "beds"),
                               categorical=("property_type", "room_type", "neighbourhood"))
        model = build_regressor(ModelSpec("ridge", {"alpha": 1}), schema,
                                preprocessing=PreprocessingConfig(True, "impute"))
        examples = [{**load_example_listing(), "accommodates": i + 1} for i in range(8)]
        model.fit(pd.DataFrame(examples), [60 + i * 20 for i in range(8)])
        model.save(model_dir / "model.pkl")
        (model_dir / "metadata.json").write_text(json.dumps({"version": "test-v1", "model_sha256": sha256_file(model_dir / "model.pkl")}), encoding="utf-8")
        self.predictor = Predictor(model_dir)
        self.store = Store(self.root / "state.sqlite")
        self.store.initialize()
        self.config = replace(settings(), data=self.root, database=self.store.path, model=model_dir)
        self.listing = Listing(**load_example_listing(), listing_id="101", description="Two bedrooms for four guests.", advertised_price=160)

    def content(self, rows=None):
        return pd.DataFrame(rows or [self.listing.model_dump()]).to_csv(index=False).encode()

    def ingest(self, batch="day_001", content=None):
        data = content or self.content()
        return process(self.store, self.predictor, batch, hashlib.sha256(data).hexdigest(), data)

    def client(self):
        return TestClient(create_app(self.config, self.predictor))

    def job(self, request="request-001"):
        return self.store.create_job("101", request, load_prompts(), {"provider": "gemini", "model": "mock"})

    def test_prediction_paths(self):
        direct = self.predictor.predict([Features(**load_example_listing())])[0].model_dump()
        self.ingest()
        self.assertEqual(self.store.listing("101")["prediction"], direct)
        with self.client() as client:
            result = client.post("/predict", json=load_example_listing())
            self.assertEqual(result.status_code, 200)
            payload = result.json()
            self.assertEqual(payload.pop("request_id"), result.headers["X-Request-ID"])
            self.assertEqual(payload, direct)

    def test_missing_prompts_does_not_disable_prediction(self):
        self.ingest()
        with patch.dict(os.environ, {"S10_PROMPTS": str(self.root / "missing-prompts")}), self.client() as client:
            self.assertEqual(client.get('/health/ready').status_code, 200)
            self.assertEqual(client.post('/predict', json=load_example_listing()).status_code, 200)
            self.assertEqual(client.post('/listings/101/reviews', json={'request_id': 'missing-prompts'}).status_code, 503)
        self.assertEqual(self.store.query('SELECT count(*) AS n FROM jobs')[0]['n'], 0)

    def test_invalid_input(self):
        with self.client() as client:
            result = client.post("/predict", json={**load_example_listing(), "accommodates": "secret-value"})
            self.assertEqual(result.status_code, 422)
            self.assertNotIn("secret-value", result.text)
        with self.assertRaises(ValidationError):
            Features(**{**load_example_listing(), "bathrooms": float("inf")})

    def test_batch_accounting(self):
        rows = [self.listing.model_dump(), {**self.listing.model_dump(), "listing_id": "102", "accommodates": -1}]
        self.assertEqual(self.ingest(content=self.content(rows)), {"total": 2, "accepted": 1, "rejected": 1})

    def test_duplicate_poll(self):
        self.ingest()
        self.assertIsNone(self.ingest())
        self.assertEqual(len(self.store.query("SELECT * FROM listings")), 1)

    def test_duplicate_listing(self):
        self.ingest()
        self.assertEqual(self.ingest("day_002"), {"total": 1, "accepted": 0, "rejected": 1})

    def test_changed_batch(self):
        self.ingest()
        with self.assertRaises(Conflict):
            self.ingest(content=self.content([{**self.listing.model_dump(), "advertised_price": 999}]))

    def test_checksum(self):
        with self.assertRaises(ValueError):
            process(self.store, self.predictor, "day_001", "incorrect", self.content())
        self.assertEqual(self.store.query("SELECT * FROM batches"), [])

    def test_interrupted_batch(self):
        checksum = hashlib.sha256(self.content()).hexdigest()
        self.store.begin_batch("day_001", checksum)
        self.assertEqual(self.ingest()["accepted"], 1)

    def test_failed_batch(self):
        with patch.object(self.predictor, "predict", side_effect=RuntimeError("bug")):
            with self.assertRaises(RuntimeError):
                self.ingest()
        self.assertEqual(self.store.query("SELECT status FROM batches")[0]["status"], "failed")
        self.assertEqual(self.ingest()["accepted"], 1)

    def test_upstream_outage(self):
        self.ingest()
        with httpx.Client(base_url="http://example", transport=httpx.MockTransport(lambda request: httpx.Response(503))) as client:
            with self.assertRaises(httpx.HTTPStatusError):
                poll(self.store, self.predictor, client)
        self.assertEqual(self.store.listing("101")["listing"]["listing_id"], "101")

    def test_simulated_arrival(self):
        (self.root / "incoming").mkdir()
        with TestClient(upstream_app((self.root / "incoming"))) as client:
            self.assertEqual(client.get("/batches").json(), [])
            ((self.root / "incoming") / "day_001.csv").write_bytes(self.content())
            index = client.get("/batches").json()
            self.assertEqual(len(index), 1)
            result = client.get("/batches/day_001")
            self.assertEqual(hashlib.sha256(result.content).hexdigest(), index[0]["sha256"])

    def test_job_deduplication(self):
        self.ingest()
        first = self.job()
        self.assertEqual(first, self.job())
        with self.assertRaises(Conflict):
            self.store.create_job("102", "request-001", load_prompts(), {})
        self.assertNotEqual(first, self.job("request-002"))

    def test_job_snapshot(self):
        self.ingest()
        job = self.store.job(self.job())
        self.assertEqual(job["snapshot"]["listing"], self.listing.model_dump())
        self.assertEqual(job["snapshot"]["prediction"]["model_version"], "test-v1")
        self.assertEqual(job["snapshot"]["instructions"], load_prompts())

    def test_prompt_inputs_are_snapshotted_and_deduplicated(self):
        self.ingest()
        instructions = load_prompts()
        original = dict(instructions)
        job_id = self.store.create_job("101", "saved-prompts-001", instructions, {})
        instructions["evidence"] = "Later instructions"
        repeated = self.store.create_job("101", "saved-prompts-001", instructions, {})
        self.assertEqual(repeated, job_id)
        self.assertEqual(self.store.job(job_id)["snapshot"]["instructions"], original)

    def test_legacy_queued_job_fails_without_provider_call(self):
        self.ingest()
        job_id = self.job()
        snapshot = self.store.job(job_id)["snapshot"]
        del snapshot["instructions"]
        snapshot["prompt_version"] = "historical"
        with self.store.connection() as db:
            db.execute("UPDATE jobs SET snapshot=? WHERE id=?", (json.dumps(snapshot), job_id))
        with patch("lab_helpers.s10_shared.reviews.chain.build_chat_model") as provider:
            self.assertTrue(process_review(self.store, job_id))
        provider.assert_not_called()
        saved = self.store.job(job_id)
        self.assertEqual(saved["status"], "failed")
        self.assertEqual(saved["error"], "prompts_unavailable")

    def test_review_context_snapshot(self):
        self.ingest()
        from lab_helpers.s10_shared.reviews.context import ContextCatalog, record_fingerprint
        path = self.root / "context.json"
        context = {"amenities": ["Paid parking"], "bathroom_description": "1 shared bath", "minimum_nights": 31}
        path.write_text(json.dumps({"version": 1, "provenance": {"snapshot": "test"}, "listings": {
            "101": {"record_sha256": record_fingerprint(self.listing.model_dump()), "context": context}}}))
        job_id = self.store.create_job("101", "context-001", load_prompts(), {}, ContextCatalog(path))
        snapshot = self.store.job(job_id)["snapshot"]
        self.assertEqual(snapshot["review_context"]["fields"], context)
        self.assertEqual(snapshot["listing"], self.listing.model_dump())
        path.unlink()
        self.assertEqual(self.store.job(job_id)["snapshot"], snapshot)

    def test_review_failure(self):
        self.ingest()
        job_id = self.job()
        def fail(snapshot, save):
            save("evidence", {"output": {"items": []}})
            raise ExternalCallError("provider body should not be persisted")
        process_review(self.store, job_id, fail)
        job = self.store.job(job_id)
        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["error"], "ExternalCallError")
        self.assertIn("evidence", job["stages"])
        self.assertIsNotNone(self.store.listing("101")["prediction"])

    def test_api_background_review_and_repeated_request(self):
        self.ingest()
        def complete(snapshot, save):
            self.assertEqual(snapshot["instructions"], load_prompts())
            save("evidence", {"output": {"items": []}})
            return {"action": "no_change"}
        with self.client() as client, patch(
            "apps.listing_review.api.helpers.reviews.review", side_effect=complete
        ) as runner:
            first = client.post("/listings/101/reviews", json={"request_id": "background-001"})
            self.assertEqual(first.status_code, 202)
            repeated = client.post("/listings/101/reviews", json={"request_id": "background-001"})
            self.assertEqual(first.json(), repeated.json())
            runner.assert_called_once()
            saved = client.get(first.json()["status_url"]).json()
            self.assertEqual(saved["status"], "completed")
            self.assertIn("evidence", saved["stages"])

    def test_repeated_request_does_not_need_current_configuration(self):
        self.ingest()
        with self.client() as client:
            job_id = self.job("repeat-without-config")
            with patch("apps.listing_review.api.app.load_prompts", side_effect=AssertionError("Must not load prompts")) as prompts, \
                 patch("apps.listing_review.api.app.generation_settings", side_effect=ValueError("invalid settings")) as generation, \
                 patch("apps.listing_review.api.app.BackgroundTasks.add_task") as schedule:
                result = client.post("/listings/101/reviews", json={"request_id": "repeat-without-config"})
                self.assertEqual(result.status_code, 202)
                self.assertEqual(result.json()["job_id"], job_id)
                conflict = client.post("/listings/102/reviews", json={"request_id": "repeat-without-config"})
                self.assertEqual(conflict.status_code, 409)
                prompts.assert_not_called()
                generation.assert_not_called()
                schedule.assert_not_called()
            self.assertEqual(self.store.job(job_id)["status"], "queued")

    def test_invalid_generation_settings_are_a_service_error(self):
        self.ingest()
        with self.client() as client, \
             patch("apps.listing_review.api.app.generation_settings", side_effect=ValueError("private config value")), \
             patch("apps.listing_review.api.app.BackgroundTasks.add_task") as schedule:
            response = client.post("/listings/101/reviews", json={"request_id": "invalid-config"})
            self.assertEqual(response.status_code, 503)
            self.assertNotIn("private config value", response.text)
            self.assertIsNone(self.store.job_for_request("invalid-config"))
            schedule.assert_not_called()
            self.assertEqual(client.post("/predict", json=load_example_listing()).status_code, 200)

    def test_generation_programming_errors_are_not_configuration_errors(self):
        self.ingest()
        with self.client() as client, \
             patch("apps.listing_review.api.app.generation_settings", side_effect=TypeError("programming defect")):
            with self.assertRaises(TypeError):
                client.post("/listings/101/reviews", json={"request_id": "programming-error"})

    def test_background_task_claims_only_its_job(self):
        self.ingest()
        other = self.job("other-request")
        target = self.job("target-request")
        runner = lambda snapshot, save: {"action": "no_change"}
        self.assertTrue(process_review(self.store, target, runner))
        self.assertEqual(self.store.job(other)["status"], "queued")
        self.assertFalse(process_review(self.store, target, runner))

    def test_api_restart_marks_pending_tasks_interrupted(self):
        self.ingest()
        queued = self.job("queued-request")
        running = self.job("running-request")
        self.store.claim_job(running)
        with self.client() as client:
            for job_id in [queued, running]:
                saved = client.get(f"/jobs/{job_id}").json()
                self.assertEqual(saved["status"], "failed")
                self.assertEqual(saved["error"], "interrupted")

    def test_programming_error(self):
        self.ingest()
        job_id = self.job()
        def fail(snapshot, save):
            raise TypeError("a programming defect")
        with self.assertRaises(TypeError):
            process_review(self.store, job_id, fail)

    def test_recovery(self):
        self.ingest()
        job_id = self.job()
        self.store.claim_job(job_id)
        self.assertEqual(self.store.recover_reviews(), 1)
        self.assertEqual(self.store.job(job_id)["error"], "interrupted")
        self.assertFalse(self.store.claim_job(job_id))

    def test_missing_model(self):
        with TestClient(create_app(replace(self.config, model=self.root / "absent"))) as client:
            self.assertEqual(client.get("/health/live").status_code, 200)
            self.assertEqual(client.get("/health/ready").status_code, 503)
            self.assertEqual(client.post("/predict", json=load_example_listing()).status_code, 503)

    def test_corrupt_bundle(self):
        (self.config.model / "model.pkl").write_bytes(b"broken")
        with self.assertRaises(ValueError):
            Predictor(self.config.model)


if __name__ == "__main__":
    unittest.main()
