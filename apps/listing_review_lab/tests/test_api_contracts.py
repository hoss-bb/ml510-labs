"""Public API contracts, safe errors and complete traversal of bounded collections."""
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
import json
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from apps.listing_review.api.app import create_app
from apps.listing_review.ui.helpers.api_client import ApiClient
from apps.listing_review.ui.helpers.common import collection
from lab_helpers.s10_shared.runtime.settings import settings
from lab_helpers.s10_shared.runtime.storage import Store
from lab_helpers.s10_shared.prediction.schemas import Listing, Prediction
from lab_helpers.s10_shared.examples import load_example_listing


class Predictor:
    version = "contract-test"

    def predict(self, records):
        return [Prediction(estimate=120, model_version=self.version, status="ok") for _ in records]


class ApiContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.config = replace(settings(), data=root, database=root / "state.sqlite")
        self.store = Store(self.config.database)
        self.store.initialize()
        self.store.begin_batch("day_001", "checksum")
        listing = Listing(**load_example_listing(), listing_id="101", advertised_price=160)
        self.store.finish_batch("day_001", [(1, listing, Predictor().predict([listing])[0])], [], .1)
        self.predictor = Predictor()
        self.client = TestClient(create_app(self.config, self.predictor), raise_server_exceptions=False)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def assert_error(self, response, status, code):
        self.assertEqual(response.status_code, status, response.text)
        body = response.json()
        self.assertEqual(set(body), {"code", "message", "request_id", "details"})
        self.assertEqual(body["code"], code)
        self.assertEqual(body["request_id"], response.headers["X-Request-ID"])
        self.assertTrue(body["request_id"])
        return body

    def test_missing_parent_is_not_an_empty_collection(self):
        for url in ["/listings/999", "/listings/999/reviews", "/batches/missing/rejections", "/jobs/missing"]:
            with self.subTest(url=url):
                self.assert_error(self.client.get(url), 404, "not_found")
        for url in ["/listings/101/reviews", "/batches/day_001/rejections"]:
            response = self.client.get(url)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), [])
        with patch("apps.listing_review.api.app.load_prompts", side_effect=AssertionError("must not load")):
            self.assert_error(self.client.post("/listings/999/reviews", json={"request_id": "missing-listing"}), 404, "not_found")

    def test_errors_are_safe_and_traceable(self):
        response = self.client.post("/predict", json={**load_example_listing(), "accommodates": "private-input"})
        body = self.assert_error(response, 422, "validation_error")
        self.assertEqual(body["details"][0]["field"], "body.accommodates")
        self.assertNotIn("private-input", response.text)
        with patch.object(self.predictor, "predict", side_effect=RuntimeError("private-internal")):
            response = self.client.post("/predict", json=load_example_listing())
            self.assert_error(response, 500, "internal_error")
            self.assertNotIn("private-internal", response.text)
        with patch.object(Store, "workspace", side_effect=sqlite3.OperationalError("private-path")):
            response = self.client.get("/listings/workspace")
            self.assert_error(response, 503, "unavailable")
            self.assertNotIn("private-path", response.text)
        self.assert_error(self.client.put("/predict"), 405, "method_not_allowed")

    def test_response_schema_rejects_invalid_server_output(self):
        with patch.object(Store, "workspace", return_value={"unexpected": "private-output"}):
            response = self.client.get("/listings/workspace")
            self.assert_error(response, 500, "internal_error")
            self.assertNotIn("private-output", response.text)

    def seed_history(self):
        with self.store.connection() as db:
            for i in range(260):
                db.execute("INSERT INTO batches(id,checksum,status) VALUES(?,?,'completed')", (f"extra_{i:03}", "sum"))
                db.execute("INSERT INTO rejections VALUES('day_001', ?, '101', 'invalid_input')", (i+1,))
                db.execute("INSERT INTO jobs(id,request_id,listing_id,status,snapshot) VALUES(?,?,'101','failed','{}')", (f"job_{i:03}", f"request_{i:03}"))

    def test_pagination_is_bounded_ordered_and_complete(self):
        self.seed_history()
        for url, key, count in [("/batches", "id", 261), ("/batches/day_001/rejections", "row_number", 260),
                                 ("/listings/101/reviews", "id", 260)]:
            with self.subTest(url=url):
                self.assertEqual(len(self.client.get(url).json()), 50)
                first = self.client.get(url, params={"limit":250}).json()
                second = self.client.get(url, params={"limit":250,"offset":250}).json()
                self.assertEqual(len(first), 250)
                self.assertEqual(len(first+second), count)
                self.assertEqual(len({r[key] for r in first+second}), count)
                self.assertEqual(self.client.get(url, params={"offset":count}).json(), [])
                for params in [{"offset":-1},{"limit":0},{"limit":251}]:
                    self.assert_error(self.client.get(url, params=params),422,"validation_error")
                with patch("httpx.Client", return_value=nullcontext(self.client)):
                    self.assertEqual(collection(ApiClient("http://testserver"), url), first+second)
        self.assertEqual(self.client.get("/listings/101/reviews", params={"limit":1}).json()[0]["id"], "job_259")

    def test_collection_does_not_display_partial_success(self):
        with patch("apps.listing_review.ui.helpers.common.request", side_effect=[[{}]*250, None]):
            self.assertIsNone(collection(None, "/batches"))

    def test_job_location_idempotency_and_historical_payload(self):
        with patch("apps.listing_review.api.app.load_prompts", return_value={"evidence":"e","assessment":"a","proposal":"p"}), \
             patch("apps.listing_review.api.app.generation_settings", return_value={"provider":"mock"}), \
             patch("apps.listing_review.api.app.BackgroundTasks.add_task") as task:
            first = self.client.post("/listings/101/reviews", json={"request_id":"contract-attempt"})
            second = self.client.post("/listings/101/reviews", json={"request_id":"contract-attempt"})
            self.assertEqual(first.status_code,202)
            self.assertEqual(first.json(),second.json())
            self.assertEqual(first.headers["Location"],first.json()["status_url"])
            self.assertEqual(second.headers["Location"],first.headers["Location"])
            self.assertEqual(task.call_count,1)
            self.assert_error(self.client.post("/listings/999/reviews",json={"request_id":"contract-attempt"}),409,"conflict")
        job_id=first.json()["job_id"]
        legacy={"action":"clarify","owner_message":"Historical wording","evidence_indices":[]}
        self.store.finish_job(job_id,result=legacy)
        job=self.client.get(first.headers["Location"]).json()
        self.assertEqual(job["result"],legacy)
        self.assertEqual(job["snapshot"]["instructions"]["evidence"],"e")
        self.assertEqual(self.store.job(job_id)["result"],legacy)

    def test_openapi_documents_success_and_error_contracts(self):
        schema=self.client.get("/openapi.json").json()
        for path, methods in schema["paths"].items():
            for method, operation in methods.items():
                with self.subTest(path=path,method=method):
                    self.assertTrue(operation["tags"])
                    self.assertTrue(operation["description"])
                    success=operation["responses"].get("200",operation["responses"].get("202"))
                    self.assertTrue(success["content"]["application/json"]["schema"])
                    for status in ["404","409","422","500","503"]:
                        self.assertEqual(operation["responses"][status]["content"]["application/json"]["schema"],
                                         {"$ref":"#/components/schemas/ErrorResponse"})
