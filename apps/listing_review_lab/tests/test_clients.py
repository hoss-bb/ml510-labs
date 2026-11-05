"""Client evidence must distinguish completed predictions from successful HTTP exchanges."""
import copy
import io
import json
import unittest
from apps.listing_review_lab.tests.prompt_fixtures import install_prompts
from unittest.mock import Mock, patch
import httpx
from pydantic import ValidationError
from apps.listing_review_lab.clients import predict
from lab_helpers.s10_shared.prediction.schemas import Prediction, PredictionResponse
from apps.listing_review_lab.clients.load import measure
from apps.listing_review.jobs.prompt_eng.helpers.compare_prompts import compare
from lab_helpers.s10_shared.reviews import prompts
from lab_helpers.s10_shared.reviews.errors import ReviewError
from lab_helpers.s10_shared.reviews.chain import review
from apps.listing_review_lab.tests import test_chain

RESULT = {"estimate": 100, "model_version": "v1", "currency": "CAD",
          "status": "ok", "request_id": "r1"}


class ClientTests(unittest.TestCase):
    def setUp(self):
        install_prompts(self)

    def test_prediction_contract_rejects_invalid_values(self):
        changes = [{"estimate": value} for value in [float("nan"), float("inf"), -float("inf")]]
        changes += [{field: value} for field in ["model_version", "request_id"] for value in ["", "   "]]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValidationError):
                PredictionResponse.model_validate({**RESULT, **change})
        for value in [0, -10]:
            parsed = PredictionResponse.model_validate({**RESULT, "estimate": value, "status": "review_required"})
            self.assertEqual(parsed.estimate, value)

    def test_listing_comparison_checks_all_prediction_fields(self):
        from lab_helpers.s10_shared.examples import load_example_listing
        batch = {key: value for key, value in RESULT.items() if key != "request_id"}
        for change in [{}, {"currency": "USD"}, {"status": "review_required"},
                       {"model_version": "other-model"}, {"estimate": 101}]:
            with self.subTest(change=change):
                client = Mock()
                client.get.return_value.json.return_value = {
                    "listing": load_example_listing(), "prediction": {**batch, **change}}
                client.post.return_value.json.return_value = RESULT
                with patch("sys.argv", ["predict", "--listing-id", "101"]), \
                     patch.object(predict.httpx, "Client") as factory, \
                     patch.object(predict, "Predictor") as model, \
                     patch("sys.stdout", new_callable=io.StringIO) as output:
                    factory.return_value.__enter__.return_value = client
                    model.return_value.predict.return_value = [Prediction.model_validate(batch)]
                    if change:
                        with self.assertRaises(SystemExit):
                            predict.main()
                    else:
                        predict.main()
                    self.assertEqual(json.loads(output.getvalue())["matches"], not bool(change))

    def test_load_uses_shared_response_validation(self):
        for change in [{"model_version": " "}, {"request_id": ""}, {"currency": "USD"}]:
            with self.subTest(change=change):
                client = Mock()
                client.post.side_effect = [httpx.Response(200, json=RESULT),
                                          httpx.Response(200, json={**RESULT, **change})]
                self.assertEqual(measure(client, 1, 1)["failure_counts"], {"invalid_response": 1})
        client = Mock()
        client.post.side_effect = [httpx.Response(200, json=RESULT),
                                  httpx.Response(200, json={**RESULT, "status": "review_required"})]
        self.assertEqual(measure(client, 1, 1)["failure_counts"], {"prediction_mismatch": 1})

    def test_prediction_client_validates_api_response(self):
        response = httpx.Response(200, json=RESULT,
                                  request=httpx.Request("POST", "http://test/predict"))
        with patch("sys.argv", ["predict", "--api"]), \
             patch.object(predict.httpx, "post", return_value=response), \
             patch("sys.stdout", new_callable=io.StringIO) as output:
            predict.main()
        self.assertEqual(json.loads(output.getvalue()), RESULT)

    def test_prediction_client_rejects_incomplete_api_response(self):
        response = httpx.Response(200, json={"estimate": 100},
                                  request=httpx.Request("POST", "http://test/predict"))
        with patch("sys.argv", ["predict", "--api"]), \
             patch.object(predict.httpx, "post", return_value=response), \
             patch("sys.stdout", new_callable=io.StringIO) as output:
            with self.assertRaises(ValidationError):
                predict.main()
        self.assertEqual(output.getvalue(), "")

    def test_example_loader_returns_independent_data(self):
        from lab_helpers.s10_shared.examples import load_example_listing
        first, second = load_example_listing(), load_example_listing()
        self.assertEqual(first, second)
        self.assertIsNot(first, second)
        first['accommodates'] = 99
        self.assertEqual(load_example_listing(), second)

    def test_example_loader_rejects_invalid_json_and_features(self):
        from pydantic import ValidationError
        from lab_helpers.s10_shared.examples import load_example_listing
        invalid = {**load_example_listing(), 'accommodates': 0}
        target = 'lab_helpers.s10_shared.examples.Path.read_text'
        with patch(target, return_value=json.dumps(invalid)), self.assertRaises(ValidationError):
            load_example_listing()
        with patch(target, return_value='{broken'), self.assertRaises(json.JSONDecodeError):
            load_example_listing()

    def test_load_classifies_failures_and_checks_results(self):
        responses = [httpx.Response(200, json=RESULT), httpx.Response(200, json=RESULT),
                     httpx.Response(503), httpx.ReadTimeout("private provider details"),
                     httpx.Response(200, json={}), httpx.Response(200, json={**RESULT, "estimate": 101}),
                     httpx.Response(200, json={**RESULT, "model_version": "v2"}),
                     httpx.Response(200, content=b"not json")]
        client = Mock()
        client.post.side_effect = responses
        report = measure(client, 7, 1)
        self.assertEqual(report["successes"], 1)
        self.assertEqual(report["failures"], 6)
        self.assertEqual(report["failure_counts"], {"http:503": 1, "network:ReadTimeout": 1,
            "invalid_response": 2, "prediction_mismatch": 1, "model_mismatch": 1})
        self.assertNotIn("private provider", json.dumps(report))
        self.assertEqual(client.post.call_count, 8)

    def test_failed_reference_does_not_certify_consistency(self):
        client = Mock()
        client.post.side_effect = [httpx.Response(503), httpx.Response(200, json=RESULT)]
        report = measure(client, 1, 1)
        self.assertEqual(report["failure_counts"], {"reference_unavailable": 1})
        self.assertIsNone(report["successful_latency_seconds"])

    def test_prompt_comparison_matches_cases_and_retains_failure(self):
        run, calls = Mock(), []
        original = prompts.load_prompts()
        def runner(snapshot, save, *, instructions):
            calls.append(copy.deepcopy(snapshot))
            save("evidence", {"output": "partial"})
            if len(calls) == 3:
                raise ReviewError("assessment_format")
            return {"action": "confirm_facts" if "more than" in snapshot["listing"]["description"] else "no_change"}
        table = compare(run, {"model": "fixed"}, runner)
        self.assertEqual(len(table), 4)
        self.assertEqual(table["action_matches"].sum(), 3)
        self.assertEqual(run.record.call_count, 4)
        self.assertEqual(calls[0]["listing"], calls[2]["listing"])
        self.assertEqual(calls[1]["listing"], calls[3]["listing"])
        self.assertTrue(all(c["ai_settings"] == {"model": "fixed"} for c in calls))
        self.assertNotEqual(calls[0]["instructions"], calls[2]["instructions"])
        self.assertEqual(original, prompts.load_prompts())
        failed = run.record.call_args_list[2].args[1]
        self.assertEqual(failed["error"], "assessment_format")
        self.assertIn("evidence", failed["stages"])

    def test_stage_logs_identify_job(self):
        snapshot = test_chain.ChainTests().snapshot()
        snapshot["job_id"] = "job-example"
        with self.assertLogs("s10", level="INFO") as logs:
            review(snapshot, lambda *args: None, test_chain.FakeModel())
        records = [json.loads(record.getMessage()) for record in logs.records]
        self.assertEqual(len(records), 3)
        for record in records:
            self.assertEqual(record["job_id"], "job-example")
            self.assertNotIn("prompt_version", record)
            self.assertEqual(record["outcome"], "completed")

    def test_candidate_is_used_without_changing_default(self):
        snapshot = test_chain.ChainTests().snapshot()
        instructions = {stage: "Candidate example" for stage in prompts.STAGES}
        model = test_chain.FakeModel()
        review(snapshot, lambda *args: None, model, instructions=instructions)
        self.assertTrue(all("Candidate example" in p for p in model.prompts))


if __name__ == "__main__":
    unittest.main()
