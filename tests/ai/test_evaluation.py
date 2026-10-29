"""Small checks for reusable evaluation records."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd
from ai_lib.evaluation import judge_score, ResultCache, check_answer, paired_scores, save_json, local_evaluation_client


class EvaluationTests(unittest.TestCase):
    def test_citations(self):
        answer = {"answer": "Supported", "source_ids": ["p"], "disposition": "answer"}
        self.assertTrue(all(check_answer(answer, ["p"]).values()))
        self.assertFalse(check_answer(answer, ["q"])["citations"])
        self.assertFalse(check_answer({**answer, "source_ids": []}, ["p"])["citations"])
        self.assertFalse(check_answer({"answer": "bad"}, ["p"])["structure"])

    def test_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = ResultCache(directory)
            calls = []
            def run():
                calls.append(1)
                return {"answer": "ok"}
            a = cache.call({"q": 1}, {"prompt": 1}, run)
            b = cache.call({"q": 1}, {"prompt": 1}, run)
            cache.call({"q": 1}, {"prompt": 2}, run)
            cache.call({"q": 1}, {"prompt": 1}, run, reuse=False)
            self.assertEqual(len(calls), 3)
            self.assertEqual(a["execution"], "live")
            self.assertEqual(b["execution"], "reused")

    def test_error_not_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = ResultCache(directory)
            def fail(): raise TimeoutError("private payload")
            self.assertEqual(cache.call({}, {}, fail)["status"], "error")
            self.assertEqual(cache.call({}, {}, lambda: 7)["value"], 7)

    def test_missing_response(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = ResultCache(directory)
            self.assertEqual(cache.call({}, {}, lambda: None)["status"], "error")
            self.assertEqual(cache.call({}, {}, lambda: {"complete": True})["status"], "ok")

    def test_secret_exclusion(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key-value"}):
            path = Path(directory)/"record.json"
            save_json(path, {"api_key": "test-key-value", "answer": "contains test-key-value"})
            self.assertNotIn("test-key-value", path.read_text())
            self.assertNotIn("api_key", json.loads(path.read_text()))

    def test_alignment(self):
        a = pd.DataFrame([{"case_id": "a", "reference_hash": "r", "score": 0},
                          {"case_id": "b", "reference_hash": "s", "score": 1}])
        b = a.iloc[::-1].copy()
        self.assertEqual(paired_scores(a,b,"score").difference.sum(),0)
        b.loc[b.index[0],"reference_hash"] = "different"
        with self.assertRaisesRegex(ValueError,"identical"):
            paired_scores(a,b,"score")
        with self.assertRaisesRegex(ValueError,"unique"):
            paired_scores(pd.concat([a,a]),a,"score")

    def test_local_client(self):
        client = local_evaluation_client()
        with self.assertRaisesRegex(RuntimeError,"local-only"):
            client.session.get("https://api.smith.langchain.com/info")

    def test_local_model_evaluation(self):
        from langsmith import evaluate
        from langsmith.schemas import Example
        from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
        from langchain_core.messages import AIMessage
        from uuid import uuid4
        from unittest.mock import Mock
        client = local_evaluation_client()
        client.session.request = Mock(side_effect=AssertionError("unexpected hosted request"))
        model = FakeMessagesListChatModel(responses=[AIMessage(content="supported")])
        with tempfile.TemporaryDirectory() as directory:
            cache = ResultCache(directory)
            def target(inputs):
                return cache.call(inputs, {"model":"fake"}, lambda:model.invoke("question").text)
            completed = list(evaluate(target, data=[Example(id=uuid4(),inputs={"question":"q"},outputs={})],
                client=client,upload_results=False,max_concurrency=0,disable_evaluator_tracing=True))
            self.assertEqual(completed[0]["run"].outputs["value"],"supported")
        client.session.request.assert_not_called()

    def test_programming_error_propagates(self):
        with tempfile.TemporaryDirectory() as directory:
            def broken():
                raise NameError("implementation bug")
            with self.assertRaises(NameError):
                ResultCache(directory).call({}, {}, broken)

    def test_incomplete_not_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = ResultCache(directory)
            first = cache.call({}, {}, lambda: {"failure_reason": "MALFORMED_FUNCTION_CALL"})
            self.assertEqual(first["status"], "error")
            self.assertEqual(cache.call({}, {}, lambda: "complete")["value"], "complete")

    def test_binary_judgment(self):
        for value in [True, False, 0, 1, 0.0, 1.0]:
            self.assertEqual(judge_score(value), bool(value))
        for value in [None, "false", "true", 0.5, float("nan"), [], {}]:
            with self.assertRaises(ValueError):
                judge_score(value)
