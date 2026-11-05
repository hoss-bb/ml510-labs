import unittest
from ai_lib.evaluation.metrics import lexical_scores, ordinal_score, unit_score, judge_disagreements
from ai_lib.guardrails.answers import assessment_gate
from langchain_core.documents import Document


class MetricTests(unittest.TestCase):
    def test_lexical(self):
        self.assertEqual(lexical_scores("a a b", "a b b")["token_f1"], 2/3)
        self.assertTrue(lexical_scores("Hello!", "hello")["exact_match"])
        self.assertIsNone(lexical_scores("", "")["token_f1"])
        self.assertEqual(lexical_scores("", "one")["token_f1"], 0)

    def test_ordinal(self):
        for n in range(4):
            self.assertEqual(ordinal_score(n), n)
        for n in [True, None, "3", 1.5, 4, -1, float("nan")]:
            with self.assertRaises(ValueError):
                ordinal_score(n)

    def test_ranking(self):
        verdicts = [{"id": str(i), "verdict": label} for i, label in enumerate(["yes", "no", "yes"])]
        kw = {"labels": ["yes", "no"], "positive": "yes", "ranked": True}
        self.assertAlmostEqual(unit_score(["0", "1", "2"], verdicts, **kw)["score"], (1+2/3)/2)
        self.assertAlmostEqual(unit_score(["1", "0", "2"], verdicts, **kw)["score"], (.5+2/3)/2)

    def test_strict_support(self):
        verdicts = [{"id": str(i), "verdict": v} for i, v in enumerate(["supported", "contradicted", "not_established"])]
        result = unit_score(["0", "1", "2"], verdicts,
            labels=["supported", "contradicted", "not_established"], positive="supported")
        self.assertEqual(result["score"], 1/3)

    def test_coverage(self):
        for ids, verdicts in [(["a", "a"], []), (["a"], []),
            (["a"], [{"id":"b", "verdict":"yes"}]),
            (["a"], [{"id":"a", "verdict":"yes"}]*2),
            (["a"], [{"id":"a", "verdict":"maybe"}])]:
            with self.assertRaises(ValueError):
                unit_score(ids, verdicts, labels=["yes", "no"], positive="yes")

    def test_empty(self):
        self.assertIsNone(unit_score([], [], labels=["yes"], positive="yes")["score"])
        result = unit_score(["a"], [{"id":"a", "verdict":"no"}], labels=["yes", "no"], positive="yes", ranked=True)
        self.assertIsNone(result["score"])
        self.assertIn("retrieval failure", result["reason"])

    def test_disagreement(self):
        result = judge_disagreements([True, True, False, False], [True, False, True, False])
        self.assertEqual(result["missed_failure_rate"], .5)
        self.assertEqual(result["false_rejection_rate"], .5)
        self.assertIsNone(judge_disagreements([True], [True])["missed_failure_rate"])
        with self.assertRaises(ValueError):
            judge_disagreements([True], [None])

    def test_prerequisites(self):
        case = {"question":"q", "expected_answer":"r"}
        docs = [Document(page_content="text", metadata={"passage_id":"p"})]
        answer = {"answer":"text", "source_ids":["p"], "disposition":"answer"}
        self.assertEqual(assessment_gate(case, answer, None)["verdict"], "ERROR")
        self.assertEqual(assessment_gate({}, answer, docs)["verdict"], "ERROR")
        self.assertEqual(assessment_gate(case, {}, docs)["verdict"], "FAIL")
        self.assertEqual(assessment_gate(case, answer, docs)["verdict"], "PASS")


class EvaluatorTypeTests(unittest.TestCase):
    def test_explicit_score_contract(self):
        from ai_lib.evaluation.schemas import evaluator_schema
        self.assertEqual(evaluator_schema("binary"), "binary-v1")
        self.assertEqual(evaluator_schema("ordinal")["properties"]["score"]["maximum"], 3)
        with self.assertRaises(ValueError):
            evaluator_schema("guess-from-prose")

    def test_cache_separates_score_contracts(self):
        import tempfile
        from ai_lib.experiment.caching import ResultCache
        from ai_lib.evaluation.schemas import evaluator_schema
        with tempfile.TemporaryDirectory() as directory:
            cache = ResultCache(directory)
            binary = cache.call({"answer": "same"}, {"schema": evaluator_schema("binary")}, lambda: {"score": True})
            ordinal = cache.call({"answer": "same"}, {"schema": evaluator_schema("ordinal")}, lambda: {"score": 3})
            reused = cache.call({"answer": "same"}, {"schema": evaluator_schema("ordinal")}, lambda: self.fail("Should reuse"))
            self.assertNotEqual(binary["key"], ordinal["key"])
            self.assertEqual(reused["execution"], "reused")
