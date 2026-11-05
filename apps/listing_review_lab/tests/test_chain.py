"""Structured chain checks without network calls or paid models."""
import unittest
from apps.listing_review_lab.tests.prompt_fixtures import install_prompts
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import AIMessage
from ai_lib.models.errors import ExternalCallError
from lab_helpers.s10_shared.reviews.chain import review
from lab_helpers.s10_shared.reviews.prompts import load_prompts


class FakeModel:
    def __init__(self, excerpt="Sleeps six guests.", missing=False):
        self.excerpt = excerpt
        self.missing = missing
        self.prompts = []

    def with_structured_output(self, schema, include_raw):
        def respond(prompt):
            self.prompts.append(prompt.to_string())
            values = {
                "Evidence": {"items": [
                    {"claim": "Described capacity", "source_id": "listing.description", "excerpt": self.excerpt},
                    {"claim": "Structured capacity", "source_id": "listing.accommodates", "excerpt": "4"}]},
                "Assessment": {"findings": [{"kind": "conflict", "observation": "Capacity differs",
                    "evidence_ids": ["e1", "e2"]}], "summary": "Confirm capacity."},
                "Proposal": {"action": "confirm_facts", "reason": "The supplied capacities differ.",
                    "next_step": "Ask the owner to confirm capacity.", "evidence_ids": ["e1", "e2"],
                    "owner_question": "Is capacity four or six?", "guest_note": None,
                    "confirm_before_use": ["Maximum guest capacity"]},
            }
            return {"parsed": None if self.missing else schema(**values[schema.__name__]),
                    "raw": AIMessage(content="", usage_metadata={"input_tokens": 10, "output_tokens": 10, "total_tokens": 20}),
                    "parsing_error": None}
        return RunnableLambda(respond)


class ChainTests(unittest.TestCase):
    def setUp(self):
        install_prompts(self)

    def snapshot(self):
        return {"listing": {"description": "Sleeps six guests.", "accommodates": 4},
                "prediction": {"estimate": 170}, "instructions": load_prompts(), "ai_settings": {}}

    def test_stages(self):
        stages = {}
        model = FakeModel()
        output = review(self.snapshot(), lambda name, result: stages.update({name: result}), model)
        self.assertEqual(list(stages), ["evidence", "assessment", "proposal"])
        self.assertEqual(output["action"], "confirm_facts")
        self.assertTrue(all("Sleeps six guests." in prompt for prompt in model.prompts))
        self.assertEqual(stages["proposal"]["usage"]["total_tokens"], 20)

    def test_unsupported_excerpt(self):
        with self.assertRaises(ExternalCallError) as caught:
            review(self.snapshot(), lambda *args: None, FakeModel("Invented detail"))
        self.assertEqual(caught.exception.code, "unsupported_excerpt")

    def test_missing_output(self):
        with self.assertRaises(ExternalCallError) as caught:
            review(self.snapshot(), lambda *args: None, FakeModel(missing=True))
        self.assertEqual(caught.exception.code, "evidence_format")

    def test_legacy_job_needs_new_attempt(self):
        snapshot = self.snapshot()
        del snapshot["instructions"]
        snapshot["prompt_version"] = "historical"
        with self.assertRaises(ExternalCallError) as caught:
            review(snapshot, lambda *args: None, FakeModel())
        self.assertEqual(caught.exception.code, "prompts_unavailable")

    def test_saved_instructions_are_used(self):
        snapshot = self.snapshot()
        snapshot["instructions"] = {stage: "Saved instructions" for stage in snapshot["instructions"]}
        model = FakeModel()
        review(snapshot, lambda *args: None, model)
        self.assertTrue(all("Saved instructions" in prompt for prompt in model.prompts))
