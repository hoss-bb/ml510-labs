"""Inventory queries and manager interactions, without cloud calls."""
from lab_helpers.s10_shared.reviews.prompts import load_prompts
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
import json
import tempfile
import unittest
from apps.listing_review_lab.tests.prompt_fixtures import install_prompts
from unittest.mock import patch
import httpx

from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from apps.listing_review.api.app import create_app
from lab_helpers.s10_shared.prediction.schemas import Listing, Prediction
from lab_helpers.s10_shared.runtime.settings import settings
from lab_helpers.s10_shared.runtime.storage import Store
from lab_helpers.s10_shared.examples import load_example_listing


class FakePredictor:
    version = "test-ui"

    def predict(self, records):
        return [Prediction(estimate=120, model_version=self.version, status="ok") for _ in records]


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        install_prompts(self)
        # Hold tasks pending here; API tests execute them with mocked generation.
        tasks = patch("apps.listing_review.api.app.BackgroundTasks.add_task")
        tasks.start()
        self.addCleanup(tasks.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = Store(Path(self.temp.name) / "state.sqlite")
        self.store.initialize()
        self.store.begin_batch("day_001", "test-checksum")
        rows = []
        for i in range(60):
            listing = Listing(**{**load_example_listing(), "neighbourhood": "East" if i % 2 else "West"},
                              listing_id=str(100 + i), description="Two bedrooms for four guests.", advertised_price=160)
            rows.append((i + 1, listing, Prediction(estimate=120, model_version="test-ui", status="ok")))
        self.store.finish_batch("day_001", rows, [], .1)
        self.config = replace(settings(), database=self.store.path)
        self.client = TestClient(create_app(self.config, FakePredictor()))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def app(self):
        patcher = patch("httpx.Client", return_value=nullcontext(self.client))
        patcher.start()
        self.addCleanup(patcher.stop)
        return AppTest.from_file("apps/listing_review/ui/app.py", default_timeout=15).run()

    def select_first(self, app):
        return next(button for button in app.button if button.key and button.key.startswith("open_listing_")).click().run()

    def tab(self, app, label):
        app.session_state[f"property_tab_{app.session_state['selected_listing']}"] = label
        return app.run()

    def back(self, app):
        return next(button for button in app.button if button.label == "Back to listings").click().run()

    def test_global_filters(self):
        data = self.client.get("/listings/workspace", params={"limit": 5, "offset": 25, "neighbourhood": "East"}).json()
        self.assertEqual(data["summary"]["listings"], 60)
        self.assertEqual(data["total"], 30)
        self.assertEqual(len(data["items"]), 5)
        self.assertTrue(all(row["listing"]["neighbourhood"] == "East" for row in data["items"]))
        self.assertEqual(self.client.get("/listings/workspace", params={"search": "' OR 1=1 --"}).json()["total"], 0)
        self.assertEqual(self.client.get("/listings/workspace", params={"review_status": "invalid"}).status_code, 422)

    def test_latest_attempt(self):
        first = self.store.create_job("100", "attempt-001", load_prompts(), {})
        self.store.finish_job(first, result={"example": True})
        self.store.create_job("100", "attempt-002", load_prompts(), {})
        data = self.store.workspace(review_status="queued")
        self.assertEqual(data["total"], 1)
        self.assertEqual(data["summary"]["in_progress"], 1)
        self.assertEqual(data["summary"]["completed"], 0)
        self.assertEqual(len(self.client.get("/listings/100/reviews").json()), 2)

    def test_navigation(self):
        app = self.select_first(self.app())
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["selected_listing"], "100")
        self.tab(app, "Price scenario")
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["property_tab_100"], "Price scenario")
        next(button for button in app.button if button.label == "Calculate estimate").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any(metric.value == "$120 CAD" for metric in app.metric))
        self.assertEqual(self.store.listing("100")["listing"]["advertised_price"], 160)

    def test_review_reruns(self):
        app = self.tab(self.select_first(self.app()), "Quality review")
        next(button for button in app.button if button.label == "Generate review").click().run()
        self.assertFalse(app.exception)
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(len(self.store.query("SELECT * FROM jobs")), 1)
        self.assertTrue(next(button for button in app.button if button.label == "Generate another review").disabled)
        job_id = self.store.query("SELECT id FROM jobs")[0]["id"]
        self.store.finish_job(job_id, error="ExternalCallError")
        app.run()
        self.assertFalse(app.exception)
        next(button for button in app.button if button.label == "Retry review").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(self.store.query("SELECT * FROM jobs")), 2)

    def test_review_remains_visible_when_status_filter_no_longer_matches(self):
        app = self.app()
        app.selectbox(key="filter_status").select("not_requested").run()
        app.text_input(key="filter_search").set_value("100").run()
        app = self.tab(self.select_first(app), "Quality review")
        next(button for button in app.button if button.label == "Generate review").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(app.session_state["listing_detail_open"])
        self.assertTrue(any("Waiting for the background" in message.value for message in app.info))
        app.run()
        self.assertEqual(len(self.store.query("SELECT * FROM jobs")), 1)

    def test_completed_review(self):
        job_id = self.store.create_job("100", "attempt-001", load_prompts(), {})
        snapshot = self.store.job(job_id)["snapshot"]
        snapshot.pop("instructions")
        snapshot["prompt_version"] = "historical-version"
        with self.store.connection() as db:
            db.execute("UPDATE jobs SET snapshot=? WHERE id=?", (json.dumps(snapshot), job_id))
        self.store.save_stage(job_id, "evidence", {"seconds": 1, "output": {"items": [
            {"claim": "Capacity is stated", "excerpt": "four guests", "structured_field": "accommodates"}]}})
        self.store.finish_job(job_id, result={"review_status": "clarification_needed", "recommended_action": "Verify capacity",
            "justification": "Check the supplied facts.", "evidence_indices": [0], "questions": ["Is capacity confirmed?"],
            "pricing_caveat": "The estimate is not an optimal price.", "owner_message": "Please confirm capacity."})
        app = self.tab(self.select_first(self.app()), "Quality review")
        self.assertFalse(app.exception)
        self.assertTrue(any(heading.value == "Verify capacity" for heading in app.subheader))
        self.assertTrue(any("Historical prompt version: historical-version" in caption.value
                            for caption in app.caption))
        self.assertEqual(len(app.get("json")), 0)

    def test_filter_resets_selection(self):
        app = self.back(self.select_first(self.app()))
        app.selectbox(key="filter_neighbourhood").select("East").run()
        self.assertFalse(app.exception)
        self.assertIsNone(app.session_state["selected_listing"])
        self.assertEqual(len([b for b in app.button if b.key and b.key.startswith("open_listing_")]), 5)

    def test_quality_review(self):
        job_id = self.store.create_job("100", "quality-001", load_prompts(), {})
        self.store.save_stage(job_id, "evidence", {"seconds": 1, "output": {"items": [
            {"id": "e1", "claim": "Capacity stated", "source_id": "listing.description", "excerpt": "four guests"}]}})
        self.store.finish_job(job_id, result={"output_version": 2, "action": "guest_note",
            "reason": "Keep the supplied condition visible.", "next_step": "Review the optional note.",
            "evidence_ids": ["e1"], "owner_question": None, "guest_note": "Maximum four guests.",
            "confirm_before_use": []})
        app = self.tab(self.select_first(self.app()), "Quality review")
        self.assertFalse(app.exception)
        self.assertTrue(any(title.value == "Prepare a guest-information note" for title in app.subheader))
        app.text_area(key=f"quality_draft_{job_id}").set_value("Edited wording for inspection.").run()
        self.assertFalse(app.exception)
        self.tab(app, "Overview")
        self.tab(app, "Quality review")
        self.assertEqual(app.text_area(key=f"quality_draft_{job_id}").value, "Edited wording for inspection.")
        self.assertEqual(self.store.job(job_id)["result"]["guest_note"], "Maximum four guests.")
        self.assertEqual(len(app.get("json")), 0)

    def test_no_change_review(self):
        job_id = self.store.create_job("100", "quality-001", load_prompts(), {})
        self.store.save_stage(job_id, "evidence", {"seconds": 1, "output": {"items": []}})
        self.store.finish_job(job_id, result={"output_version": 2, "action": "no_change",
            "reason": "No useful change supported.", "next_step": "Leave unchanged.",
            "evidence_ids": [], "owner_question": None, "guest_note": None, "confirm_before_use": []})
        app = self.tab(self.select_first(self.app()), "Quality review")
        self.assertFalse(app.exception)
        self.assertEqual(len(app.text_area), 0)

    def test_arrivals(self):
        app = self.app()
        app.radio(key="view").set_value("Arrivals").run()
        self.assertFalse(app.exception)
        self.assertTrue(app.success)

    def test_navigation_preserves_filters(self):
        app = self.app()
        app.selectbox(key="filter_neighbourhood").select("East").run()
        app.radio(key="view").set_value("Arrivals").run()
        app.radio(key="view").set_value("Listings").run()
        self.assertFalse(app.exception)
        self.assertEqual(app.selectbox(key="filter_neighbourhood").value, "East")
        self.assertEqual(app.session_state["inventory_filters"]["neighbourhood"], "East")

    def test_page_reset(self):
        app = self.back(self.select_first(self.app()))
        next(button for button in app.button if button.label == "Next").click().run()
        self.assertFalse(app.exception)
        self.assertIsNone(app.session_state["selected_listing"])
        next(button for button in app.button if button.label == "Previous").click().run()
        self.assertIsNone(app.session_state["selected_listing"])

    def test_empty_inventory(self):
        with self.store.connection() as db:
            db.execute("DELETE FROM listings")
        app = self.app()
        self.assertFalse(app.exception)
        self.assertTrue(any("No arrivals yet" in message.value for message in app.info))

    def test_unavailable_service(self):
        app = self.app()
        with patch.object(self.client, "request", side_effect=httpx.ConnectError("unreachable")):
            app.run()
        self.assertFalse(app.exception)
        self.assertTrue(app.error)
        self.assertFalse(any("No arrivals" in message.value for message in app.info))

    def test_nonpositive_estimate(self):
        with self.store.connection() as db:
            db.execute("UPDATE listings SET prediction=json_set(prediction, '$.estimate', -10) WHERE id='100'")
        app = self.select_first(self.app())
        self.assertFalse(app.exception)
        self.assertTrue(any(metric.value == "Requires review" for metric in app.metric))
        self.assertTrue(app.warning)

    def test_back_restores_page_without_reopening(self):
        app = self.app()
        next(b for b in app.button if b.label == "Next").click().run()
        self.select_first(app)
        self.assertEqual(app.session_state["selected_listing"], "105")
        self.back(app).run()
        self.assertFalse(app.exception)
        self.assertFalse(app.session_state["listing_detail_open"])
        self.assertEqual(app.session_state["inventory_page"], 1)
        app.session_state["inventory_page"] = 10
        app.run()
        next(b for b in app.button if b.label == "Next").click().run()
        self.assertTrue(next(b for b in app.button if b.label == "Next").disabled)
        self.assertEqual(len([b for b in app.button if b.key and b.key.startswith("open_listing_")]), 5)

    def test_clear_combined_filters(self):
        app = self.app()
        self.assertEqual(app.selectbox(key="filter_status").value, "")
        app.selectbox(key="filter_neighbourhood").select("East").run()
        app.text_input(key="filter_search").set_value("101").run()
        self.assertEqual(len([b for b in app.button if b.key and b.key.startswith("open_listing_")]), 1)
        next(b for b in app.button if b.label == "Clear filters").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["inventory_filters"], {})
        self.assertEqual(len([b for b in app.button if b.key and b.key.startswith("open_listing_")]), 5)

    def test_scenario_persists_and_isolates_properties(self):
        before = self.store.listing("100")
        app = self.tab(self.select_first(self.app()), "Price scenario")
        app.number_input(key="scenario_input_100_accommodates").set_value(7)
        next(b for b in app.button if b.label == "Calculate estimate").click().run()
        app.number_input(key="scenario_input_100_accommodates").set_value(8).run()
        self.tab(app, "Overview")
        self.tab(app, "Price scenario")
        self.assertEqual(app.number_input(key="scenario_input_100_accommodates").value, 8)
        self.assertEqual(app.session_state["scenario_result_100"]["inputs"]["accommodates"], 7)
        self.assertFalse(any("Session State API" in warning.value for warning in app.warning))
        self.back(app)
        app.text_input(key="filter_search").set_value("101").run()
        self.tab(self.select_first(app), "Price scenario")
        self.assertEqual(app.number_input(key="scenario_input_101_accommodates").value, before["listing"]["accommodates"])
        self.assertNotIn("scenario_result_101", app.session_state.filtered_state)
        self.assertEqual(self.store.listing("100"), before)
        self.assertFalse(self.store.query("SELECT * FROM jobs"))

    def test_owner_outcomes_and_grouped_evidence(self):
        for action, heading in [("confirm_facts", "Confirm conflicting facts"),
                                ("clarify_condition", "Clarify an important condition")]:
            with self.subTest(action=action):
                job_id = self.store.create_job("100", action, load_prompts(), {})
                self.store.save_stage(job_id, "evidence", {"seconds": 1, "output": {"items": [
                    {"id": "e1", "claim": "Capacity stated", "source_id": "listing.description", "excerpt": "four guests"},
                    {"id": "e2", "claim": "Guest limit stated", "source_id": "listing.description", "excerpt": "four guests"}]}})
                self.store.finish_job(job_id, result={"output_version": 2, "action": action,
                    "reason": "Inspect the supplied facts.", "next_step": "Ask the owner.",
                    "evidence_ids": ["e1", "e2"], "owner_question": "Please confirm capacity.",
                    "guest_note": None, "confirm_before_use": ["Capacity"]})
                app = self.tab(self.select_first(self.app()), "Quality review")
                self.assertFalse(app.exception)
                self.assertTrue(any(title.value == heading for title in app.subheader))
                # The historical panel may show another job; inspect the latest panel first.
                self.assertEqual(app.text_area[0].value, "Please confirm capacity.")
                self.assertEqual(app.text[0].value, "four guests")
                self.assertTrue(any("e1 · Capacity stated" in text.value for text in app.markdown))
                self.assertTrue(any("e2 · Guest limit stated" in text.value for text in app.markdown))

    def test_inventory_distinguishes_similar_properties(self):
        app = self.app()
        app.selectbox(key="filter_neighbourhood").select("West").run()
        buttons = [b for b in app.button if b.key and b.key.startswith("open_listing_")]
        self.assertEqual(len({b.key for b in buttons}), len(buttons))
        self.assertEqual(buttons[0].key, "open_listing_100")
        self.assertEqual(buttons[0].label, "")
        self.assertTrue(any("bedrooms" in c.value for c in app.caption))
        self.select_first(app)
        self.assertEqual(app.session_state["selected_listing"], "100")

    def test_sidebar_scenario_starts_empty_and_stays_independent(self):
        before = self.store.listing("100")
        app = self.tab(self.select_first(self.app()), "Price scenario")
        app.number_input(key="scenario_input_100_accommodates").set_value(7).run()
        app.radio(key="view").set_value("Price scenario").run()
        self.assertFalse(app.exception)
        self.assertIsNone(app.number_input(key="scenario_input_custom_accommodates").value)
        self.assertEqual(app.text_input(key="scenario_input_custom_neighbourhood").value, "")
        self.assertEqual(app.text_input(key="scenario_input_custom_property_type").value, "")
        self.assertIsNone(app.selectbox(key="scenario_input_custom_room_type").value)
        self.assertTrue(all(value.value is None for value in app.number_input))
        self.assertFalse(any(b.label.startswith("Back to") for b in app.button))
        self.assertTrue(next(b for b in app.button if b.label == "Calculate estimate").disabled)
        app.number_input(key="scenario_input_custom_accommodates").set_value(3).run()
        app.text_input(key="scenario_input_custom_neighbourhood").set_value("West").run()
        app.text_input(key="scenario_input_custom_property_type").set_value("Entire rental unit").run()
        app.selectbox(key="scenario_input_custom_room_type").select("Entire home/apt").run()
        next(b for b in app.button if b.label == "Calculate estimate").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["scenario_result_custom"]["inputs"]["accommodates"], 3)
        self.assertIsNone(app.session_state["scenario_result_custom"]["inputs"]["bedrooms"])
        self.assertFalse(any("compared with the original" in c.value for c in app.caption))
        app.radio(key="view").set_value("Listings").run()
        self.assertEqual(app.number_input(key="scenario_input_100_accommodates").value, 7)
        app.radio(key="view").set_value("Price scenario").run()
        self.assertEqual(app.number_input(key="scenario_input_custom_accommodates").value, 3)
        self.assertEqual(self.store.listing("100"), before)


if __name__ == "__main__":
    unittest.main()
