"""Checks for the extracted HTTP, storage and review boundaries."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

from apps.listing_review.ui.helpers.api_client import ApiClient, ServiceError
from lab_helpers.s10_shared.runtime.storage import Store
from lab_helpers.s10_shared.reviews.schemas import Evidence, Assessment, Proposal
from lab_helpers.s10_shared.reviews.validation import validate_stage
from lab_helpers.s10_shared.reviews.errors import ReviewError


class BoundaryTests(unittest.TestCase):
    def request(self, handler):
        client = httpx.Client(base_url="http://fixture", transport=httpx.MockTransport(handler))
        with patch("apps.listing_review.ui.helpers.api_client.httpx.Client", return_value=client):
            return ApiClient("http://fixture").request("GET", "/listings")

    def test_http_empty_success_is_not_failure(self):
        self.assertEqual(self.request(lambda _: httpx.Response(200, json=[])), [])

    def test_http_errors_keep_reference_without_response_body(self):
        with self.assertRaises(ServiceError) as caught:
            self.request(lambda _: httpx.Response(503, text="private body", headers={"X-Request-ID": "abc"}))
        self.assertEqual(caught.exception.request_id, "abc")
        self.assertNotIn("private body", str(caught.exception))

    def test_http_network_and_invalid_json(self):
        def unavailable(request):
            raise httpx.ConnectError("private network detail", request=request)
        for handler in [unavailable, lambda _: httpx.Response(200, text="invalid JSON")]:
            with self.subTest(handler=handler), self.assertRaises(ServiceError):
                self.request(handler)

    def test_named_queries_preserve_order_and_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "db.sqlite")
            store.initialize()
            store.check_ready()
            self.assertIsNone(store.batch("missing"))
            store.begin_batch("b", "checksum")
            with store.connection() as db:
                for identifier in ["3", "1", "2"]:
                    db.execute("INSERT INTO listings VALUES (?, 'b', ?, ?)",
                               (identifier, '{"listing_id":"' + identifier + '"}', '{"estimate":100}'))
            with patch.object(store, "listing", side_effect=AssertionError("Per-row lookup")):
                rows = store.listings(offset=1, limit=1)
            self.assertEqual(rows, [{"batch_id": "b", "listing": {"listing_id": "2"},
                                     "prediction": {"estimate": 100}}])
            self.assertEqual(store.batches()[0]["id"], "b")
            self.assertEqual(store.batch("b")["checksum"], "checksum")
            self.assertEqual(store.rejections("b' OR 1=1 --"), [])
            self.assertEqual(store.listing_reviews("missing"), [])

    def test_evidence_and_citation_validation(self):
        evidence = Evidence(items=[{"claim": "Shared bath", "source_id": "listing.description",
                                   "excerpt": "shared bath"}])
        sources = {"listing.description": "Room with shared bath."}
        checked = validate_stage("evidence", evidence, sources, {})
        self.assertEqual(checked["items"][0]["id"], "e1")
        with self.assertRaises(ReviewError) as caught:
            validate_stage("evidence", evidence, {"listing.description": "Private bath"}, {})
        self.assertEqual(caught.exception.code, "unsupported_excerpt")
        assessment = Assessment(summary="Check", findings=[{
            "kind": "guest_condition", "observation": "Shared bath", "evidence_ids": ["e99"]}])
        with self.assertRaises(ReviewError) as caught:
            validate_stage("assessment", assessment, sources, {"evidence": checked})
        self.assertEqual(caught.exception.code, "assessment_citation")

    def test_proposal_cannot_skip_required_confirmation(self):
        proposal = Proposal(action="no_change", reason="No change", next_step="Keep",
                            evidence_ids=[], owner_question=None, guest_note=None, confirm_before_use=[])
        outputs = {"evidence": {"items": []}, "assessment": {"findings": []}}
        self.assertEqual(validate_stage("proposal", proposal, {}, outputs)["action"], "no_change")
        outputs["assessment"]["findings"] = [{"kind": "conflict"}]
        with self.assertRaises(ReviewError) as caught:
            validate_stage("proposal", proposal, {}, outputs)
        self.assertEqual(caught.exception.code, "proposal_support")
