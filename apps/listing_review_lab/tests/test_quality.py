"""Action contracts and optional source context, independent of cloud availability."""
import json
from pathlib import Path
import tempfile
import unittest
from apps.listing_review_lab.tests.prompt_fixtures import install_prompts
from pydantic import ValidationError
from lab_helpers.s10_shared.reviews.context import ContextCatalog, evidence_sources, record_fingerprint
from lab_helpers.s10_shared.reviews.schemas import Proposal, Finding, Evidence
from apps.listing_review_lab.tests.test_chain import FakeModel
from apps.listing_review_lab.tests import test_chain
from lab_helpers.s10_shared.reviews.chain import review
from lab_helpers.s10_shared.reviews.errors import ReviewError
from langchain_core.runnables import RunnableLambda


class QualityTests(unittest.TestCase):
    def setUp(self):
        install_prompts(self)

    def proposal(self, **changes):
        return Proposal.model_validate({'action': 'no_change', 'reason': 'Consistent facts.',
            'next_step': 'Leave unchanged.', 'evidence_ids': [], 'owner_question': None,
            'guest_note': None, 'confirm_before_use': [], **changes})

    def test_action_choices(self):
        with self.assertRaises(ValidationError):
            self.proposal(action='change_price')

    def test_evidence_limit_matches_extraction_instructions(self):
        item = {'claim': 'Four guests.', 'source_id': 'listing.accommodates', 'excerpt': '4'}
        self.assertEqual(len(Evidence(items=[item] * 10).items), 10)
        with self.assertRaises(ValidationError):
            Evidence(items=[item] * 11)

    def test_no_change_has_no_draft(self):
        with self.assertRaises(ValidationError):
            self.proposal(owner_question='Anything to add?')

    def test_owner_question_required(self):
        with self.assertRaises(ValidationError):
            self.proposal(action='clarify_condition', evidence_ids=['e1'])

    def test_guest_note_uses_known_facts(self):
        self.proposal(action='guest_note', evidence_ids=['e1'], guest_note='Shared bathroom.')
        with self.assertRaises(ValidationError):
            self.proposal(action='guest_note', evidence_ids=['e1'], guest_note='Parking may cost $20.',
                          confirm_before_use=['Parking price'])

    def test_conflict_needs_two_facts(self):
        with self.assertRaises(ValidationError):
            Finding(kind='conflict', observation='Different capacities', evidence_ids=['e1', 'e1'])

    def test_context_match(self):
        record = {'listing_id': '123', 'description': 'Private room.'}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'context.json'
            self.assertIsNone(ContextCatalog(path).for_listing(record)['provenance'])
            path.write_text(json.dumps({'version': 1, 'provenance': {'snapshot': 'test'}, 'listings': {
                '123': {'record_sha256': record_fingerprint(record),
                        'context': {'amenities': ['Paid parking'], 'minimum_nights': 31, 'bathroom_description': '1 shared bath'}}}}))
            catalog = ContextCatalog(path)
            context = catalog.for_listing(record)
            self.assertEqual(context['fields']['minimum_nights'], 31)
            self.assertIsNone(catalog.for_listing({**record, 'description': 'Entire home'})['provenance'])
            sources = evidence_sources({'listing': record, 'review_context': context})
            self.assertEqual(sources['context.amenities.0'], 'Paid parking')
            self.assertNotIn('prediction', sources)

    def test_unknown_citation(self):
        base = FakeModel()
        class WrongCitation:
            def with_structured_output(self, schema, include_raw):
                chain = base.with_structured_output(schema, include_raw)
                def alter(prompt):
                    result = chain.invoke(prompt)
                    if schema.__name__ == 'Proposal':
                        result['parsed'].evidence_ids = ['e99']
                    return result
                return RunnableLambda(alter)
        with self.assertRaises(ReviewError) as caught:
            review(test_chain.ChainTests().snapshot(), lambda *args: None, WrongCitation())
        self.assertEqual(caught.exception.code, 'proposal_citation')

    def test_unassessed_action(self):
        base = FakeModel()
        class WrongAction:
            def with_structured_output(self, schema, include_raw):
                chain = base.with_structured_output(schema, include_raw)
                def alter(prompt):
                    result = chain.invoke(prompt)
                    if schema.__name__ == 'Proposal':
                        result['parsed'].action = 'clarify_condition'
                    return result
                return RunnableLambda(alter)
        with self.assertRaises(ReviewError) as caught:
            review(test_chain.ChainTests().snapshot(), lambda *args: None, WrongAction())
        self.assertEqual(caught.exception.code, 'proposal_support')


if __name__ == '__main__':
    unittest.main()
