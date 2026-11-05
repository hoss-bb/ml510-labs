"""User-facing review messages shared with the API client UI."""
REVIEW_ERRORS = {
    'evidence_source': 'The AI cited a source that was not supplied. The review was stopped; retry it.',
    'proposal_support': 'The proposed action did not match its supporting findings. The review was stopped; retry it.',
    'prompt_changed': 'The review instructions changed. Request a new review.',
    'prompts_unavailable': 'This review has no complete saved instructions. Request a new review.',
    'configuration': 'The AI provider configuration is unavailable. Check the API provider configuration.',
    'evidence_format': 'The AI did not return a complete evidence response. Retry the review.',
    'assessment_format': 'The AI did not return a complete assessment. Retry the review.',
    'proposal_format': 'The AI did not return a complete recommendation. Retry the review.',
    'unsupported_excerpt': 'An AI quotation did not match the listing description. The review was stopped; retry to generate fresh evidence.',
    'assessment_citation': 'The assessment cited unknown evidence. The review was stopped; retry it.',
    'proposal_citation': 'The recommendation cited unknown evidence. The review was stopped; retry it.',
    'interrupted': 'The API restarted before this review finished. Request a new attempt.',
    'ExternalCallError': 'The AI response could not be accepted. This older record has no detailed reason. You can retry the review.',
}
