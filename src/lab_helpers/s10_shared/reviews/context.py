"""Match optional snapshot context to unchanged listing records."""
import hashlib
import json
from pathlib import Path
from .schemas import ReviewContext


def record_fingerprint(record):
    payload = json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


class ContextCatalog:
    def __init__(self, path: Path):
        self.entries = {}
        self.provenance = None
        if path.exists():
            payload = json.loads(path.read_text(encoding='utf-8'))
            if payload['version'] != 1:
                raise ValueError('Unsupported review context version; regenerate the context file.')
            self.provenance = payload['provenance']
            self.entries = payload['listings']
            for entry in self.entries.values():
                ReviewContext.model_validate(entry['context'])

    def for_listing(self, record):
        entry = self.entries.get(record['listing_id'])
        if entry is None or entry['record_sha256'] != record_fingerprint(record):
            return {'fields': ReviewContext().model_dump(), 'provenance': None}
        return {'fields': ReviewContext.model_validate(entry['context']).model_dump(),
                'provenance': self.provenance}


def evidence_sources(snapshot):
    """The model cites these source IDs; numeric values are compared exactly."""
    sources = {}
    for field in ('description', 'accommodates', 'bedrooms', 'beds', 'bathrooms',
                  'room_type', 'property_type', 'neighbourhood'):
        value = snapshot['listing'].get(field)
        if value is not None and value != '':
            sources['listing.' + field] = str(value)
    context = ReviewContext.model_validate(snapshot.get('review_context', {}).get('fields', {}))
    if context.bathroom_description:
        sources['context.bathroom_description'] = context.bathroom_description
    if context.minimum_nights is not None:
        sources['context.minimum_nights'] = str(context.minimum_nights)
    for index, amenity in enumerate(context.amenities or []):
        sources[f'context.amenities.{index}'] = amenity
    return sources
