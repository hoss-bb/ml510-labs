"""Add AI context from the recorded source without changing training or arrival files."""
import json
import pandas as pd
from ml_lib.data.loading import sha256_file
from lab_helpers.s10_shared.reviews.context import record_fingerprint
from lab_helpers.s10_shared.prediction.schemas import Listing
from lab_helpers.s10_shared.reviews.schemas import ReviewContext
from .prepare import SOURCE, SOURCE_SHA256


def prepare_context(data):
    raw = data / 'raw' / 'listings-2026-06-15.csv.gz'
    prepared = data / 'prepared' / 'listings.csv'
    if not raw.exists() or not prepared.exists():
        raise FileNotFoundError('Run the training job to obtain the recorded dataset.')
    if sha256_file(raw) != SOURCE_SHA256:
        raise ValueError('Raw snapshot checksum differs; do not attach unrelated context.')
    source = pd.read_csv(raw, dtype={'id': str}, usecols=['id', 'amenities', 'bathrooms_text', 'minimum_nights']).set_index('id')
    listings = pd.read_csv(prepared, dtype={'listing_id': str})
    entries = {}
    for record in listings.to_dict('records'):
        fields = {field: record[field] for field in Listing.model_fields}
        for key, value in fields.items():
            if pd.isna(value):
                fields[key] = '' if key == 'description' else None
        listing = Listing.model_validate(fields)
        row = source.loc[listing.listing_id]
        amenities = None if pd.isna(row.amenities) else json.loads(row.amenities)
        nights = row.minimum_nights
        nights = int(nights) if pd.notna(nights) and nights >= 1 and float(nights).is_integer() else None
        context = ReviewContext(amenities=amenities,
                                bathroom_description=None if pd.isna(row.bathrooms_text) else row.bathrooms_text,
                                minimum_nights=nights)
        entries[listing.listing_id] = {'record_sha256': record_fingerprint(listing.model_dump()),
                                     'context': context.model_dump()}
    payload = {'version': 1, 'provenance': {'source': SOURCE, 'raw_sha256': SOURCE_SHA256,
               'snapshot': '2026-06-15', 'attribution': 'Inside Airbnb, Montreal', 'license': 'CC BY 4.0'},
               'listings': entries}
    path = data / 'prepared' / 'review_context.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)
    print(f'Prepared AI context for {len(entries)} listings: {path}')
    return path
