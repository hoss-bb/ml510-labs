"""One-off job preparation and reuse, using a tiny local snapshot and mocked fitting."""
from dataclasses import replace
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import pandas as pd
from ml_lib.data.loading import sha256_file
from lab_helpers.s10_shared.examples import load_example_listing
from lab_helpers.s10_shared.runtime.settings import settings
from apps.listing_review.jobs.ml_training.helpers import workflow as job, context, prepare


class TrainingJobTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.data, self.sim = root/'data', root/'simulator'
        raw = self.data/'raw/listings-2026-06-15.csv.gz'
        raw.parent.mkdir(parents=True)
        rows = [{**load_example_listing(), 'id': str(i+1), 'host_id': str(i//2+1), 'price': '$100.00',
                 'neighbourhood_cleansed': load_example_listing()['neighbourhood'], 'description': 'Example',
                 'amenities': '[]', 'bathrooms_text': '1 private bath', 'minimum_nights': 2} for i in range(40)]
        pd.DataFrame(rows).to_csv(raw,index=False,compression='gzip')
        for module in [prepare, context]:
            replacement = patch.object(module,'SOURCE_SHA256',sha256_file(raw))
            replacement.start(); self.addCleanup(replacement.stop)

    def fingerprints(self):
        return {str(p):sha256_file(p) for folder in [self.data,self.sim] for p in folder.rglob('*') if p.is_file()}

    def test_first_run_reuse_and_missing_support(self):
        job.ensure_prepared(self.data,self.sim)
        before=self.fingerprints()
        job.ensure_prepared(self.data,self.sim)
        self.assertEqual(before,self.fingerprints())
        (self.data/'prepared/review_context.json').unlink()
        (self.sim/'arrival_examples/day_001.csv').unlink()
        incoming=self.sim/'incoming';incoming.mkdir()
        (incoming/'day_001.csv').write_text('published file remains untouched')
        job.ensure_prepared(self.data,self.sim)
        self.assertTrue((self.data/'prepared/review_context.json').exists())
        self.assertTrue((self.sim/'arrival_examples/day_001.csv').exists())
        self.assertEqual((incoming/'day_001.csv').read_text(),'published file remains untouched')

    def test_changed_data_rejected(self):
        job.ensure_prepared(self.data,self.sim)
        p=self.data/'prepared/listings.csv';p.write_text(p.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'fingerprint'):
            job.ensure_prepared(self.data,self.sim)

    def test_partial_preparation_rejected(self):
        (self.data/'prepared').mkdir()
        with self.assertRaisesRegex(ValueError,'Incomplete preparation'):
            job.ensure_prepared(self.data,self.sim)

    def test_changed_arrival_not_overwritten(self):
        job.ensure_prepared(self.data,self.sim)
        p=self.sim/'arrival_examples/day_001.csv'
        text=p.read_text().replace('100.0','999.0');p.write_text(text)
        with self.assertRaisesRegex(ValueError,'differs'):
            job.ensure_prepared(self.data,self.sim)
        self.assertEqual(p.read_text(),text)

    def test_new_candidate_each_time_and_no_deployment(self):
        config=replace(settings(),data=self.data,model=self.data/'deployed')
        with patch.object(job,'settings',return_value=config), patch.object(job,'ensure_prepared'), patch.object(job,'train',return_value=Path('evidence')) as train:
            first,_=job.run_job(1)
            second,_=job.run_job(10)
        self.assertNotEqual(first,second)
        self.assertEqual(first.parent,self.data/'candidates')
        self.assertEqual(train.call_args_list[0].kwargs['output_directory'],first)
        self.assertFalse(config.model.exists())


if __name__ == '__main__':
    unittest.main()
