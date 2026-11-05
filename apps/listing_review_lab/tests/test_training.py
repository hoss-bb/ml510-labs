"""Check the experiment boundary with small synthetic listings."""
import unittest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch
from ml_lib.data.loading import sha256_file
from apps.listing_review.jobs.ml_training.helpers import prepare as preparation
import numpy as np
import pandas as pd
from lab_helpers.s10_shared.examples import load_example_listing
from lab_helpers.s10_shared.evaluation import check_partitions, group_comparisons
from apps.listing_review.jobs.ml_training.helpers.train import compare_models


def fixture():
    rows = []
    for split, size in [('train', 40), ('validation', 12), ('test', 12)]:
        for i in range(size):
            capacity = i % 6 + 1
            rows.append({**load_example_listing(), 'listing_id': f'{split}-{i}', 'host_id': f'{split}-host-{i // 2}',
                         'split': split, 'accommodates': capacity,
                         'bedrooms': np.nan if i % 4 == 0 else float(i % 3 + 1),
                         'room_type': 'Private room' if i % 2 else 'Entire home/apt',
                         'neighbourhood': 'Known' if split == 'train' else 'Unseen',
                         'advertised_price': 30 + 25 * capacity})
    return pd.DataFrame(rows)


class TrainingTests(unittest.TestCase):
    def test_preparation_exports_only_held_out_arrivals(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data, simulator = root / 'review', root / 'simulator'
            raw = data / 'raw/listings-2026-06-15.csv.gz'
            raw.parent.mkdir(parents=True)
            rows = [{**load_example_listing(), 'id': str(i+1), 'host_id': str(i//2+1), 'price': '$100.00',
                     'neighbourhood_cleansed': load_example_listing()['neighbourhood'], 'description': 'Example'} for i in range(40)]
            pd.DataFrame(rows).to_csv(raw, index=False, compression='gzip')
            with patch.dict(os.environ, {'LISTINGS_SIMULATOR_DATA': str(simulator)}), patch.object(preparation, 'SOURCE_SHA256', sha256_file(raw)):
                preparation.prepare(data)
                prepared = pd.read_csv(data / 'prepared/listings.csv', dtype={'listing_id': str})
                batches = pd.concat([pd.read_csv(p, dtype={'listing_id': str}) for p in (simulator/'arrival_examples').glob('*.csv')])
                self.assertEqual(set(batches.listing_id), set(prepared.query("split == 'test'").listing_id))
                self.assertNotIn('host_id', batches)
                self.assertNotIn('split', batches)
                with self.assertRaisesRegex(FileExistsError, 'Arrival examples already exist'):
                    preparation.prepare(root / 'another-preparation')

    def test_partition_checks(self):
        data = fixture()
        for kind in ['empty', 'duplicate', 'overlap', 'unknown', 'missing']:
            changed = data.copy()
            if kind == 'empty': changed = changed[changed.split != 'test']
            if kind == 'duplicate': changed.loc[1, 'listing_id'] = changed.loc[0, 'listing_id']
            if kind == 'overlap': changed.loc[changed.split == 'test', 'host_id'] = changed.loc[0, 'host_id']
            if kind == 'unknown': changed.loc[0, 'split'] = 'other'
            if kind == 'missing': changed.loc[0, 'host_id'] = None
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                check_partitions(changed)

    def test_training_boundary(self):
        data = fixture()
        first = compare_models(data)
        changed = data.copy()
        changed.loc[changed.split == 'test', 'advertised_price'] *= 100
        second = compare_models(changed)
        self.assertEqual(first['selection'], second['selection'])
        np.testing.assert_allclose(first['predictions'].prediction, second['predictions'].prediction)
        self.assertNotEqual(first['metrics'].query("split == 'test'").mae.tolist(),
                            second['metrics'].query("split == 'test'").mae.tolist())
        model = first['models']['ridge']
        training = data[data.split == 'train']
        numeric = list(model.schema.numeric)
        pipeline = model.pipeline.named_steps['prepare']
        numeric_steps = pipeline.named_transformers_['numeric']
        np.testing.assert_allclose(numeric_steps.named_steps['impute'].statistics_, training[numeric].median())
        np.testing.assert_allclose(numeric_steps.named_steps['scale'].mean_, training[numeric].fillna(training[numeric].median()).mean())
        categories = pipeline.named_transformers_['category'].named_steps['encode'].categories_
        self.assertEqual(set(categories[-1]), {'Known'})
        test = first['metrics'].query("split == 'test'")
        self.assertEqual(set(test.model), {'median', first['selection']['selected_model']})
        self.assertFalse(test.model.duplicated().any())
        for _, group in first['group_coverage'].groupby(['split', 'view']):
            self.assertEqual(group.n.sum(), len(data[data.split == group.iloc[0]['split']]))

    def test_group_baseline(self):
        part = fixture().query("split == 'validation'").copy()
        # Explicitly different from every group's target median.
        baseline = pd.Series(999., index=pd.Index(part.listing_id))
        predictions = {'median': baseline, 'candidate': pd.Series(part.advertised_price.to_numpy(), index=baseline.index)}
        table = group_comparisons(part, predictions, split='validation')
        for _, row in table[table.model == 'candidate'].iterrows():
            if row.n:
                self.assertEqual(row.mae, 0)
                self.assertEqual(row.mae_reduction_percent, 100)
            else:
                self.assertTrue(pd.isna(row.mae))
                self.assertTrue(pd.isna(row.negative_predictions))
        private = table[(table.model == 'candidate') & (table['group'] == 'Private room')].iloc[0]
        self.assertAlmostEqual(private.baseline_mae, (999 - part[part.room_type == 'Private room'].advertised_price).mean())
        predictions['candidate'] = predictions['candidate'].iloc[::-1]
        reordered = group_comparisons(part, predictions, split='validation')
        pd.testing.assert_frame_equal(table, reordered)

    def test_baseline_winner(self):
        data = fixture()
        data['advertised_price'] = 100.
        result = compare_models(data)
        self.assertEqual(result['selection']['selected_model'], 'median')
        self.assertEqual(len(result['metrics'].query("split == 'test'")), 1)
        self.assertEqual(result['predictions'].query("split == 'test'").model.unique().tolist(), ['median'])


if __name__ == '__main__':
    unittest.main()
