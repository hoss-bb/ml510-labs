"""Fit on training, select on validation, then assess the frozen choice on test."""
from pathlib import Path
import json
from time import perf_counter
import pandas as pd
from ml_lib.data.loading import sha256_file
from ml_lib.features.schema import FeatureSchema
from ml_lib.features.preprocessing import PreprocessingConfig
from ml_lib.models.classical import build_regressor
from ml_lib.models.interfaces import ModelSpec
from ml_lib.evaluation.metrics import regression_metrics
from ml_lib.evaluation.selection import select_model
from ml_lib.experiment.artifacts import RunArtifacts, write_json
from lab_helpers.s10_shared.prediction.schemas import Features
from lab_helpers.s10_shared.runtime.settings import settings, ROOT
from lab_helpers.s10_shared.evaluation import check_partitions, group_comparisons, group_coverage


def compare_models(data, ridge_alpha=10.0):
    # 1. Check the evaluation boundary before fitting anything.
    parts = check_partitions(data)
    schema = FeatureSchema(numeric=('accommodates', 'bedrooms', 'bathrooms', 'beds'),
                           categorical=('property_type', 'room_type', 'neighbourhood'))
    specs = {'median': ModelSpec('median'), 'ridge': ModelSpec('ridge', {'alpha': ridge_alpha}),
             'boosting': ModelSpec('boosting', {'learning_rate': .05, 'max_iter': 200,
                                              'max_leaf_nodes': 31, 'min_samples_leaf': 20})}
    models, scores, prediction_rows = {}, [], []
    validation_predictions = {}
    # 2. Fit each complete pipeline on training; compare on the same validation rows.
    for name, spec in specs.items():
        model = build_regressor(spec, schema, preprocessing=PreprocessingConfig(True, 'impute'), seed=42)
        started = perf_counter()
        model.fit(parts['train'][schema.columns], parts['train'].advertised_price)
        elapsed = perf_counter() - started
        models[name] = model
        for split in ['train', 'validation']:
            part = parts[split]
            predicted = model.predict(part[schema.columns])
            scores.append({'model': name, 'split': split, 'fit_seconds': elapsed,
                           **regression_metrics(part.advertised_price, predicted)})
            if split == 'validation':
                validation_predictions[name] = pd.Series(predicted, index=pd.Index(part.listing_id))
                prediction_rows.append(pd.DataFrame({'listing_id': part.listing_id, 'model': name,
                                       'split': split, 'actual': part.advertised_price, 'prediction': predicted}))
    validation_table = pd.DataFrame(scores).query("split == 'validation'")
    validation_groups = group_comparisons(parts['validation'], validation_predictions, split='validation')

    # 3. Freeze selection using overall validation MAE, not a favourable group or test score.
    selection = select_model(validation_table.set_index('model'), 'mae')
    winner = selection['selected_model']

    # 4. Assess the frozen winner and the already-fitted baseline. Never refit here.
    test = parts['test']
    test_predictions = {}
    for name in dict.fromkeys(['median', winner]):
        predicted = models[name].predict(test[schema.columns])
        test_predictions[name] = pd.Series(predicted, index=pd.Index(test.listing_id))
        scores.append({'model': name, 'split': 'test', **regression_metrics(test.advertised_price, predicted)})
        prediction_rows.append(pd.DataFrame({'listing_id': test.listing_id, 'model': name, 'split': 'test',
                                            'actual': test.advertised_price, 'prediction': predicted}))
    test_groups = group_comparisons(test, test_predictions, split='test')
    if parts['validation'].query("room_type == 'Hotel room'").empty:
        test_groups.loc[test_groups['group'] == 'Hotel room', 'evidence_note'] = 'No validation coverage in this snapshot; test evidence is limited.'
    metrics = pd.DataFrame(scores)
    metrics['baseline_mae'] = metrics['split'].map(metrics[metrics.model == 'median'].set_index('split').mae)
    metrics['mae_improvement'] = metrics.baseline_mae - metrics.mae
    metrics['mae_reduction_percent'] = 100 * metrics.mae_improvement / metrics.baseline_mae.where(metrics.baseline_mae > 0)
    return {'models': models, 'selection': selection, 'metrics': metrics,
            'predictions': pd.concat(prediction_rows, ignore_index=True),
            'validation_groups': validation_groups, 'test_groups': test_groups,
            'group_coverage': group_coverage(parts)}


def train(ridge_alpha=10.0, *, data_directory=None, output_directory=None):
    config = settings() if data_directory is None or output_directory is None else None
    model_directory = Path(output_directory) if output_directory is not None else config.model
    if model_directory.exists():
        raise FileExistsError('Model destination exists. Set S10_MODEL to a new folder; compare before replacing it.')
    data_directory = Path(data_directory) if data_directory is not None else config.data
    source = data_directory / 'prepared' / 'listings.csv'
    manifest = json.loads((source.parent / 'manifest.json').read_text(encoding='utf-8'))
    if sha256_file(source) != manifest['prepared_sha256']:
        raise ValueError('Prepared data changed; prepare a new recorded dataset.')
    data = pd.read_csv(source, dtype={'listing_id': str, 'host_id': str})
    result = compare_models(data, ridge_alpha)
    selection, models = result['selection'], result['models']
    winner = selection['selected_model']

    # 5. Save evidence and the same fitted pipeline used for the test assessment.
    run = RunArtifacts(ROOT, 's10_training', config={'seed': 42, 'selection': 'mae'}, dataset_path=source)
    source_folders = [ROOT / 'apps/listing_review', ROOT / 'src/lab_helpers/s10_shared']
    run.record('application_sources', {str(path.relative_to(ROOT)): sha256_file(path)
               for folder in source_folders for path in folder.rglob('*.py')
               if 'local' not in path.parts and 'instructor' not in path.parts})
    for name in ['metrics', 'predictions', 'validation_groups', 'test_groups', 'group_coverage']:
        run.save_table(name.replace('_', '-'), result[name])
    run.record('experiment', {'selection': selection, 'models': {n: m.describe() for n, m in models.items()},
                              'dataset': manifest, 'group_views': ['room_type', 'capacity'],
                              'baseline': 'Global training median; unchanged across groups and splits.'})
    model_directory.mkdir(parents=True)
    models[winner].save(model_directory / 'model.pkl')
    # JSON uses null rather than NaN for values unavailable for a split.
    scores = json.loads(result['metrics'].to_json(orient='records', double_precision=15))
    metadata = {'version': run.directory.name, 'selected_model': winner, 'currency': 'CAD',
                'model_sha256': sha256_file(model_directory / 'model.pkl'), 'dataset': manifest,
                'description': models[winner].describe(), 'input_contract': Features.model_json_schema(),
                'metrics': scores, 'selection': selection}
    write_json(model_directory / 'metadata.json', metadata)
    run.record('bundle', metadata)
    run.write_overview('S10 model comparison', f'Validation MAE selected {winner}. Group results describe limitations, not separate winners.',
                       inspect_first=['tables/metrics.csv', 'tables/validation-groups.csv',
                                      'tables/test-groups.csv', 'tables/group-coverage.csv'])
    for split, title in [('validation', 'Validation comparison'), ('test', 'Frozen test assessment')]:
        print('\n' + title)
        columns = ['model', 'n', 'mae', 'rmse', 'mae_reduction_percent']
        print(result['metrics'].query('split == @split')[columns].to_string(index=False, float_format=lambda value: f'{value:.2f}'))
    print(f'Saved {winner} to {model_directory}; evidence: {run.directory}')
    return run.directory
