"""Optional runtime integration tests. A missing backend is explicitly reported."""
import importlib.util
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import numpy as np
from tests.ml.fixtures import numeric_hourly
from ml_lib.features.temporal import make_sequences
from ml_lib.models.deep_learning import SequenceConfig, TransformerConfig, SequenceForecaster
from ml_lib.models.ensemble import SequenceEnsemble


TRAINING = SequenceConfig(architecture="gru", hidden_units=3, dense_units=3,
                          epochs=1, batch_size=16, learning_rate=0.001, patience=4)
ATTENTION = TransformerConfig(model_dim=8, num_heads=2, num_blocks=1,
                             feedforward_dim=16, dropout=0.1, dense_units=3,
                             epochs=1, batch_size=16, learning_rate=0.001, patience=4)


class SequenceConfigurationTests(unittest.TestCase):
    def test_training_choices_required(self):
        with self.assertRaises(TypeError): SequenceConfig()

    def test_training_policies_recorded(self):
        policies = SequenceForecaster(TRAINING).describe()["policies"]
        self.assertEqual(policies["loss"], "mse")
        self.assertEqual(policies["monitor"], "val_loss")
        self.assertTrue(policies["restore_best_weights"])

    def test_invalid_architecture(self):
        with self.assertRaises(ValueError): replace(TRAINING, architecture="magic")

    def test_invalid_epoch_count(self):
        with self.assertRaises(ValueError): replace(TRAINING, epochs=0)

    def test_integer_training_sizes(self):
        for field in ("hidden_units", "dense_units", "epochs", "batch_size"):
            for value in (1.5, True, 0, -1):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    replace(TRAINING, **{field: value})

    def test_finite_learning_rate(self):
        for value in (np.nan, np.inf, -np.inf, 0, -0.001):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(TRAINING, learning_rate=value)

    def test_integer_patience(self):
        for value in (np.nan, np.inf, 1.5, True, -1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(TRAINING, patience=value)
        self.assertEqual(replace(TRAINING, patience=0).patience, 0)

    def test_transformer_controls(self):
        for changes in ({"model_dim": 7}, {"num_heads": 3}, {"num_blocks": 0},
                        {"dropout": 1.0}, {"dropout": np.nan}, {"learning_rate": np.nan},
                        {"patience": -1}, {"batch_size": 0}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(ATTENTION, **changes)
        with self.assertRaises(TypeError): TransformerConfig()

    def test_predict_before_fit(self):
        data=make_sequences(numeric_hourly(50),history_columns=["y"],target="y",lookback=4)
        with self.assertRaises(RuntimeError): SequenceForecaster(TRAINING).predict(data)
        with self.assertRaises(RuntimeError):
            SequenceForecaster(TRAINING).predict_inputs(
                data.history, data.future,
                history_names=data.history_names, future_names=data.future_names,
            )


@unittest.skipUnless(importlib.util.find_spec("tensorflow") is not None, "TensorFlow runtime not installed; integration not executed")
class TensorFlowIntegrationTests(unittest.TestCase):
    def run_case(self,architecture="gru",future_columns=("calendar",)):
        frame=numeric_hourly(160)
        data=make_sequences(frame,history_columns=["y","weather"],future_columns=future_columns,target="y",lookback=6)
        # Use disjoint target timestamps for training, validation and test.
        train=data.subset(np.arange(60))
        validation=data.subset(np.arange(70,90))
        test=data.subset(np.arange(100,110))
        config = ATTENTION if architecture == "transformer" else replace(TRAINING, architecture=architecture)
        model=SequenceForecaster(config)
        model.fit(train,validation)
        predictions=model.predict(test)
        self.assertEqual(predictions.shape,test.y.shape)
        self.assertEqual(predictions.ndim, 1)
        self.assertEqual(model.model.output_shape[-1], 1)
        self.assertEqual(model.describe()["config"]["architecture"], architecture)
        if architecture == "bilstm":
            self.assertEqual(model.describe()["hidden_units_scope"], "per direction")
            layer = model.model.get_layer("bidirectional_history")
            self.assertEqual(layer.forward_layer.units, TRAINING.hidden_units)
            self.assertEqual(layer.backward_layer.units, TRAINING.hidden_units)
        if architecture == "transformer":
            self.assertEqual(model.describe()["representation"]["positions"], "fixed sinusoidal")
            positions = model.model.get_layer("history_positions")
            encoded = positions(np.zeros((1, 6, 8), dtype=np.float32)).numpy()
            self.assertFalse(np.array_equal(encoded[:, 0], encoded[:, 1]))
        self.assertTrue(np.isfinite(predictions).all())
        names = {"history_names": test.history_names, "future_names": test.future_names}
        np.testing.assert_allclose(model.predict_inputs(test.history, test.future, **names), predictions)
        invalid = [
            (test.history[:, 0], test.future, names),
            (test.history[:, :-1], test.future, names),
            (test.history[:0], test.future[:0], names),
            (test.history, test.future[:-1], names),
            (test.history, test.future, {**names, "history_names": tuple(reversed(test.history_names))}),
            (np.full_like(test.history, np.nan), test.future, names),
        ]
        for history, future, feature_names in invalid:
            with self.assertRaises(ValueError):
                model.predict_inputs(history, future, **feature_names)
        self.assertAlmostEqual(float(model.scaling["y_mean"]),float(train.y.mean()),places=4)
        with tempfile.TemporaryDirectory() as tmp:
            model.save(tmp)
            loaded=SequenceForecaster.load(tmp)
            np.testing.assert_allclose(predictions,loaded.predict(test),rtol=1e-5,atol=1e-5)
            np.testing.assert_allclose(predictions, loaded.predict_inputs(test.history, test.future, **names), rtol=1e-5, atol=1e-5)

    def test_gru_next_hour_roundtrip(self): self.run_case()
    def test_bilstm_roundtrip(self): self.run_case("bilstm")
    def test_lstm_roundtrip(self): self.run_case("lstm",future_columns=())
    def test_transformer_roundtrip(self): self.run_case("transformer")
    def test_transformer_without_calendar(self): self.run_case("transformer", future_columns=())

    def test_ensemble_roundtrip(self):
        data = make_sequences(numeric_hourly(90), history_columns=["y", "weather"],
                              future_columns=["calendar"], target="y", lookback=6)
        train, validation, test = (data.subset(np.arange(a, b))
                                 for a, b in ((0, 35), (40, 50), (60, 65)))
        members = [SequenceForecaster(replace(ATTENTION, seed=seed)).fit(train, validation)
                   for seed in (43, 44)]
        ensemble = SequenceEnsemble(members)
        expected = np.mean([member.predict(test) for member in members], axis=0)
        np.testing.assert_allclose(ensemble.predict(test), expected)
        self.assertEqual(ensemble.describe()["parameters"], sum(m.model.count_params() for m in members))
        names = dict(history_names=test.history_names, future_names=test.future_names)
        with self.assertRaises(ValueError):
            ensemble.predict_inputs(test.history, test.future,
                                    **{**names, "history_names": tuple(reversed(test.history_names))})
        with tempfile.TemporaryDirectory() as tmp:
            ensemble.save(tmp)
            restored = SequenceEnsemble.load(tmp)
            np.testing.assert_allclose(restored.predict_inputs(test.history, test.future, **names),
                                       expected, rtol=1e-5, atol=1e-5)
