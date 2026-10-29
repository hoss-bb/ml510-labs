"""Small checks for binary model adapters."""
import importlib.util
import tempfile
import json
import pickle
import unittest
from unittest.mock import patch
from pathlib import Path
from dataclasses import replace
import numpy as np
import pandas as pd
from ml_lib.features.schema import FeatureSchema
from ml_lib.features.preprocessing import PreprocessingConfig
from ml_lib.models.classification import ClassifierSpec, build_classifier, load_classifier
from ml_lib.models.neural_classification import MLPConfig, TabularTransformerConfig, build_neural_classifier


class ClassifierTests(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame({"value": [-3., -2., -1., 1., 2., 3.], "group": ["a", "a", "b", "b", "c", "c"]})
        self.y = pd.Series([0, 0, 0, 1, 1, 1])
        self.schema = FeatureSchema(("value",), ("group",))
        self.prep = PreprocessingConfig(True, "impute")

    def test_curated_controls(self):
        for family, parameters in [("logistic", {}), ("tree", {"max_depth": 2}), ("prior", {"random_state": 3})]:
            with self.subTest(family=family), self.assertRaises(ValueError):
                ClassifierSpec(family, parameters)

    def test_parameter_snapshot(self):
        p = {"C": 1.0}
        model = build_classifier(ClassifierSpec("logistic", p), self.schema, preprocessing=self.prep)
        p["C"] = 50
        self.assertEqual(model.describe()["spec"]["parameters"]["C"], 1.0)

    def test_adapters(self):
        specs = [ClassifierSpec("prior"), ClassifierSpec("logistic", {"C": 1.}),
                 ClassifierSpec("tree", {"max_depth": 2, "min_samples_leaf": 1}),
                 ClassifierSpec("forest", {"n_estimators": 2, "max_depth": 2, "min_samples_leaf": 1, "max_features": "sqrt"}),
                 ClassifierSpec("boosting", {"max_iter": 2, "learning_rate": .1, "max_leaf_nodes": 3, "min_samples_leaf": 1, "class_weight": None})]
        for spec in specs:
            with self.subTest(family=spec.family), tempfile.TemporaryDirectory() as tmp:
                model = build_classifier(spec, self.schema, preprocessing=self.prep).fit(self.X, self.y)
                p = model.predict_proba(self.X)
                self.assertEqual(p.shape, (6, 2))
                np.testing.assert_allclose(p.sum(axis=1), 1)
                model.save(tmp)
                np.testing.assert_allclose(load_classifier(tmp, trusted=True).predict_proba(self.X), p)
                with self.assertRaises(ValueError):
                    load_classifier(tmp)

    def test_training_only_preparation(self):
        model = build_classifier(ClassifierSpec("logistic", {"C": 1}), self.schema, preprocessing=self.prep).fit(self.X, self.y)
        scaler = model.pipeline.named_steps["prepare"].named_transformers_["numeric"].named_steps["scale"]
        expected = scaler.mean_.copy()
        _ = model.predict_proba(pd.DataFrame({"value": [1000.], "group": ["new"]}))
        np.testing.assert_array_equal(scaler.mean_, expected)
        self.assertAlmostEqual(float(expected[0]), 0)

    def test_boosting_weights(self):
        controls = dict(max_iter=2, learning_rate=.1, max_leaf_nodes=3, min_samples_leaf=1)
        with self.assertRaisesRegex(ValueError, "class_weight"):
            ClassifierSpec("boosting", controls)
        with self.assertRaisesRegex(ValueError, "class_weight"):
            ClassifierSpec("boosting", {**controls, "class_weight": {1: 10}})
        spec = ClassifierSpec("boosting", {**controls, "class_weight": "balanced"})
        model = build_classifier(spec, self.schema, preprocessing=self.prep).fit(self.X, self.y)
        self.assertEqual(model.pipeline.named_steps["model"].class_weight, "balanced")
        self.assertEqual(model.describe()["policies"]["class_weight"], "balanced")
        with tempfile.TemporaryDirectory() as tmp:
            model.save(tmp)
            restored = load_classifier(tmp, trusted=True)
            self.assertEqual(restored.describe(), model.describe())
            np.testing.assert_allclose(restored.predict_proba(self.X), model.predict_proba(self.X))

    def test_missing_inputs(self):
        model = build_classifier(ClassifierSpec("logistic", {"C": 1}), self.schema,
                                 preprocessing=PreprocessingConfig(True, "error"))
        X = self.X.copy()
        X.loc[0, "value"] = np.nan
        with self.assertRaises(ValueError):
            model.fit(X, self.y)

    def test_invalid_labels(self):
        model = build_classifier(ClassifierSpec("prior"), self.schema, preprocessing=self.prep)
        for labels in [[0]*6, [0, 0, 0, 1, 1, 2], [0, 0, 0, 1, 1, np.nan]]:
            with self.assertRaises(ValueError):
                model.fit(self.X, labels)

    def test_empty_training_column(self):
        model = build_classifier(ClassifierSpec("prior"), self.schema, preprocessing=self.prep)
        X = self.X.copy()
        X["value"] = np.nan
        with self.assertRaisesRegex(ValueError, "no observed values"):
            model.fit(X, self.y)

    def test_neural_controls(self):
        base = MLPConfig((4,), .1, .001, 4, 1, 1)
        for update in [{"hidden_units": (2.5,)}, {"epochs": True}, {"learning_rate": np.nan}, {"dropout": 1.0}]:
            with self.assertRaises(ValueError):
                replace(base, **update)
        with self.assertRaises(ValueError):
            TabularTransformerConfig(5, 2, 1, 8, .1, .001, 4, 1, 1)

    @unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow is not installed")
    def test_neural_roundtrip(self):
        configs = [MLPConfig((4,), .1, .001, 4, 1, 1),
                   TabularTransformerConfig(4, 2, 1, 8, .1, .001, 4, 1, 1)]
        X_val = self.X.copy().set_axis(range(10, 16))
        X_val["value"] += 100.0
        y_val = self.y.copy().set_axis(range(10, 16))
        for config in configs:
            with self.subTest(architecture=config.architecture), tempfile.TemporaryDirectory() as tmp:
                model = build_neural_classifier(config, self.schema, preprocessing=self.prep)
                training = Path(tmp) / "training"
                model.fit(self.X, self.y, validation=(X_val, y_val), training_directory=training)
                status = json.loads((training / "training.json").read_text())
                self.assertTrue(status["completed"])
                self.assertEqual(status["epochs_finished"], 1)
                self.assertEqual(len(pd.read_csv(training / "history.csv")), 1)
                self.assertTrue((training / "configuration.json").is_file())
                self.assertFalse((training / "experiment.json").exists())
                with (training / "preparation.pkl").open("rb") as stream:
                    prepared = pickle.load(stream)
                np.testing.assert_allclose(prepared.named_transformers_["numeric"].named_steps["scale"].mean_, [0.])
                scaler = model.prepare.named_transformers_["numeric"].named_steps["scale"]
                np.testing.assert_allclose(scaler.mean_, [self.X.value.mean()])
                probe = X_val.copy()
                probe.loc[10, "group"] = "unseen"
                p = model.predict_proba(probe)
                self.assertTrue(np.isfinite(p).all())
                model.save(tmp)
                restored = load_classifier(tmp, trusted=True)
                np.testing.assert_allclose(restored.predict_proba(probe), p, atol=1e-6)
                import tensorflow as tf
                best = tf.keras.models.load_model(training / "best.keras", safe_mode=True)
                np.testing.assert_allclose(best(model._inputs(probe), training=False).numpy()[:, 0], p[:, 1], atol=1e-6)
                with self.assertRaises(ValueError):
                    restored.predict_proba(probe.drop(columns="value"))
                with self.assertRaises(ValueError):
                    model.fit(self.X, self.y, validation=(self.X, self.y))

    @unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow is not installed")
    def test_interrupted_training(self):
        from ml_lib.models.neural_classification import _tensorflow
        config = MLPConfig((4,), .1, .001, 4, 2, 1)
        tf = _tensorflow(config)
        original_fit = tf.keras.Model.fit
        class Interrupt(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                raise KeyboardInterrupt("simulated interruption")
        def interrupted_fit(instance, *args, **kwargs):
            kwargs["callbacks"].append(Interrupt())
            return original_fit(instance, *args, **kwargs)
        X_val = self.X.copy().set_axis(range(10, 16))
        y_val = self.y.copy().set_axis(range(10, 16))
        with tempfile.TemporaryDirectory() as tmp:
            model = build_neural_classifier(config, self.schema, preprocessing=self.prep)
            model.history = {"loss": [999.]}
            with patch.object(tf.keras.Model, "fit", new=interrupted_fit), self.assertRaises(KeyboardInterrupt):
                model.fit(self.X, self.y, validation=(X_val, y_val), training_directory=tmp)
            self.assertEqual(model.history, {})
            status = json.loads((Path(tmp) / "training.json").read_text())
            self.assertFalse(status["completed"])
            self.assertEqual(status["state"], "interrupted")
            self.assertEqual(status["epochs_finished"], 1)
            self.assertTrue((Path(tmp) / "best.keras").is_file())
            self.assertEqual(len(pd.read_csv(Path(tmp) / "history.csv")), 1)
