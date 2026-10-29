import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
from ml_lib.models.classical import build_regressor, load_regressor
from ml_lib.models.interfaces import ModelSpec
from ml_lib.features.preprocessing import PreprocessingConfig
from ml_lib.problems.regression import FeatureSchema


class ClassicalTests(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame({"x": np.arange(20, dtype=float), "kind": ["a", "b"]*10})
        self.y = pd.Series(2*self.X.x+3)
        self.preprocessing = PreprocessingConfig(scale_numeric=True, missing_values="impute")
        self.schema = FeatureSchema(numeric=("x",), categorical=("kind",))

    def test_all_implemented_families_fit_predict(self):
        for family in ["mean", "median", "linear", "ridge", "tree", "forest", "boosting"]:
            with self.subTest(family=family):
                params = {
                    "ridge": {"alpha": 1.0},
                    "tree": {"max_depth": None, "min_samples_leaf": 1},
                    "forest": {"n_estimators": 3, "max_depth": None, "min_samples_leaf": 1, "max_features": 1.0},
                    "boosting": {"max_iter": 3, "learning_rate": 0.1, "max_leaf_nodes": 31, "min_samples_leaf": 20},
                }.get(family, {})
                model = build_regressor(ModelSpec(family, params), self.schema, preprocessing=self.preprocessing).fit(self.X, self.y)
                self.assertEqual(model.predict(self.X).shape, (20,))

    def test_unfitted_fails(self):
        with self.assertRaises(RuntimeError): build_regressor(ModelSpec("linear"), self.schema, preprocessing=self.preprocessing).predict(self.X)

    def test_unknown_family_fails(self):
        with self.assertRaises(ValueError): build_regressor(ModelSpec("magic"), self.schema, preprocessing=self.preprocessing)

    def test_train_only_preprocessing(self):
        model = build_regressor(ModelSpec("linear"), self.schema, preprocessing=self.preprocessing).fit(self.X.iloc[:10], self.y.iloc[:10])
        mean = model.pipeline.named_steps["prepare"].named_transformers_["numeric"].named_steps["scale"].mean_[0]
        self.assertEqual(mean, 4.5)
        X2 = pd.DataFrame({"x": [9999.], "kind": ["new"]})
        self.assertTrue(np.isfinite(model.predict(X2)).all())
        self.assertEqual(model.pipeline.named_steps["prepare"].named_transformers_["numeric"].named_steps["scale"].mean_[0], 4.5)

    def test_save_reload_roundtrip(self):
        model = build_regressor(ModelSpec("ridge", {"alpha": 1.0}), self.schema, preprocessing=self.preprocessing).fit(self.X, self.y)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"model.pkl"; model.save(p)
            with self.assertRaisesRegex(ValueError,"Pickle"): load_regressor(p)
            np.testing.assert_allclose(load_regressor(p,trusted=True).predict(self.X),model.predict(self.X))

    def test_misaligned_labels_rejected(self):
        with self.assertRaises(ValueError): build_regressor(ModelSpec("linear"),self.schema, preprocessing=self.preprocessing).fit(self.X,self.y.iloc[::-1])

    def test_coefficient_inspection(self):
        model=build_regressor(ModelSpec("linear"),self.schema, preprocessing=self.preprocessing).fit(self.X,self.y)
        self.assertIn("coefficient",model.coefficients())

    def test_explicit_controls_required(self):
        for family in ("ridge", "tree", "forest", "boosting"):
            with self.subTest(family=family), self.assertRaisesRegex(ValueError, "missing"):
                ModelSpec(family)

    def test_unsupported_controls_rejected(self):
        for key in ("random_state", "early_stopping", "n_jobs", "unknown"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "unsupported"):
                ModelSpec("ridge", {"alpha": 1.0, key: 1})

    def test_effective_policies_recorded(self):
        spec = ModelSpec("boosting", {"learning_rate": 0.1, "max_iter": 3,
                                     "max_leaf_nodes": 15, "min_samples_leaf": 20})
        model = build_regressor(spec, self.schema, preprocessing=self.preprocessing, seed=7)
        self.assertFalse(model.pipeline.named_steps["model"].early_stopping)
        self.assertEqual(model.pipeline.named_steps["model"].random_state, 7)
        description = model.describe()
        self.assertEqual(description["spec"]["parameters"], spec.parameters)
        self.assertEqual(description["preprocessing"], {"scale_numeric": True, "missing_values": "impute"})
        self.assertFalse(description["policies"]["early_stopping"])

    def test_parameter_snapshot(self):
        parameters = {"alpha": 10.0}
        spec = ModelSpec("ridge", parameters)
        model = build_regressor(spec, self.schema, preprocessing=self.preprocessing)
        parameters["alpha"] = 100.0
        self.assertEqual(model.describe()["spec"]["parameters"], {"alpha": 10.0})
        self.assertEqual(model.pipeline.named_steps["model"].alpha, 10.0)
