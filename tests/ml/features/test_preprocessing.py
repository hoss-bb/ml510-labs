"""Only explicit preprocessing choices and training isolation."""
import unittest
import numpy as np
import pandas as pd
from ml_lib.features.preprocessing import PreprocessingConfig
from ml_lib.models.classical import build_regressor
from ml_lib.models.interfaces import ModelSpec
from ml_lib.problems.regression import FeatureSchema
from ml_lib.models.classification import ClassifierSpec, build_classifier


class PreprocessingTests(unittest.TestCase):
    def model(self, scale=True, missing="impute"):
        return build_regressor(ModelSpec("linear"), FeatureSchema(numeric=("x",)),
            preprocessing=PreprocessingConfig(scale_numeric=scale, missing_values=missing))

    def test_missing_inputs_rejected(self):
        model = self.model(missing="error")
        with self.assertRaisesRegex(ValueError, "Missing selected inputs"):
            model.fit(pd.DataFrame({"x": [1., np.nan, 3.]}), [1, 2, 3])
        model.fit(pd.DataFrame({"x": [1., 2., 3.]}), [1, 2, 3])
        with self.assertRaisesRegex(ValueError, "Missing selected inputs"):
            model.predict(pd.DataFrame({"x": [np.nan]}))

    def test_imputation_uses_training_median(self):
        model = self.model(scale=False).fit(pd.DataFrame({"x": [1., np.nan, 3.]}), [1, 2, 3])
        imputer = model.pipeline.named_steps["prepare"].named_transformers_["numeric"].named_steps["impute"]
        self.assertEqual(imputer.statistics_[0], 2.)
        np.testing.assert_allclose(model.predict(pd.DataFrame({"x": [np.nan, 100.]})), [2., 100.])
        self.assertEqual(imputer.statistics_[0], 2.)

    def test_scaling_can_be_disabled(self):
        model = self.model(scale=False, missing="error").fit(pd.DataFrame({"x": [1., 3., 5.]}), [2, 6, 10])
        transformed = model.pipeline.named_steps["prepare"].transform(pd.DataFrame({"x": [7.]}))
        np.testing.assert_array_equal(transformed, [[7.]])

    def test_invalid_policy_rejected(self):
        with self.assertRaises(ValueError): PreprocessingConfig(scale_numeric=True, missing_values="auto")

    def test_shared_input_checks(self):
        schema = FeatureSchema(numeric=("x",))
        for missing in ("impute", "error"):
            prep = PreprocessingConfig(True, missing)
            models = [build_regressor(ModelSpec("linear"), schema, preprocessing=prep),
                      build_classifier(ClassifierSpec("logistic", {"C": 1.}), schema, preprocessing=prep)]
            invalid = [pd.DataFrame({"x": []}), pd.DataFrame({"wrong": [1., 2.]}),
                       pd.DataFrame({"x": [np.inf, 2.]}), pd.DataFrame({"x": [np.nan, np.nan]})]
            if missing == "error":
                invalid.append(pd.DataFrame({"x": [np.nan, 2.]}))
            for frame in invalid:
                messages = []
                for model in models:
                    with self.assertRaises(ValueError) as error:
                        model.fit(frame, [0, 1])
                    messages.append(str(error.exception))
                self.assertEqual(messages[0], messages[1])
            for model in models:
                model.fit(pd.DataFrame({"x": [1., 2.]}), [0, 1])
                predict = getattr(model, "predict_proba", model.predict)
                with self.assertRaisesRegex(ValueError, "not infinite"):
                    predict(pd.DataFrame({"x": [np.inf]}))
