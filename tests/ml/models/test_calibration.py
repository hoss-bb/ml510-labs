import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from ml_lib.models.calibration import SigmoidCalibrator, CalibratedClassifier
from ml_lib.models.classification import build_classifier, ClassifierSpec, load_classifier
from ml_lib.features.schema import FeatureSchema
from ml_lib.features.preprocessing import PreprocessingConfig


class CalibrationTests(unittest.TestCase):
    def test_mapping_roundtrip(self):
        p = [0, .1, .3, .4, .6, .7, .9, 1]
        y = [0, 1, 0, 0, 1, 0, 1, 1]
        model = SigmoidCalibrator().fit(p, y)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibration.json"
            model.save(path)
            np.testing.assert_allclose(SigmoidCalibrator.load(path).predict(p), model.predict(p))
        self.assertTrue(np.isfinite(model.predict(p)).all())

    def test_invalid_calibration(self):
        with self.assertRaises(ValueError):
            SigmoidCalibrator().fit([.1, .2], [0, 0])
        with self.assertRaises(RuntimeError):
            SigmoidCalibrator().predict([.1])

    def test_bundle_roundtrip(self):
        X = pd.DataFrame({"x": [-2., -1., 1., 2.]})
        y = [0, 0, 1, 1]
        model = build_classifier(ClassifierSpec("logistic", {"C": 1.}), FeatureSchema(("x",)),
                                 preprocessing=PreprocessingConfig(True, "impute")).fit(X, y)
        original = model.predict_proba(X).copy()
        mapping = SigmoidCalibrator().fit(original[:, 1], [0, 1, 0, 1])
        calibrated = CalibratedClassifier(model, mapping)
        np.testing.assert_array_equal(model.predict_proba(X), original)
        with tempfile.TemporaryDirectory() as tmp:
            calibrated.save(tmp)
            np.testing.assert_allclose(load_classifier(tmp, trusted=True).predict_proba(X), calibrated.predict_proba(X))
