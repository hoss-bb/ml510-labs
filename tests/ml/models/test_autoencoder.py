import importlib.util
import json
import tempfile
import unittest
import numpy as np
from pathlib import Path
from ml_lib.models.autoencoder import Autoencoder, AutoencoderConfig


class AutoencoderTests(unittest.TestCase):
    def test_controls(self):
        with self.assertRaises(ValueError):
            AutoencoderConfig('gru',2.5,2,.001,8,1,0)
        with self.assertRaises(ValueError):
            AutoencoderConfig('dense',4,2,float('nan'),8,1,0)

    @unittest.skipUnless(importlib.util.find_spec('tensorflow'),'TensorFlow absent')
    def test_roundtrip(self):
        x=np.random.default_rng(42).normal(size=(16,3,2))
        for architecture in ['dense','gru']:
            with self.subTest(architecture=architecture), tempfile.TemporaryDirectory() as d:
                model=Autoencoder(AutoencoderConfig(architecture,4,2,.001,8,1,0))
                model.fit(x[:12],feature_names=['a','b'],validation=x[12:],training_directory=Path(d)/'training')
                pred=model.score_samples(x[12:],feature_names=['a','b'])
                self.assertEqual(model.reconstruct(x[12:],feature_names=['a','b']).shape,(4,3,2))
                np.testing.assert_allclose(model.contributions(x[12:],feature_names=['a','b']).mean(axis=1),pred,rtol=1e-5)
                self.assertTrue(json.loads((Path(d)/'training/training.json').read_text())['completed'])
                self.assertTrue((Path(d)/'training/best.keras').exists())
                model.save(Path(d)/'model')
                restored=Autoencoder.load(Path(d)/'model',trusted=True)
                np.testing.assert_allclose(pred,restored.score_samples(x[12:],feature_names=['a','b']),rtol=1e-5)
                with self.assertRaises(ValueError):
                    restored.score_samples(x[12:],feature_names=['b','a'])
