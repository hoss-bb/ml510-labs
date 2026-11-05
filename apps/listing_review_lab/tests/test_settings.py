"""Configuration precedence in fresh processes, without provider calls."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import importlib


class SettingsTests(unittest.TestCase):
    def check_precedence(self, app_values, process_values, expected):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            application = root / 'application'
            application.mkdir()
            (root / '.env').write_text('AI_PROVIDER=gemini\nAI_MODEL=root-model\nGOOGLE_API_KEY=root-test-value\n', encoding='utf-8')
            (application / '.env').write_text(app_values, encoding='utf-8')
            environment = {key: value for key, value in os.environ.items()
                           if key not in {'AI_PROVIDER', 'AI_MODEL', 'GOOGLE_API_KEY'}}
            environment.update(process_values)
            code = '''
import importlib, os, sys
from pathlib import Path
runtime = importlib.import_module('lab_helpers.s10_shared.runtime.settings')
chain = importlib.import_module('lab_helpers.s10_shared.reviews.configuration')
runtime.ROOT = chain.ROOT = Path(sys.argv[1])
runtime.APP = runtime.ROOT / 'application'
result = chain.generation_settings()
assert result['model'] == sys.argv[2]
assert os.environ['GOOGLE_API_KEY'] == sys.argv[3]
'''
            result = subprocess.run([sys.executable, '-c', code, directory, *expected],
                                    env=environment, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_application_precedence(self):
        self.check_precedence('AI_MODEL=app-model\nGOOGLE_API_KEY=app-test-value\n', {},
                              ('app-model', 'app-test-value'))

    def test_process_precedence(self):
        self.check_precedence('AI_MODEL=app-model\nGOOGLE_API_KEY=app-test-value\n',
                              {'AI_MODEL': 'process-model', 'GOOGLE_API_KEY': 'process-test-value'},
                              ('process-model', 'process-test-value'))

    def test_root_fallback(self):
        self.check_precedence('', {}, ('root-model', 'root-test-value'))

    def test_root_local_defaults_and_overrides(self):
        runtime = importlib.import_module('lab_helpers.s10_shared.runtime.settings')
        simulator = importlib.import_module('apps.listings_simulator.simulator.helpers.settings')
        with patch.dict(os.environ, {}, clear=True), patch.object(runtime, 'load_environment'):
            config = runtime.settings()
            self.assertEqual(config.data, runtime.ROOT / 'local/listing_review')
            self.assertEqual(config.database, config.data / 'state/app.sqlite')
            self.assertEqual(simulator.settings().data, runtime.ROOT / 'local/listings_simulator')
            with patch.dict(os.environ, {'S10_DATA': '/custom/data', 'S10_MODEL': '/custom/model',
                                       'S10_DATABASE': '/custom/db', 'LISTINGS_SIMULATOR_DATA': '/custom/simulator'}):
                config = runtime.settings()
                self.assertEqual(config.data, Path('/custom/data'))
                self.assertEqual(config.model, Path('/custom/model'))
                self.assertEqual(config.database, Path('/custom/db'))
                self.assertEqual(simulator.settings().data, Path('/custom/simulator'))

    def test_prompt_default_follows_data_and_explicit_override(self):
        from lab_helpers.s10_shared.reviews.prompts import save_prompts, load_prompts
        from apps.listing_review.jobs.prompt_eng.helpers.templates import assemble
        runtime = importlib.import_module('lab_helpers.s10_shared.runtime.settings')
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            starting, candidate = assemble('starting'), assemble('candidate')
            save_prompts(data / 'prompts', starting)
            save_prompts(data / 'override', candidate)
            with patch.dict(os.environ, {'S10_DATA': str(data)}, clear=True), patch.object(runtime, 'load_environment'):
                self.assertEqual(load_prompts(), starting)
                with patch.dict(os.environ, {'S10_PROMPTS': str(data / 'override')}):
                    self.assertEqual(load_prompts(), candidate)
