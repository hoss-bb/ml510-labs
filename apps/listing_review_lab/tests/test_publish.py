"""Publish arrivals without touching the real incoming folder."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import os
import subprocess
import sys
from fastapi.testclient import TestClient
from apps.listings_simulator.simulator.helpers.publication import publish_day, publish_periodically
from apps.listings_simulator.simulator.api import create_app


class PublishTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.examples = self.root / 'examples'
        self.incoming = self.root / 'incoming'
        self.examples.mkdir()
        self.content = b'listing_id,description\n101,"First line\nsecond line"\n102,Example\n'
        (self.examples / 'day_003.csv').write_bytes(self.content)

    def test_standalone_import_and_data_override(self):
        code = """
import sys
from pathlib import Path
from apps.listings_simulator.simulator.helpers.settings import settings
from apps.listings_simulator.simulator.api import app
assert settings().incoming == Path(sys.argv[1]) / 'incoming'
assert not any(name.startswith(('apps.listing_review', 'ml_lib', 'ai_lib')) for name in sys.modules)
"""
        result = subprocess.run([sys.executable, '-c', code, str(self.root)],
                                env={**os.environ, 'LISTINGS_SIMULATOR_DATA': str(self.root)},
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_publication(self):
        with TestClient(create_app(self.incoming)) as client:
            self.assertEqual(client.get('/batches').json(), [])
            self.assertEqual(publish_day(3, self.examples, self.incoming), 'Day 3 published: 2 listings.')
            batches = client.get('/batches').json()
            self.assertEqual([batch['id'] for batch in batches], ['day_003'])
            self.assertEqual(client.get('/batches/day_003').content, self.content)
        self.assertEqual([path.name for path in self.incoming.iterdir()], ['day_003.csv'])

    def test_repeated_day(self):
        publish_day(3, self.examples, self.incoming)
        target = self.incoming / 'day_003.csv'
        before = target.stat().st_mtime_ns
        self.assertEqual(publish_day(3, self.examples, self.incoming), 'Day 3 is already available.')
        self.assertEqual(target.stat().st_mtime_ns, before)

    def test_changed_day(self):
        publish_day(3, self.examples, self.incoming)
        (self.examples / 'day_003.csv').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'different contents'):
            publish_day(3, self.examples, self.incoming)
        self.assertEqual((self.incoming / 'day_003.csv').read_bytes(), self.content)

    def test_missing_day(self):
        with self.assertRaisesRegex(ValueError, 'Available days: 3'):
            publish_day(8, self.examples, self.incoming)
        self.assertFalse(self.incoming.exists())

    def test_invalid_day(self):
        for day in [0, -1, 1.5, True]:
            with self.subTest(day=day), self.assertRaises(ValueError):
                publish_day(day, self.examples, self.incoming)

    def test_timed_order_and_exhaustion(self):
        for day in [1, 2]:
            (self.examples / f'day_{day:03d}.csv').write_bytes(self.content)
        wait, emit = Mock(), Mock()
        self.assertEqual(publish_periodically(10, self.examples, self.incoming, wait=wait, emit=emit), 3)
        self.assertEqual(wait.call_args_list, [((10,),), ((10,),)])
        messages = [call.args[0] for call in emit.call_args_list]
        self.assertEqual([m.split()[1] for m in messages[:3]], ['1', '2', '3'])
        wait.reset_mock()
        self.assertEqual(publish_periodically(10, self.examples, self.incoming, wait=wait, emit=Mock()), 0)
        wait.assert_not_called()

    def test_restart_after_interruption(self):
        (self.examples / 'day_001.csv').write_bytes(self.content)
        with self.assertRaises(KeyboardInterrupt):
            publish_periodically(10, self.examples, self.incoming,
                                 wait=Mock(side_effect=KeyboardInterrupt), emit=Mock())
        original = (self.incoming / 'day_001.csv').stat().st_mtime_ns
        wait = Mock()
        self.assertEqual(publish_periodically(10, self.examples, self.incoming, wait=wait, emit=Mock()), 1)
        wait.assert_not_called()
        self.assertEqual((self.incoming / 'day_001.csv').stat().st_mtime_ns, original)

    def test_timed_conflict_and_invalid_interval(self):
        publish_day(3, self.examples, self.incoming)
        (self.examples / 'day_003.csv').write_text('changed')
        wait = Mock()
        with self.assertRaisesRegex(ValueError, 'different contents'):
            publish_periodically(10, self.examples, self.incoming, wait=wait, emit=Mock())
        wait.assert_not_called()
        for value in [0, -1, float('nan'), float('inf'), True]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                publish_periodically(value, self.examples, self.incoming, wait=wait)
        with self.assertRaisesRegex(ValueError, 'No prepared days'):
            publish_periodically(1, self.root / 'missing', self.incoming, wait=wait)

    def test_cli_modes_are_exclusive(self):
        result = subprocess.run([sys.executable, '-m', 'apps.listings_simulator.simulator', '--day', '3', '--interval', '10'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)

    def test_environment_modes_and_overrides(self):
        from apps.listings_simulator.simulator.helpers.settings import publication_settings
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(publication_settings(), (1, None))
            with patch.dict(os.environ, {'SIMULATOR_MODE': 'periodic', 'SIMULATOR_INTERVAL_SECONDS': '5', 'SIMULATOR_DAY': 'invalid'}):
                self.assertEqual(publication_settings(), (None, 5))
                self.assertEqual(publication_settings(day=3), (3, None))
            with patch.dict(os.environ, {'SIMULATOR_MODE': 'invalid'}):
                self.assertEqual(publication_settings(interval=2), (None, 2))
                with self.assertRaises(ValueError): publication_settings()
            for values in [{'SIMULATOR_DAY': '0'}, {'SIMULATOR_DAY': 'x'},
                           {'SIMULATOR_MODE': 'periodic', 'SIMULATOR_INTERVAL_SECONDS': 'nan'},
                           {'SIMULATOR_MODE': 'periodic', 'SIMULATOR_INTERVAL_SECONDS': '-1'},
                           {'SIMULATOR_MODE': 'periodic', 'SIMULATOR_INTERVAL_SECONDS': 'x'}]:
                with patch.dict(os.environ, values, clear=True), self.assertRaises(ValueError):
                    publication_settings()

    def test_native_module_from_application_directory(self):
        application = Path(__file__).resolve().parents[2] / 'listings_simulator'
        (self.root / 'arrival_examples').mkdir()
        (self.root / 'arrival_examples/day_003.csv').write_bytes(self.content)
        environment = {**os.environ, 'LISTINGS_SIMULATOR_DATA': str(self.root),
                       'SIMULATOR_MODE': 'once', 'SIMULATOR_DAY': '3'}
        for attempt in range(2):
            result = subprocess.run([sys.executable, '-m', 'simulator'], cwd=application,
                                    env=environment, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('already available' if attempt else 'published', result.stdout)
        self.assertEqual((self.incoming / 'day_003.csv').read_bytes(), self.content)

    def test_environment_periodic_command_uses_mocked_wait(self):
        import importlib
        command = importlib.import_module('apps.listings_simulator.simulator.__main__')
        from apps.listings_simulator.simulator.helpers.publication import publish_periodically as actual
        (self.root / 'arrival_examples').mkdir()
        for day in [1, 2]:
            (self.root / f'arrival_examples/day_{day:03d}.csv').write_bytes(self.content)
        wait = Mock()
        def publish(interval, examples, incoming):
            return actual(interval, examples, incoming, wait=wait, emit=Mock())
        with patch.dict(os.environ, {'LISTINGS_SIMULATOR_DATA': str(self.root), 'SIMULATOR_MODE': 'periodic',
                                     'SIMULATOR_INTERVAL_SECONDS': '7'}, clear=True), \
             patch.object(sys, 'argv', ['simulator']), patch.object(command, 'publish_periodically', side_effect=publish), \
             patch('builtins.print'):
            command.main()
        wait.assert_called_once_with(7)


if __name__ == '__main__':
    unittest.main()
