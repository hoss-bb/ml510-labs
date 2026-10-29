"""Provider setup checks require no network or real credentials."""
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch, MagicMock

from ai_lib.models import ModelSettings, load_settings, build_chat_model, api_error_guidance


class ModelSetupTests(unittest.TestCase):
    def test_missing_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "GOOGLE_API_KEY"):
                build_chat_model(ModelSettings("gemini", "test"))

    def test_invalid_provider(self):
        with self.assertRaisesRegex(ValueError, "gemini, groq, or cohere"):
            ModelSettings("unknown", "test")

    def test_environment_precedence(self):
        with TemporaryDirectory() as temp, patch.dict(os.environ, {"AI_PROVIDER": "groq", "AI_MODEL": "process-model"}, clear=True):
            Path(temp, ".env").write_text("AI_PROVIDER=gemini\nAI_MODEL=file-model\n", encoding="utf-8")
            settings = load_settings(Path(temp))
            self.assertEqual(settings.provider, "groq")
            self.assertEqual(settings.model, "process-model")
            self.assertEqual(os.environ["LANGSMITH_TRACING"], "false")

    def test_provider_dispatch(self):
        cases = [("gemini", "GOOGLE_API_KEY", "langchain_google_genai.ChatGoogleGenerativeAI"),
                 ("groq", "GROQ_API_KEY", "langchain_groq.ChatGroq"),
                 ("cohere", "COHERE_API_KEY", "langchain_cohere.ChatCohere")]
        for provider, key, target in cases:
            with self.subTest(provider=provider), patch.dict(os.environ, {key: "test-secret"}), patch(target) as constructor:
                native = MagicMock()
                constructor.return_value = native
                self.assertIs(build_chat_model(ModelSettings(provider, "test-model")), native)
                self.assertEqual(constructor.call_args.kwargs["model"], "test-model")

    def test_safe_error_guidance(self):
        result = api_error_guidance(RuntimeError("429 secret-payload"))
        self.assertIn("quota", result)
        self.assertNotIn("secret-payload", result)
