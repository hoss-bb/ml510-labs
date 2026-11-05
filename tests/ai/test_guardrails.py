"""Search boundaries before and after a tool call."""
import unittest
from unittest.mock import Mock
from ai_lib.models.errors import ExternalCallError
from ai_lib.guardrails.search import validate_search, search_documents, restricted_search
from ai_lib.guardrails.answers import redact_email


class GuardrailTests(unittest.TestCase):
    def test_arguments(self):
        self.assertEqual(validate_search({"query": " IBM CLI "}), "IBM CLI")
        for arguments in ({"query":""}, {"query":"x", "include_domains": ["bad.org"]}, {"query": 12}):
            with self.assertRaises(ValueError): validate_search(arguments)

    def test_sources(self):
        payload = {"results": [{"url": "https://ibm.com.attacker.org", "content":"bad"},
            {"url":"https://cloud.ibm.com/docs", "content":"evidence"}]}
        self.assertEqual(len(search_documents(payload)), 1)
        self.assertEqual(search_documents({"results":[]}), [])
        with self.assertRaisesRegex(ExternalCallError,"malformed"): search_documents({})

    def test_execution(self):
        tool = Mock(max_results=3)
        tool.invoke.return_value = {"results": []}
        restricted_search({"query":"IBM CLI"},tool)
        self.assertEqual(tool.invoke.call_args.args[0]["include_domains"],["ibm.com"])
        self.assertNotIn("max_results",tool.invoke.call_args.args[0])
        tool.reset_mock()
        with self.assertRaises(ValueError): restricted_search({"query":"x", "max_results":99},tool)
        tool.invoke.assert_not_called()

    def test_request_redaction(self):
        self.assertEqual(redact_email("Help a@example.org"), "Help [REDACTED_EMAIL]")
        self.assertEqual(redact_email("Reset an API key"), "Reset an API key")


class AdditionalBoundaryTests(unittest.TestCase):
    def test_empty_search(self):
        from langchain_core.tools import ToolException
        tool = Mock(max_results=3)
        tool.invoke.side_effect = ToolException(
            "No search results found for 'IBM CLI'. Suggestions: Try modifying your search parameters.")
        self.assertEqual(restricted_search({"query": "IBM CLI"}, tool), [])

    def test_tool_failure(self):
        from langchain_core.tools import ToolException
        tool = Mock(max_results=3)
        tool.invoke.side_effect = ToolException("Provider unavailable")
        with self.assertRaises(ToolException):
            restricted_search({"query": "IBM CLI"}, tool)

    def test_native_empty_search(self):
        from ai_lib.tools.search import build_web_search
        from unittest.mock import patch
        with patch.dict('os.environ', {'TAVILY_API_KEY': 'synthetic-key'}):
            tool = build_web_search()
            with patch.object(type(tool.api_wrapper), 'raw_results', return_value={'results': []}):
                self.assertEqual(restricted_search({'query': 'IBM CLI'}, tool), [])

    def test_malformed_url(self):
        from ai_lib.guardrails.search import search_documents
        docs = search_documents({"results": [
            {"url": "https://[bad", "content": "broken"},
            {"url": "https://www.ibm.com/docs", "content": "valid"}]})
        self.assertEqual(len(docs), 1)

    def test_complete_proposal(self):
        from ai_lib.guardrails.search import validate_tool_calls
        call = {"id": "one", "name": "search", "args": {"query": "IBM Cloud"}}
        validate_tool_calls([call], "search")
        for calls in [[call, call], [call, {**call, "id": "two", "name": "unknown"}], [call]*3]:
            with self.assertRaises(ValueError):
                validate_tool_calls(calls, "search")
