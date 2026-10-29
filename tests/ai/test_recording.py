import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from ai_lib.recording import LocalTrace


class RecordingTests(unittest.TestCase):
    def test_secret_exclusion(self):
        saved = {}
        run = SimpleNamespace(record=lambda name, value: saved.update({name: value}))
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-secret"}):
            trace = LocalTrace(run)
            trace.event("test", api_key="test-secret", text="contains test-secret", client=object())
        encoded = json.dumps(saved)
        self.assertNotIn("test-secret", encoded)
        self.assertNotIn("api_key", encoded)
        self.assertIn("[REDACTED]", encoded)

    def test_tool_association(self):
        saved = {}
        trace = LocalTrace(SimpleNamespace(record=lambda name, value: saved.update({name: value})))
        trace.on_chat_model_start({}, [[HumanMessage(content="search")]], run_id="model-1")
        message = AIMessage(content="", tool_calls=[{"name": "search", "args": {"query": "cloud"}, "id": "call-1", "type": "tool_call"}])
        trace.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]), run_id="model-1")
        trace.on_tool_start({"name": "search"}, '{"query":"cloud"}', run_id="tool-1")
        trace.on_tool_end(ToolMessage(content="evidence", tool_call_id="call-1"), run_id="tool-1")
        events = saved["trace"]["events"]
        self.assertEqual(events[1]["responses"][0]["tool_calls"][0]["id"], "call-1")
        self.assertEqual(events[-1]["result"]["tool_call_id"], "call-1")
        self.assertGreaterEqual(events[1]["seconds"], 0)

    def test_failed_call_cleanup(self):
        trace = LocalTrace(SimpleNamespace(record=lambda *args: None))
        trace.started["model"] = 0
        trace.started["tool"] = 0
        trace.on_llm_error(TimeoutError(),run_id="model")
        trace.on_tool_error(TimeoutError(),run_id="tool")
        self.assertEqual(trace.started,{})

    def test_recording_error_visible(self):
        def broken(*args): raise OSError("disk unavailable")
        trace = LocalTrace(SimpleNamespace(record=broken))
        self.assertTrue(trace.raise_error)
        with self.assertRaises(OSError): trace.event("test")
