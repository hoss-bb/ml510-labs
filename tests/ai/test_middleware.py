"""Native middleware checks with deterministic model responses, no API calls."""
import unittest
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware, PIIMiddleware
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool


class ToolModel(FakeMessagesListChatModel):
    seen: list = []
    def bind_tools(self, tools, **kwargs):
        return self
    def _generate(self, messages, **kwargs):
        self.seen.append([m.content for m in messages])
        return super()._generate(messages, **kwargs)


class MiddlewareTests(unittest.TestCase):
    def test_call_budget(self):
        calls = []
        @tool
        def search(query: str) -> str:
            """Return teaching evidence."""
            calls.append(query)
            return "evidence"
        model = ToolModel(responses=[AIMessage(content="",tool_calls=[
            {"name":"search","args":{"query":"first"},"id":"a"},
            {"name":"search","args":{"query":"second"},"id":"b"}]), AIMessage(content="done")])
        agent = create_agent(model, tools=[search],middleware=[
            ModelCallLimitMiddleware(run_limit=1,exit_behavior="end"),
            ToolCallLimitMiddleware(tool_name="search",run_limit=1,exit_behavior="end")])
        agent.invoke({"messages":[{"role":"user","content":"search"}]})
        self.assertEqual(len(model.seen),1)
        self.assertLessEqual(len(calls),1)

    def test_redaction_before_model(self):
        model = ToolModel(responses=[AIMessage(content="Use official documentation.")])
        agent = create_agent(model,middleware=[PIIMiddleware("email",strategy="redact")])
        agent.invoke({"messages":[{"role":"user","content":"Help student@example.org"}]})
        self.assertNotIn("student@example.org",str(model.seen))
        self.assertIn("REDACTED",str(model.seen))

    def test_structured_completion(self):
        from ai_lib.evaluation.schemas import SupportAnswer
        @tool
        def search(query: str) -> str:
            """Return a source passage."""
            return "[p] Use the official installation documentation."
        model = ToolModel(responses=[
            AIMessage(content="",tool_calls=[{"name":"search","args":{"query":"CLI"},"id":"a"}]),
            AIMessage(content="",tool_calls=[{"name":"SupportAnswer","args":{
                "answer":"Use the official installation documentation.","source_ids":["p"],"disposition":"answer"},"id":"b"}])])
        agent = create_agent(model,tools=[search],response_format=SupportAnswer,middleware=[
            PIIMiddleware("email",strategy="redact",apply_to_input=True,apply_to_output=True),
            ModelCallLimitMiddleware(run_limit=3,exit_behavior="end"),
            ToolCallLimitMiddleware(tool_name="search",run_limit=1,exit_behavior="end")])
        result = agent.invoke({"messages":[{"role":"user","content":"Help student@example.org install the CLI"}]},
                              config={"recursion_limit":40})
        self.assertEqual(result["structured_response"].source_ids,["p"])
        self.assertEqual(len(model.seen),2)
        self.assertNotIn("student@example.org",str(model.seen))
