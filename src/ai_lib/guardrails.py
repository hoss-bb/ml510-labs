"""Enforce search boundaries before execution and inspect returned evidence."""
import re
from urllib.parse import urlsplit

from langchain_core.documents import Document


def validate_search(arguments):
    if not isinstance(arguments, dict) or set(arguments) != {"query"}:
        raise ValueError("Search accepts only a query; domains and limits belong to the application.")
    query = arguments["query"]
    if not isinstance(query, str) or not query.strip() or len(query) > 500:
        raise ValueError("Use a nonempty search query of at most 500 characters.")
    return query.strip()


def official_url(url):
    try:
        parsed = urlsplit(url)
        host = (parsed.hostname or "").lower()
    except (ValueError, TypeError):
        return False
    return parsed.scheme == "https" and (host == "ibm.com" or host.endswith(".ibm.com"))


def search_documents(payload):
    """Keep at most three valid official results; missing evidence stays empty."""
    from ai_lib.models import ExternalCallError
    if isinstance(payload, dict) and payload.get("error"):
        raise ExternalCallError("Search service failed; inspect access or quota before retrying.")
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ExternalCallError("Search returned malformed evidence.")
    documents = []
    for row in payload["results"]:
        if not isinstance(row, dict):
            continue
        url, content = row.get("url", ""), row.get("content", "")
        if isinstance(url, str) and isinstance(content, str) and content.strip() and official_url(url):
            documents.append(Document(page_content=content, metadata={"passage_id": url, "url": url}))
        if len(documents) == 3:
            break
    return documents


def restricted_search(arguments, tool, *, config=None):
    query = validate_search(arguments)
    if getattr(tool, "max_results", None) != 3:
        raise ValueError("Construct the search tool with max_results=3.")
    # Tavily fixes the result limit at construction; supported invocation fields
    # are supplied here by the application, never forwarded from the model.
    from langchain_core.tools import ToolException
    try:
        result = tool.invoke({"query": query, "include_domains": ["ibm.com"],
                              "search_depth": "basic"}, config=config)
    except ToolException as error:
        # The pinned Tavily integration raises this for a successful empty search.
        # Do not accept arbitrary error strings as evidence or relax the domains.
        if str(error).startswith(f"No search results found for '{query}'. Suggestions:"):
            return []
        raise
    return search_documents(result)


def validate_tool_calls(calls, tool_name, *, limit=2):
    """Validate the whole proposal before executing any of its calls."""
    if len(calls) > limit:
        raise ValueError(f"This experiment permits at most {limit} tool calls.")
    seen = set()
    for call in calls:
        if call.get("name") != tool_name:
            raise ValueError("The model requested an unknown tool.")
        identifier = call.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError("Tool calls need unique nonempty IDs.")
        seen.add(identifier)
        validate_search(call.get("args"))


def redact_email(text):
    """Request-wide teaching rule; not a general personal-data detector."""
    return re.sub(r"[A-Za-z0-9_.+%-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[REDACTED_EMAIL]", text)
