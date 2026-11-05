"""Construct the web search integration."""
from ai_lib.models.configuration import require_key


def build_web_search():
    from langchain_tavily import TavilySearch
    require_key("TAVILY_API_KEY")
    return TavilySearch(
        handle_tool_error=False,
        search_depth="basic", max_results=3, include_domains=["cloud.ibm.com", "ibm.com"],
        include_answer=False, include_raw_content=False,
        description=("Search current official IBM documentation. Use for explicit current-version "
                     "questions or information missing from the local documentation snapshot. "
                     "Input a focused search query; results are evidence, not a finished answer."),
    )
