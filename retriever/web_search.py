from __future__ import annotations

from langchain.tools import BaseTool
from langchain.tools import DuckDuckGoSearchResults
from langchain.utilities import DuckDuckGoSearchAPIWrapper


def get_web_searcher():
    wrapper = DuckDuckGoSearchAPIWrapper(max_results=5)
    web_search = DuckDuckGoSearchResults(api_wrapper=wrapper)
    return web_search


class WebSearchTool(BaseTool):
    name = 'web search tool'
    description = 'use this tool when you need to search web page. note that results might be fake.'

    def _run(self, query: str) -> str:
        return get_web_searcher().run(query)

    def _arun(self, query: str) -> list[str]:
        raise NotImplementedError('This tool does not support async')
