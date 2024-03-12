from __future__ import annotations

from langchain_community.tools import (
    BaseTool,
    BingSearchResults,
    DuckDuckGoSearchResults,
    GoogleSearchResults,
)
from langchain_community.utilities import (
    BingSearchAPIWrapper,
    DuckDuckGoSearchAPIWrapper,
    GoogleSearchAPIWrapper,
)
from pydantic import BaseModel, Field

from config import config


def get_web_searcher():
    if config.search_engine == "duckduckgo":
        wrapper = DuckDuckGoSearchAPIWrapper(max_results=5)
        web_search = DuckDuckGoSearchResults(api_wrapper=wrapper)
    elif config.search_engine == "bing":
        wrapper = BingSearchAPIWrapper(k=5)
        web_search = BingSearchResults(api_wrapper=wrapper)
    else:
        wrapper = GoogleSearchAPIWrapper(k=5)
        web_search = GoogleSearchResults(api_wrapper=wrapper)
    return web_search


class WebSearchInput(BaseModel):
    query: str = Field(
        description="The query to search the Internet. Should be in English or Chinese."
    )


class WebSearchTool(BaseTool):
    name = "web_search"
    cn_name = "网页搜索"
    description = "use this tool when you need to search web page."
    args_schema: type[BaseModel] = WebSearchInput

    def _run(self, query: str) -> str:
        """use string 'query' as input.

        could be any language.
        """
        return get_web_searcher().run(query) + "\n"

    async def _arun(self, query: str) -> str:
        """use string 'query' as input.

        could be any language.
        """
        return get_web_searcher().run(query) + "\n"
