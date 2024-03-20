from __future__ import annotations

from langchain.prompts import PromptTemplate
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
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai.chat_models import ChatOpenAI

from config import config
from utils.pydantic import PydanticOutputParser

SEARCH_PROMPT = """Given the query and search results, you need to extract key information and information sources \
from the search results:
query: {query}
search results: {search_results}

{format_instructions}
"""


class SearchResult(BaseModel):
    results: list[tuple[str, str]] = Field(
        description="Formatted search results, output in \
    (information to the query, URL) format."
    )


# Set up a parser
parser = PydanticOutputParser(pydantic_object=SearchResult)
prompt = PromptTemplate(
    template=SEARCH_PROMPT, input_variables=["query", "search_results"]
).partial(format_instructions=parser.get_format_instructions())
llm = ChatOpenAI(model_name=config.model_name, temperature=0.0)


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


chain = prompt | llm | parser


def format_search_results(results: SearchResult) -> str:
    ret = "Search Results: "
    for i, (res, url) in enumerate(results.results):
        ret += f"{i + 1}: {res}\tSource: {url}\t"
    return ret


class WebSearchTool(BaseTool):
    name = "web_search"
    cn_name = "网页搜索"
    is_multimodal: bool = False
    description = "use this tool when you need to search web page."
    args_schema: type[BaseModel] = WebSearchInput

    def _run(self, query: str) -> str:
        search_results = get_web_searcher().run(query)
        res = chain.invoke({"query": query, "search_results": search_results})
        return format_search_results(res)

    async def _arun(self, query: str) -> str:
        search_results = get_web_searcher().run(query)
        res = await chain.ainvoke({"query": query, "search_results": search_results})
        return format_search_results(res)
