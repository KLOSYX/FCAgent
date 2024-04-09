import asyncio
import operator
from enum import Enum
from typing import Annotated, Literal, TypedDict

from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain_core.tools import BaseTool
from langchain_openai.chat_models import ChatOpenAI
from langgraph.graph import END, StateGraph

from config import config
from retriever import ask_llm, web_search, wikipedia
from utils.pydantic import PydanticOutputParser


class ClassificationScheme(BaseModel):
    yes_or_no: Literal["Yes", "No"] = None


llm = ChatOpenAI(model_name=config.model_name, temperature=0, streaming=False)

classification_parser = PydanticOutputParser(pydantic_object=ClassificationScheme)
CLASSIFICATION_PROMPT_TEMPLATE = """Given the query "{query}", please determine if it is about "{type}".
---
{format_instructions}
"""
CLASSIFICATION_PROMPT = PromptTemplate(
    template=CLASSIFICATION_PROMPT_TEMPLATE,
    input_variables=["query", "type"],
).partial(format_instructions=classification_parser.get_format_instructions())
classification_chain = CLASSIFICATION_PROMPT | llm | classification_parser

ADEQUACY_PROMPT_TEMPLATE = """Given the query, do you think the search results search results is an adequacy?
Query: {query}
Search Results: {search_results}
---
{format_instructions}
"""
ADEQUACY_PROMPT = PromptTemplate(
    template=ADEQUACY_PROMPT_TEMPLATE,
    input_variables=["query", "search_results"],
).partial(format_instructions=classification_parser.get_format_instructions())
adequacy_chain = ADEQUACY_PROMPT | llm | classification_parser


class QueryType(Enum):
    COMMON = "common_sense"
    NEWS = "news_or_time_sensitive"
    UNDETERMINED = "undetermined"
    OTHER = "other"


class RouterState(TypedDict):
    query: str
    search_results: Annotated[list[str], operator.add]
    type: None | QueryType
    is_enough: bool


async def init_router_state(state: RouterState) -> dict:
    return {"search_results": [], "type": None, "is_enough": False}


async def common_sense(state: RouterState) -> dict:
    is_common_sense = await classification_chain.ainvoke(
        {"query": state["query"], "type": "common sense or encyclopedia"}
    )
    if is_common_sense.yes_or_no == "Yes":
        return {"type": QueryType.COMMON}
    return {"type": QueryType.UNDETERMINED}


async def news_or_time_sensitive(state: RouterState) -> dict:
    is_news_or_time_sensitive = await classification_chain.ainvoke(
        {"query": state["query"], "type": "news or time sensitive"}
    )
    if is_news_or_time_sensitive.yes_or_no == "Yes":
        return {"type": QueryType.NEWS}
    return {"type": QueryType.OTHER}


async def router(state: RouterState) -> dict:
    knowledge = []
    if state["type"] == QueryType.COMMON:
        knowledge.append(str(await wikipedia.WikipediaTool().ainvoke({"query": state["query"]})))
    elif state["type"] == QueryType.NEWS:
        knowledge.append(str(await web_search.WebSearchTool().ainvoke({"query": state["query"]})))
    elif state["type"] == QueryType.OTHER:
        knowledge.append(str(await ask_llm.AskLlmTool().ainvoke({"question": state["query"]})))
    return {"search_results": state["search_results"] + knowledge}


async def gather_all(state: RouterState) -> dict:
    if state["is_enough"]:
        return {}
    tasks = []
    if state["type"] != QueryType.COMMON:
        tasks.append(wikipedia.WikipediaTool().ainvoke({"query": state["query"]}))
    if state["type"] != QueryType.NEWS:
        tasks.append(web_search.WebSearchTool().ainvoke({"query": state["query"]}))
    if state["type"] != QueryType.OTHER:
        tasks.append(ask_llm.AskLlmTool().ainvoke({"question": state["query"]}))
    knowledge = await asyncio.gather(*tasks)
    return {"search_results": state["search_results"] + list(map(str, knowledge))}


async def enough(state: RouterState) -> dict:
    is_enough = await adequacy_chain.ainvoke(
        {"query": state["query"], "search_results": state["search_results"]}
    )
    return {"is_enough": is_enough.yes_or_no == "Yes"}


def conditional_edge(state: RouterState) -> str:
    if state["type"] == QueryType.UNDETERMINED:
        return "news_or_time_sensitive"
    if state["type"] in (QueryType.COMMON, QueryType.NEWS, QueryType.OTHER):
        return "router"


workflow = StateGraph(RouterState)

workflow.add_node("start", init_router_state)
workflow.add_node("common_sense", common_sense)
workflow.add_node("news_or_time_sensitive", news_or_time_sensitive)
workflow.add_node("router", router)
workflow.add_node("enough", enough)
workflow.add_node("gather_all", gather_all)

workflow.set_entry_point("start")

workflow.add_edge("start", "common_sense")
workflow.add_conditional_edges(
    "common_sense",
    conditional_edge,
    {
        "news_or_time_sensitive": "news_or_time_sensitive",
        "router": "router",
    },
)
workflow.add_edge("news_or_time_sensitive", "router")
workflow.add_edge("router", "enough")
workflow.add_edge("enough", "gather_all")
workflow.add_edge("gather_all", END)

app = workflow.compile()


class SearchInput(BaseModel):
    query: str = Field(
        description="The query to search the Internet. Should be in English or Chinese."
    )


class QueryRouterTool(BaseTool):
    name = "search"
    cn_name = "聚合搜索"
    is_multimodal: bool = False
    description = "use this tool when you need to search any information."
    args_schema: type[BaseModel] = SearchInput

    def _run(self, query: str) -> str:
        res = app.invoke({"query": query})
        return res["search_results"]

    async def _arun(self, query: str) -> str:
        res = await app.ainvoke({"query": query})
        return res["search_results"]


async def main():
    res = await app.ainvoke({"query": "The capital of China"})
    return res


if __name__ == "__main__":
    print(asyncio.run(main()))
