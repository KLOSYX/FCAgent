import asyncio
import json
import operator
from enum import Enum
from typing import Annotated, Literal, TypedDict

from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_openai.chat_models import ChatOpenAI
from langgraph.graph import END, StateGraph

from config import config
from utils import load_base_tools, tool_exception_catch
from utils.pydantic import PydanticOutputParser

RETRIEVER_MAP = {x.name: x for x in load_base_tools("retriever", ["QueryRouterTool"])}


class ToolSelectScheme(BaseModel):
    tool_name: str = Field(description="name of the tool")


class AdequacyScheme(BaseModel):
    yes_or_no: Literal["yes", "no"]


llm = ChatOpenAI(model_name=config.model_name, temperature=0, streaming=False)

classification_parser = PydanticOutputParser(pydantic_object=ToolSelectScheme)
CLASSIFICATION_PROMPT_TEMPLATE = """Given the query "{query}", Which of the following do you think is the best search \
tool?
{tools}
---
{format_instructions}
"""
CLASSIFICATION_PROMPT = PromptTemplate(
    template=CLASSIFICATION_PROMPT_TEMPLATE,
    input_variables=["query", "tools"],
).partial(format_instructions=classification_parser.get_format_instructions())
classification_chain = CLASSIFICATION_PROMPT | llm | classification_parser

ADEQUACY_PROMPT_TEMPLATE = """Given the query, do you think the search results search results is an adequacy?
Query: {query}
Search Results: {search_results}
---
{format_instructions}
"""
adequacy_parser = PydanticOutputParser(pydantic_object=AdequacyScheme)
ADEQUACY_PROMPT = PromptTemplate(
    template=ADEQUACY_PROMPT_TEMPLATE,
    input_variables=["query", "search_results"],
).partial(format_instructions=adequacy_parser.get_format_instructions())
adequacy_chain = ADEQUACY_PROMPT | llm | adequacy_parser


class QueryType(Enum):
    COMMON = "common_sense"
    NEWS = "news_or_time_sensitive"
    UNDETERMINED = "undetermined"
    OTHER = "other"


class RouterState(TypedDict):
    query: str
    search_results: Annotated[list[str], operator.add]
    retriever: None | str
    is_enough: bool


async def init_router_state(state: RouterState) -> dict:
    return {"search_results": [], "retriever": None, "is_enough": False}


async def dispatch(state: RouterState) -> dict:
    retrievers = list(RETRIEVER_MAP.values())
    tools = json.dumps(
        [{"tool_name": x.name, "tool_description": x.description} for x in retrievers],
        ensure_ascii=False,
    )
    res: ToolSelectScheme = await classification_chain.ainvoke(
        {"query": state["query"], "tools": tools}
    )
    return {"retriever": res.tool_name}


async def router(state: RouterState) -> dict:
    knowledge = []
    if state["retriever"] in RETRIEVER_MAP:
        knowledge.append(
            str(await RETRIEVER_MAP[state["retriever"]].ainvoke({"query": state["query"]}))
        )
    return {"search_results": state["search_results"] + knowledge}


async def gather_all(state: RouterState) -> dict:
    if state["is_enough"]:
        return {}
    tasks = []
    for retriever_name in RETRIEVER_MAP.keys():
        if retriever_name != state["retriever"]:
            tasks.append(RETRIEVER_MAP[retriever_name].ainvoke({"query": state["query"]}))
    knowledge = await asyncio.gather(*tasks)
    return {"search_results": state["search_results"] + list(map(str, knowledge))}


async def enough(state: RouterState) -> dict:
    is_enough = await adequacy_chain.ainvoke(
        {"query": state["query"], "search_results": state["search_results"]}
    )
    return {"is_enough": is_enough.yes_or_no == "Yes"}


workflow = StateGraph(RouterState)

workflow.add_node("start", init_router_state)
workflow.add_node("dispatch", dispatch)
workflow.add_node("router", router)
# workflow.add_node("enough", enough)
# workflow.add_node("gather_all", gather_all)

workflow.set_entry_point("start")

workflow.add_edge("start", "dispatch")
workflow.add_edge("dispatch", "router")
workflow.add_edge("router", END)
# workflow.add_edge("router", "enough")
# workflow.add_edge("enough", "gather_all")
# workflow.add_edge("gather_all", END)

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

    @tool_exception_catch(name)
    def _run(self, query: str) -> str:
        res = app.invoke({"query": query})
        return res["search_results"]

    @tool_exception_catch(name)
    async def _arun(self, query: str) -> str:
        res = await app.ainvoke({"query": query})
        return res["search_results"]


async def main():
    res = await app.ainvoke({"query": "The capital of China"})
    return res


if __name__ == "__main__":
    print(asyncio.run(main()))
