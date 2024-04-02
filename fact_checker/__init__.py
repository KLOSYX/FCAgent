from __future__ import annotations

import json
import operator
from datetime import datetime
from typing import Annotated, TypedDict, Union

from langchain.prompts import PromptTemplate
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt.tool_executor import ToolExecutor
from loguru import logger
from pyrootutils import setup_root

from config import config
from fact_checker.get_agent import _get_agent

__all__ = ["get_fact_checker_agent"]

ROOT = setup_root(".")


class AgentState(TypedDict):
    input: dict | str
    chat_history: list[BaseMessage]
    agent_outcome: list[AgentAction] | AgentAction | AgentFinish | None
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]


def get_fact_checker_agent(tools):
    tool_executor = ToolExecutor(tools)
    llm = ChatOpenAI(
        model_name=config.model_name,
        streaming=True,
        stop=["✿RESULT✿"],
        temperature=0.0,
    )
    agent = _get_agent(config.agent_type, llm, tools)

    async def init_agent(data: AgentState):
        logger.debug(f"init agent with agent state: {data}")
        msg = await AGENT_PROMPT.ainvoke(data["input"])
        return {"input": msg.text, "intermediate_steps": [], "chat_history": []}

    async def run_agent(data: AgentState):
        logger.debug(f"run agent with agent state: {data}")
        agent_outcome = await agent.ainvoke(data)
        return {"agent_outcome": agent_outcome}

    async def execute_tools(data: AgentState):
        logger.debug(f"execute tools with agent state: {data}")
        ret = {}
        agent_actions = data["agent_outcome"]
        prev_steps = data.copy()["intermediate_steps"]
        steps = []
        if isinstance(agent_actions, list):
            for action in agent_actions:
                if not isinstance(action, AgentAction):
                    continue
                output = await tool_executor.ainvoke(action)
                steps.append((action, str(output)))
        elif isinstance(agent_actions, AgentAction):
            output = await tool_executor.ainvoke(agent_actions)
            steps.append((agent_actions, str(output)))
        ret["intermediate_steps"] = prev_steps + steps
        return ret

    def should_continue(data):
        logger.debug(f"should continue with agent state: {data}")
        if isinstance(data["agent_outcome"], AgentFinish):
            if (
                "核查结束" in data["agent_outcome"].messages[0].content
                or config.agent_type == "shoggoth13_react_json"
            ):
                logger.debug("agent stopped")
                return "end"
            else:
                logger.debug("agent re-enter")
                return "agent"
        else:
            return "continue"

    # Define a new graph
    workflow = StateGraph(AgentState)

    workflow.add_node("start", init_agent)
    workflow.add_node("agent", run_agent)
    workflow.add_node("action", execute_tools)

    workflow.set_entry_point("start")

    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "action",
            "agent": "agent",
            "end": END,
        },
    )

    workflow.add_edge("start", "agent")
    workflow.add_edge("action", "agent")

    app = workflow.compile()

    return app


CN_AGENT_TEMPLATE = """你是一名专业的事实核查机构编辑，给定如下的推文文本内容\
以及推文图片名称，请借助工具，核查问题是否真实，并给出你的判断依据。
当前日期：{date}
待核查文本：{tweet_text}
待核查图片：{tweet_image_name}"""

EN_AGENT_TEMPLATE = """You are a professional fact checking agency editor, providing the following tweet text content \
as well as the tweet image name, please use tools to verify whether the claim is true or false and provide your \
judgment basis.
Current date: {date}
Text to be verified: {tweet_text}
Image to be verified: {tweet_image_name}
"""

AGENT_PROMPT = PromptTemplate(
    input_variables=["tweet_text", "tweet_image_name"],
    template=CN_AGENT_TEMPLATE,
    partial_variables={
        "date": datetime.now().strftime("%Y-%m-%d"),
    },
)
