from __future__ import annotations

import json
import operator
from datetime import datetime
from typing import Annotated, List, TypedDict, Union

from agents_deconstructed import react_chat
from agents_deconstructed.format_tools import format_tools_args
from langchain.agents import (
    create_json_chat_agent,
    create_openai_tools_agent,
    create_structured_chat_agent,
)
from langchain.prompts import MessagesPlaceholder, PromptTemplate
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt.tool_executor import ToolExecutor
from loguru import logger
from pyrootutils import setup_root

from config import Config
from utils import react_chat

__all__ = ["get_fact_checker_agent"]
config = Config()

ROOT = setup_root(".")

agent_template = """你是一名专业的事实核查机构编辑，给定如下的推文文本内容\
以及推文图片名称，请借助工具，核查问题是否真实，并给出你的判断依据。\
核查结束后，请用“核查结束：(你的结论)”结尾。
当前日期：{date}
待核查文本：{tweet_text}
待核查图片：{tweet_image_name}"""

en_agent_template = """You are a professional fact checking agency editor, providing the following tweet text content \
as well as the tweet image name, please use tools to verify whether the claim is true or false and provide your \
judgment basis.
Current date: {date}
Text to be verified: {tweet_text}
Image to be verified: {tweet_image_name}
"""

agent_prompt = PromptTemplate(
    input_variables=["tweet_text", "tweet_image_name"],
    template=en_agent_template,
    partial_variables={
        "date": datetime.now().strftime("%Y-%m-%d"),
    },
)


class AgentState(TypedDict):
    input: dict | str
    chat_history: list[BaseMessage]
    agent_outcome: list | AgentAction | AgentFinish | None
    intermediate_steps: list[tuple[AgentAction, str]]


def _get_agent(agent_name: str, llm, tools):
    if agent_name == "openai_tools":
        with open(ROOT / "prompt" / "openai-tools-agent.json") as f:
            prompt_raw = json.load(f)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompt_raw["system"]),
                MessagesPlaceholder("chat_history", optional=True),
                ("human", prompt_raw["human"]),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )
        return create_openai_tools_agent(llm, tools, prompt)
    elif agent_name == "react_json":
        with open(ROOT / "prompt" / "react_json.json") as f:
            prompt_raw = json.load(f)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompt_raw["system"]),
                MessagesPlaceholder("chat_history", optional=True),
                ("human", prompt_raw["human"]),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )
        return create_json_chat_agent(llm, tools, prompt)
    elif agent_name == "structured_chat":
        with open(ROOT / "prompt" / "structured_chat.json") as f:
            prompt_raw = json.load(f)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompt_raw["system"]),
                MessagesPlaceholder("chat_history", optional=True),
                ("human", prompt_raw["human"]),
            ]
        )
        return create_structured_chat_agent(llm, tools, prompt)
    elif agent_name == "shoggoth13_react_json":
        llm.bind(stop=["Observation"])
        with open(ROOT / "prompt" / "shoggoth13_react_json.json") as f:
            prompt_raw = json.load(f)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompt_raw["system"]),
                ("human", prompt_raw["human"]),
                MessagesPlaceholder("intermediate_steps"),
            ]
        )
        prompt = prompt.partial(
            tools=format_tools_args(tools), tool_names=", ".join([tool.name for tool in tools])
        )
        agent = (
            RunnablePassthrough.assign(
                intermediate_steps=lambda x: react_chat.format_steps(x["intermediate_steps"]),
            )
            | prompt
            | llm
            | react_chat.ReActOutputParser()
        )
        return agent


def get_fact_checker_agent(tools):
    tool_executor = ToolExecutor(tools)
    llm = ChatOpenAI(
        model_name=config.model_name,
        streaming=True,
    )
    agent = _get_agent(config.agent_type, llm, tools)

    async def init_agent(data: AgentState):
        logger.debug(f"init agent with agent state: {data}")
        msg = await agent_prompt.ainvoke(data["input"])
        return {"input": msg.text, "intermediate_steps": [], "chat_history": []}

    async def run_agent(data: AgentState):
        logger.debug(f"run agent with agent state: {data}")
        agent_outcome = await agent.ainvoke(data)
        return {"agent_outcome": agent_outcome}

    async def execute_tools(data: AgentState):
        logger.debug(f"execute tools with agent state: {data}")
        agent_actions = data["agent_outcome"]
        old_steps = data.copy()["intermediate_steps"]
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
        return {"intermediate_steps": old_steps + steps}

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
