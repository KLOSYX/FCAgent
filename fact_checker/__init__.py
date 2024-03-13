from __future__ import annotations

import json
import operator
from datetime import datetime
from typing import Annotated, List, TypedDict, Union

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import MessagesPlaceholder, PromptTemplate
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt.tool_executor import ToolExecutor
from pyrootutils import setup_root

from config import Config

__all__ = ["get_fact_checker_agent"]
config = Config()

ROOT = setup_root(".")

agent_template = """你是一名专业的事实核查机构编辑，给定如下的推文文本内容 \
以及推文图片路径，请首先将内容分解为待核查的子问题清单:
1. xxx
2. xxx
...
随后借助工具，逐一核查子问题是否真实，并给出你的判断依据。 \
所有子问题核查结束后，请用“核查结束：(你的结论)”结尾。
当前日期：{date}
待核查文本：{tweet_text}
待核查图片：{tweet_image_name}"""

agent_prompt = PromptTemplate(
    input_variables=["tweet_text", "tweet_image_name"],
    template=agent_template,
    partial_variables={
        "date": datetime.now().strftime("%Y-%m-%d"),
    },
)


class AgentState(TypedDict):
    input: str
    chat_history: list[BaseMessage]
    agent_outcome: AgentAction | AgentFinish | None
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]


def get_fact_checker_agent(tools):
    tool_executor = ToolExecutor(tools)
    llm = ChatOpenAI(
        model_name=config.model_name,
        streaming=True,
    )
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
    agent = create_openai_tools_agent(llm, tools, prompt)

    async def init_agent(data):
        inputs = data["input"]
        msg = await agent_prompt.ainvoke(inputs)
        return {"input": msg.text, "intermediate_steps": [], "chat_history": []}

    async def run_agent(data):
        agent_outcome = await agent.ainvoke(data)
        return {"agent_outcome": agent_outcome}

    async def execute_tools(data):
        agent_actions = data["agent_outcome"]
        steps = []
        for action in agent_actions:
            output = await tool_executor.ainvoke(action)
            steps.append((action, str(output)))
        agent_actions.clear()
        return {"intermediate_steps": steps}

    def should_continue(data):
        if isinstance(data["agent_outcome"], AgentFinish):
            if "核查结束" in data["agent_outcome"].messages[0].content:
                return "end"
            else:
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
