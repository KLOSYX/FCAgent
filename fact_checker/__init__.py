from __future__ import annotations

import ast
import json
import operator
from datetime import datetime
from typing import Annotated, TypedDict, Union

from langchain.prompts import PromptTemplate
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt.tool_executor import ToolExecutor
from loguru import logger
from pyrootutils import setup_root

from config import config
from fact_checker.get_agent import _get_agent
from tools.summarizer import SummarizerScheme, get_summarizer_chain
from utils.react_chat import format_steps

__all__ = ["get_fact_checker_agent"]

ROOT = setup_root(".")


class AgentState(TypedDict):
    input: dict | str
    chat_history: list[BaseMessage]
    agent_outcome: list[AgentAction] | AgentAction | AgentFinish | None
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
    final_output: None | SummarizerScheme


def get_fact_checker_agent(tools):
    tool_executor = ToolExecutor(tools)
    llm = ChatOpenAI(
        model_name=config.model_name,
        streaming=True,
        stop=["✿RESULT✿", "\n\n\n"],
        temperature=0.0,
    )
    agent = _get_agent(config.agent_type, llm, tools)
    if config.use_ocr:
        import os

        from paddleocr import PaddleOCR

        os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
        ocr = PaddleOCR(use_angle_cls=True, lang="ch")
    else:
        ocr = None

    def _get_ocr_result(img_name: str) -> str:
        img_path = ROOT / ".temp" / img_name
        ocr_res = ocr.ocr(str(img_path), cls=True)
        res_list = []
        for idx in range(len(ocr_res[0])):
            text, score = ocr_res[0][idx][-1]
            if score > 0.1:
                res_list.append(text.strip())
        return ", ".join(res_list)

    async def init_agent(data: AgentState):
        logger.debug(f"init agent with agent state: {data}")
        tweet_image_name = data["input"]["tweet_image_name"]
        if tweet_image_name != "No image" and config.use_ocr:
            ref_image_ocr_res = _get_ocr_result(tweet_image_name)
        else:
            ref_image_ocr_res = "No OCR result"
        data["input"]["ref_image_ocr_res"] = ref_image_ocr_res
        msg = await AGENT_PROMPT.ainvoke(data["input"])
        init_thought = "I need to test if the tool calls properly."
        init_action = "test_tool"
        init_action_input = ast.literal_eval(json.dumps({"params": "Hello World!"}))
        init_observation = "The tool call is working properly."
        intermediate_steps = [
            (
                AgentAction(log=init_thought, tool=init_action, tool_input=init_action_input),
                init_observation,
            )
        ]
        return {
            "input": msg.text,
            "intermediate_steps": intermediate_steps,
            "chat_history": [],
            "final_output": None,
        }

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

    async def run_summarizer(data: AgentState):
        logger.debug(f"run summarizer with agent state: {data}")
        summarizer = get_summarizer_chain()
        procedures: list = format_steps(data["intermediate_steps"])
        history = "".join(x.content for x in procedures)
        res: SummarizerScheme = await summarizer.ainvoke(
            {"claim_text": data["input"], "history": history}
        )
        return {"final_output": res}

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
    workflow.add_node("summarize", run_summarizer)

    workflow.set_entry_point("start")

    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "action",
            "agent": "agent",
            "end": "summarize",
        },
    )

    workflow.add_edge("start", "agent")
    workflow.add_edge("action", "agent")
    workflow.add_edge("summarize", END)

    app = workflow.compile()

    return app


CN_AGENT_TEMPLATE = """你是一名专业的事实核查机构编辑，给定如下的推文文本内容\
以及推文图片名称，请借助工具，核查问题是否真实，并给出你的判断依据。
当前日期：{date}
待核查文本：{tweet_text}
待核查图片：{tweet_image_name}
参考图片OCR结果：{ref_image_ocr_res}
"""

EN_AGENT_TEMPLATE = """You are a professional fact checking agency editor, providing the following tweet text content \
as well as the tweet image name, please use tools to verify whether the claim is true or false.
Current date: {date}
Text to be verified: {tweet_text}
Image to be verified: {tweet_image_name}
Reference image OCR result: {ref_image_ocr_res}
"""

AGENT_PROMPT = PromptTemplate(
    input_variables=["tweet_text", "tweet_image_name", "ref_image_ocr_res"],
    template=EN_AGENT_TEMPLATE,
    partial_variables={
        "date": datetime.now().strftime("%Y-%m-%d"),
    },
)
