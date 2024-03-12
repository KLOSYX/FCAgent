from __future__ import annotations

import json
from datetime import datetime

from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActJsonSingleInputOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.tools.render import render_text_description_and_args
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from pydantic import BaseModel, Field
from pyrootutils import setup_root

from config import Config

__all__ = ["get_fact_checker_agent"]
config = Config()

ROOT = setup_root(".")


agent_template = """You are a professional fact checker. Given the following tweet text \
and tweet image path, please judge whether the tweet is true or false and \
give your reasons step by step in Chinese. Current date: {date}
tweet text: {tweet_text}
tweet image path: {tweet_image_path}"""

agent_prompt = PromptTemplate(
    input_variables=["tweet_text", "tweet_image_path"],
    template=agent_template,
    partial_variables={
        "date": datetime.now().strftime("%Y-%m-%d"),
    },
)


def get_fact_checker_agent(tools):
    llm = ChatOpenAI(
        temperature=0.7,
        model_name=config.model_name,
        streaming=True,
    )
    with open(ROOT / "prompt" / "react_json.json") as f:
        prompt_raw = json.load(f)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_raw["system"]),
            ("human", prompt_raw["human"]),
        ]
    )
    prompt = prompt.partial(
        tools=render_text_description_and_args(tools),
        tool_names=", ".join([t.name for t in tools]),
    )
    llm_with_stop = llm.bind(stop=["\nObservation"])
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
        }
        | prompt
        | llm_with_stop
        | ReActJsonSingleInputOutputParser()
    )
    chain = (
        agent_prompt
        | {"input": lambda x: x}
        | AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors="请检查调用工具的格式！\n",
        )
    )
    return chain
