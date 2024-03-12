from __future__ import annotations

import json
from datetime import datetime

from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
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


agent_template = """你是一名专业的事实核查机构编辑，给定如下的推文文本内容 \
以及推文图片路径，请一步一步地判断推文内容是否真实，并给出你的判断依据。 \
当前日期：{date}
推文文本：{tweet_text}
推文图片路径：{tweet_image_path}"""

agent_prompt = PromptTemplate(
    input_variables=["tweet_text", "tweet_image_path"],
    template=agent_template,
    partial_variables={
        "date": datetime.now().strftime("%Y-%m-%d"),
    },
)


def get_fact_checker_agent(tools):
    llm = ChatOpenAI(
        model_name=config.model_name,
        streaming=True,
    )
    with open(ROOT / "prompt" / "openai-tools-agent.json") as f:
        prompt_raw = json.load(f)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_raw["system"]),
            ("human", prompt_raw["human"]),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
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
