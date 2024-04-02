import json

from agents_deconstructed.format_tools import format_tools_args
from langchain.agents import (
    create_json_chat_agent,
    create_openai_tools_agent,
    create_structured_chat_agent,
)
from langchain.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.utils.function_calling import format_tool_to_openai_function
from pyrootutils import setup_root

from utils import react_chat

ROOT = setup_root(".")


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
        with open(ROOT / "prompt" / "shoggoth13_react_json.json", encoding="utf-8") as f:
            prompt_raw = json.load(f)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompt_raw["system"]),
                ("human", prompt_raw["human"]),
                MessagesPlaceholder("intermediate_steps"),
            ]
        )
        prompt = prompt.partial(
            tools=json.dumps(
                [format_tool_to_openai_function(tool) for tool in tools], ensure_ascii=False
            ),
            tool_names=", ".join([tool.name for tool in tools]),
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
