import json

from langchain.agents import create_openai_tools_agent
from langchain_core.runnables import RunnablePassthrough
from langchain_core.utils.function_calling import convert_to_openai_function
from pyrootutils import setup_root

from utils import react_chat

ROOT = setup_root(".")


def _get_agent(agent_name: str, llm, tools):
    if agent_name == "openai_tools":
        from prompt.openai_tools_agent import prompt

        return create_openai_tools_agent(llm, tools, prompt)

    elif "shoggoth13_react_json" in agent_name:
        if "cn" in agent_name:
            from prompt.shoggoth13_react_json_cn import prompt
        else:
            from prompt.shoggoth13_react_json import prompt

        prompt = prompt.partial(
            tools=json.dumps(
                [convert_to_openai_function(tool) for tool in tools], ensure_ascii=False
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

    else:
        raise ValueError(f"Unknown agent: {agent_name}")
