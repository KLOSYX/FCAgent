"""Adopted from Llama Index.

https://github.com/jerryjliu/llama_index/blob/main/llama_index/agent/react/output_parser.py
"""
import ast
import re
from typing import Tuple, Union

from langchain.schema import AgentAction, AgentFinish
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.schema.output_parser import BaseOutputParser


def extract_tool_use(input_text: str) -> tuple[str, str, str]:
    # pattern = r"\s*Thought:(.*?)Action:(.*?)Action Input:(.*?)(?:\n|$)"
    pattern = r"\s*(.*?)✿FUNCTION✿:(.*?)✿ARGS✿:(.*?)(?:\n|$)"

    match = re.search(pattern, input_text, re.DOTALL)
    if not match:
        raise ValueError(f"Could not extract tool use from input text: {input_text}")

    try:
        thought = match.group(1).strip()
    except AttributeError:
        thought = ""
    thought = thought.replace("✿THOUGHT✿:", "").replace("\n", " ").strip()
    if not thought:
        thought = "I should use tools to help me solve this."
    action = match.group(2).strip()
    action_input = match.group(3).strip()
    return thought, action, action_input


def extract_final_response(input_text: str) -> tuple[str, str]:
    pattern = r"\s*✿THOUGHT✿:(.*?)✿RETURN✿:(.*?)(?:\n|$)"

    match = re.search(pattern, input_text, re.DOTALL)
    if not match:
        raise ValueError(f"Could not extract final answer from input text: {input_text}")

    thought = match.group(1).strip()
    answer = match.group(2).strip()
    return thought, answer


def extract_json_str(text: str) -> str:
    """Extract JSON string from text."""
    # NOTE: this regex parsing is taken from langchain.output_parsers.pydantic
    match = re.search(r"\{.*\}", text.strip(), re.MULTILINE | re.IGNORECASE | re.DOTALL)
    if not match:
        raise ValueError(f"Could not extract json string from output: {text}")

    return match.group()


class ReActOutputParser(BaseOutputParser):
    def parse(self, text: str) -> AgentAction | AgentFinish:
        """Parse output from ReAct agent.

        We expect the output to be in one of the following formats:
        1. If the agent need to use a tool to answer the question:
            ```
            Thought: <thought>
            Action: <action>
            Action Input: <action_input>
            ```
        2. If the agent can answer the question without any tools:
            ```
            Thought: <thought>
            Answer: <answer>
            ```
        """
        if "✿RETURN✿" in text:
            thought, answer = extract_final_response(text)
            return AgentFinish(log=thought, return_values={"output": answer})

        if "✿FUNCTION✿" in text:
            thought, action, action_input = extract_tool_use(text)
            json_str = extract_json_str(action_input)

            # NOTE: we found that json.loads does not reliably parse
            # json with single quotes, so we use ast instead
            # action_input_dict = json.loads(json_str)
            action_input_dict = ast.literal_eval(json_str)

            return AgentAction(log=thought, tool=action, tool_input=action_input_dict)

        if "✿THOUGHT✿" not in text:
            # NOTE: handle the case where the agent directly outputs the answer
            # instead of following the thought-answer format
            return AgentFinish(
                log="I can answer without any tools.", return_values={"output": text}
            )

        raise ValueError(f"Could not parse output: {text}")


def format_steps(intermediate_steps):
    log = []
    for action, observation in intermediate_steps:
        log.extend(
            [
                AIMessage(
                    content="\n\n".join(
                        (
                            f"✿THOUGHT✿: {action.log}",
                            f"✿FUNCTION✿: {action.tool}",
                            f"✿ARGS✿: {action.tool_input}",
                        )
                    )
                ),
                HumanMessage(
                    content=f"✿RESULT✿: {observation}\n\n",
                ),
            ]
        )
    return log


REACT_AGENT_INSTRUCTIONS = """\

You are designed to help with a variety of tasks, from answering questions \
    to providing summaries to other types of analyses.

## Tools
You have access to a wide variety of tools. You are responsible for using
the tools in any sequence you deem appropriate to complete the task at hand.
This may require breaking the task into subtasks and using different tools
to complete each subtask.

You have access to the following tools:
{tool_desc}

## Output Format
To answer the question, please use the following format.

```
Thought: I need to use a tool to help me answer the question.
Action: tool name (one of {tool_names})
Action Input: the input to the tool, in a JSON format representing the kwargs (e.g. {{"text": "hello world", "num_beams": 5}})
```
Please use a valid JSON format for the action input. Do NOT do this {{'text': 'hello world', 'num_beams': 5}}.

If this format is used, the user will respond in the following format:

```
Observation: tool response
```

You should keep repeating the above format until you have enough information
to answer the question without using any more tools. At that point, you MUST respond
in the following format:

```
Thought: I can answer without using any more tools.
Answer: [your answer here]
```

## Current Conversation
Below is the current conversation consisting of interleaving human and assistant messages.

"""
