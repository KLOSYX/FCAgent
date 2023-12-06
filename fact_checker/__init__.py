from __future__ import annotations

from datetime import datetime

from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActJsonSingleInputOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.tools.render import render_text_description
from pydantic import BaseModel
from pydantic import Field

from config import Config

__all__ = ['get_fact_checker_agent']
config = Config()


class FactChecker(BaseModel):
    reason: str = Field(
        'Explain why the tweet is real/fake in detail. Output in Markdown format.',
    )
    conclusion: str = Field(
        "The truthfulness of the tweet, could be 'real'/'fake'.",
    )


parser = PydanticOutputParser(pydantic_object=FactChecker)

template = """You are a fact-checking expert. Now I give you the content of a tweet (including the text as well as the \
caption of the image), and the probability that the tweet is true/false predicted by AI model, and some knowledge that \
may be relevant to determining whether the tweet is true or false (note that this knowledge may be irrelevant or false,\
 and that you can't fully trust it); you need to give a conclusion as to the truthfulness of the tweet, and a reason \
 for it, based on this information. today is {time}.

real probability predicted by AI model: {real_prob}

fake probability predicted by AI model: {fake_prob}

tweet text: {claim}

tweet image caption: {image_caption}

AI knowledge: {ai_knowledge}

WiKi knowledge: {wiki_knowledge}

web knowledge: {web_knowledge}

{format_instruction}

output:"""

agent_prompt = PromptTemplate(
    template=template,
    input_variables=[
        'claim', 'image_caption',
        'ai_knowledge', 'wiki_knowledge', 'web_knowledge', 'real_prob', 'fake_prob',
    ],
    partial_variables={
        'format_instruction': parser.get_format_instructions(),
        'time': datetime.now().strftime('%Y-%m-%d'),
    },
)


def get_fact_checker_chain():
    chain = agent_prompt | ChatOpenAI(
        temperature=.7, model_name=config.model_name,
    ) | parser
    return chain


agent_template = """You are a professional fact checker. Given the following tweet text, \
please judge whether the tweet is true or false and give your reasons step by step.
tweet text: {tweet_text}
tweet image path: {tweet_image_path}"""

agent_prompt = PromptTemplate(
    input_variables=['tweet_text', 'tweet_image_path'],
    template=agent_template,
)


def get_fact_checker_agent(tools):
    llm = ChatOpenAI(temperature=.7, model_name=config.model_name)
    prompt = hub.pull('hwchase17/react-json')
    prompt = prompt.partial(
        tools=render_text_description(tools),
        tool_names=', '.join([t.name for t in tools]),
    )
    llm_with_stop = llm.bind(stop=['\nObservation'])
    agent = (
        {
            'input': lambda x: x['input'],
            'agent_scratchpad': lambda x: format_log_to_str(x['intermediate_steps']), }
        | prompt
        | llm_with_stop
        | ReActJsonSingleInputOutputParser()
    )
    agent = agent_prompt | {'input': lambda x: x} | AgentExecutor(
        agent=agent, tools=tools, verbose=True, handle_parsing_errors='Check your output and make sure it conforms!\n',
    )
    return agent
