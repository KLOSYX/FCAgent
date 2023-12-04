from __future__ import annotations

from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.tools import StructuredTool
from pydantic import BaseModel
from pydantic import Field

from config import Config
from retriever import ClosedBookTool
from retriever import WebSearchTool
from retriever import WikipediaTool
from tools import FakeNewsDetectionTool

__all__ = ['get_fact_checker_chain']
config = Config()


class FactChecker(BaseModel):
    reason: str = Field(
        'Explain why the tweet is real/fake in detail. Output in Markdown format.',
    )
    conclusion: str = Field(
        "The truthfulness of the tweet, could be 'real'/'fake'.",
    )


parser = PydanticOutputParser(pydantic_object=FactChecker)

template = """You are a fact-checking expert. Now I give you the content of a tweet (including the text as well as the caption of the image), and the probability that the tweet is true/false predicted by AI model, and some knowledge that may be relevant to determining whether the tweet is true or false (note that this knowledge may be irrelevant or false, and that you can't fully trust it); you need to give a conclusion as to the truthfulness of the tweet, and a reason for it, based on this information.

real probability predicted by AI model: {real_prob}

fake probability predicted by AI model: {fake_prob}

tweet text: {claim}

tweet image caption: {image_caption}

AI knowledge: {ai_knowledge}

WiKi knowledge: {wiki_knowledge}

web knowledge: {web_knowledge}

{format_instruction}

output:"""

prompt = PromptTemplate(
    template=template,
    input_variables=[
        'claim', 'image_caption',
        'ai_knowledge', 'wiki_knowledge', 'web_knowledge', 'real_prob', 'fake_prob',
    ],
    partial_variables={
        'format_instruction': parser.get_format_instructions(),
    },
)


def get_fact_checker_chain():
    chain = prompt | ChatOpenAI(
        temperature=.7, model_name=config.model_name,
    ) | parser
    return chain


agent_template = """You are a professional fact checker. Given the following tweet text, \
                    please judge whether the tweet is true or false and give your reasons step by step.
                    tweet text: {tweet_text}
                    """

agent_prompt = PromptTemplate(
    input_variables=['tweet_text'],
    template=agent_template,
)


def get_fact_checker_agent():
    tools = [
        ClosedBookTool(), WikipediaTool(), WebSearchTool(),
        FakeNewsDetectionTool(),
    ]
    llm = ChatOpenAI(temperature=.7, model_name=config.model_name)
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        return_intermediate_steps=True,
    )
    chain = agent_prompt | agent
    return chain
