from __future__ import annotations

from langchain.llms import OpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel
from pydantic import Field

__all__ = ['get_fact_checker_chain']


class FactChecker(BaseModel):
    conclusion: str = Field(
        "The truthfulness of the tweet, could be 'real'/'fake'.")
    reason: str = Field(
        'Why the tweet is real/fake. Output in Markdown format.')


parser = PydanticOutputParser(pydantic_object=FactChecker)

template = """You are a fact-checking expert. Now I give you the content of a tweet (including the text as well as the caption of the image), and the probability that the tweet is true/false, and some knowledge that may be relevant to determining whether the tweet is true or false (note that this knowledge may be irrelevant or false, and that you can't fully trust it); you need to give a conclusion as to the truthfulness of the tweet, and a reason for it, based on this information.

real probability: {real_prob}

fake probability: {fake_prob}

tweet text: {claim}

tweet image caption: {image_caption}

knowledge: {knowledge}

{format_instruction}

output:"""

prompt = PromptTemplate(
    template=template,
    input_variables=['claim', 'image_caption',
                     'knowledge', 'real_prob', 'fake_prob'],
    partial_variables={
        'format_instruction': parser.get_format_instructions() + '请用中文回答。'},
)


def get_fact_checker_chain():
    chain = prompt | OpenAI(temperature=.7) | parser
    return chain
