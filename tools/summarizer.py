from __future__ import annotations

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

template = """claim: {claim_text}

info: {history}

You are an editor of a fact-check agency, and you need to release a fact-checking article \
based on the information above, and provide your judgment on the authenticity of social media tweet.
Your output must include two parts in the following format:
- Rating: (True/False/Uncertain)
- Comment: (Your fact-checking article. Should be professional and rational.)
Now please give your rating and comment:
"""

prompt = PromptTemplate(
    template=template,
    input_variables=['claim_text', 'history'],
)


def get_summarizer_chain():
    chain = prompt | ChatOpenAI(temperature=.7)
    return chain
