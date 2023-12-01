from __future__ import annotations

from langchain.llms import OpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool
from pydantic import BaseModel
from pydantic import Field


class ClosedBookKnowledge(BaseModel):
    knowledges: list[str] = Field(description='List of the knowledge.')


parser = PydanticOutputParser(pydantic_object=ClosedBookKnowledge)

template = """Please now play the role of an encyclopaedic knowledge base, I will provide a social media tweet, I want to verify the authenticity of the tweet and you are responsible for providing knowledge that can support/refute the content of the tweet. The knowledge must be true and reliable, so if you don't have the relevant knowledge, please don't provide it.
{format_template}
---
text: {text_input}
image caption: {image_caption}
real probability: {real_prob}
fake probability: {fake_prob}
output: """

prompt = PromptTemplate(
    template=template,
    input_variables=['text_input', 'image_caption', 'real_prob', 'fake_prob'],
    partial_variables={'format_template': parser.get_format_instructions()},
)


def get_closed_knowledge_chain():
    chain = prompt | OpenAI(temperature=.7) | parser
    return chain


class ClosedBookTool(BaseTool):
    name = 'closed book knowledge tool'
    description = (
        'Use this tool when you need to search for knowledge within ChatGPT, '
        'note that the knowledge you get is relatively unreliable but will be more on topic.'
    )

    def _run(self, query: str) -> list[str]:
        return get_closed_knowledge_chain().invoke({'text_input': query}).knowledges

    def _arun(self, query: str) -> list[str]:
        raise NotImplementedError('This tool does not support async')


if __name__ == '__main__':
    from pyrootutils import setup_root

    root = setup_root('.', dotenv=True)

    print(
        prompt.invoke({
            'text_input': 'NASA has just discovered a new planet in the Andromeda galaxy',
            'image_caption': 'Hello world!',
            'fake_prob': 1,
            'real_prob': 0,
        }).text,
    )
    chain = get_closed_knowledge_chain()
    print(
        chain.invoke({
            'text_input': 'NASA has just discovered a new planet in the Andromeda galaxy',
            'image_caption': 'Hello world!',
            'fake_prob': 1,
            'real_prob': 0,
        }).knowledges,
    )
