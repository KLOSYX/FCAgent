from __future__ import annotations

from langchain.llms import OpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel
from pydantic import Field


class ClosedBookKnowledge(BaseModel):
    knowledges: list[str] = Field(description='List of the knowledge.')


parser = PydanticOutputParser(pydantic_object=ClosedBookKnowledge)

template = """Please now play the role of a knowledge base, I will provide a social media tweet including the text message of the tweet and the caption of the image, I want to verify the authenticity of the tweet and you are responsible for providing the knowledge that can support/refute the content of the tweet, this knowledge must be authentic and reliable, if you don't have any relevant knowledge of the subject then it shouldn't have any output.
{format_template}
---
text: {text_input}
image caption: {image_caption}
output: """

prompt = PromptTemplate(
    template=template,
    input_variables=['text_input', 'image_caption'],
    partial_variables={'format_template': parser.get_format_instructions()},
)


def get_closed_knowledge_chain():
    chain = prompt | OpenAI(temperature=0) | parser
    return chain


if __name__ == '__main__':
    from pyrootutils import setup_root

    root = setup_root('.', dotenv=True)

    print(
        prompt.invoke({
            'text_input': 'NASA has just discovered a new planet in the Andromeda galaxy',
            'image_caption': 'Hello world!',
        }).text,
    )
    chain = get_closed_knowledge_chain()
    print(
        chain.invoke({
            'text_input': 'NASA has just discovered a new planet in the Andromeda galaxy',
            'image_caption': 'Hello world!',
        }).knowledges,
    )
