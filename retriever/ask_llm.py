from __future__ import annotations

from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

from config import config
from utils.pydantic import PydanticOutputParser

template = """Please now play the role of an encyclopaedic knowledge base, I will provide a social media tweet, I want \
to verify the authenticity of the tweet and you are responsible for providing knowledge that can support/refute the \
content of the tweet. The knowledge must be true and reliable, so if you don't have the relevant knowledge, please \
don't provide it.
{format_instructions}
---
text: {text_input}
output: (list in Markdown format)"""


class Answer(BaseModel):
    knowledge_list: list[str] = Field(description="List of knowledge string.")


# Set up a parser
parser = PydanticOutputParser(pydantic_object=Answer)

prompt = PromptTemplate(
    template=template,
    input_variables=["text_input"],
).partial(format_instructions=parser.get_format_instructions())
llm = ChatOpenAI(model_name=config.model_name, temperature=0.0)


def get_closed_knowledge_chain():
    knowledge_chain = prompt | llm | parser
    return knowledge_chain


class AskLlmInput(BaseModel):
    query: str = Field(description="The query to search closed book. Should be any language.")


class AskLlmTool(BaseTool):
    name = "ask_llm"
    cn_name = "大模型"
    is_multimodal: bool = False
    description = (
        "use this tool when you need to search for knowledge within ChatGPT, "
        "note that the knowledge you get is relatively unreliable but will be more specific."
    )
    args_schema: type[BaseModel] = AskLlmInput

    def _run(self, query: str):
        return (
            "\t".join(
                f"{i}. {s}"
                for i, s in enumerate(
                    get_closed_knowledge_chain().invoke({"text_input": query}).knowledge_list
                )
            )
            + "\n"
        )

    async def _arun(self, query: str):
        return (
            "\t".join(
                f"{i}. {s}"
                for i, s in enumerate(
                    get_closed_knowledge_chain().invoke({"text_input": query}).knowledge_list
                )
            )
            + "\n"
        )


if __name__ == "__main__":
    from pyrootutils import setup_root

    root = setup_root(".", dotenv=True)

    print(
        prompt.invoke(
            {
                "text_input": "NASA has just discovered a new planet in the Andromeda galaxy",
            }
        ).text,
    )
    chain = get_closed_knowledge_chain()
    print(
        chain.invoke(
            {
                "text_input": "NASA has just discovered a new planet in the Andromeda galaxy",
            }
        ).knowledges,
    )
