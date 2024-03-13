from __future__ import annotations

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool
from langchain_openai.chat_models import ChatOpenAI
from pydantic import BaseModel, Field

template = """Please now play the role of an encyclopaedic knowledge base, I will provide a social media tweet, I want \
to verify the authenticity of the tweet and you are responsible for providing knowledge that can support/refute the \
content of the tweet. The knowledge must be true and reliable, so if you don't have the relevant knowledge, please \
don't provide it.
---
text: {text_input}
output: (list in Markdown format)"""

prompt = PromptTemplate(
    template=template,
    input_variables=["text_input"],
)


def get_closed_knowledge_chain():
    chain = prompt | ChatOpenAI(temperature=0.0, streaming=False)
    return chain


class ClosedBookInput(BaseModel):
    query: str = Field(description="The query to search closed book. Should be any language.")


class ClosedBookTool(BaseTool):
    name = "ask_llm"
    cn_name = "大模型"
    description = (
        "use this tool when you need to search for knowledge within ChatGPT, "
        "note that the knowledge you get is relatively unreliable but will be more specific."
    )
    args_schema: type[BaseModel] = ClosedBookInput

    def _run(self, query: str):
        return (
            "\n".join(
                f"{i}. {s}"
                for i, s in enumerate(get_closed_knowledge_chain().invoke({"text_input": query}))
            )
            + "\n"
        )

    async def _arun(self, query: str):
        return (
            "\n".join(
                f"{i}. {s}"
                for i, s in enumerate(get_closed_knowledge_chain().invoke({"text_input": query}))
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