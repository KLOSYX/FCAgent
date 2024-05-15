from __future__ import annotations

from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

from config import config
from utils import tool_exception_catch

template = """Assuming you are Wikipedia, given the context and question, you need to search for all relevant \
knowledge related to the question. Note: You only need to output reliable knowledge without any explanation \
or explanation.

---

Context: {context}

Question: {question}"""


prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"],
)
llm = ChatOpenAI(model_name=config.model_name, streaming=True, temperature=1.0)


def get_closed_knowledge_chain():
    knowledge_chain = prompt | llm
    return knowledge_chain


class AskLlmInput(BaseModel):
    context: str = Field(
        description="The context to make question more specific. Could be any language.",
        default="No context provided.",
    )
    question: str = Field(description="The question to ask. Could be any language.")


class AskLlmTool(BaseTool):
    name = "ask_llm"
    cn_name = "大模型"
    is_multimodal: bool = False
    description = (
        "use this tool when you need to ask the expert, " "note that responses are less current."
    )
    args_schema: type[BaseModel] = AskLlmInput

    @tool_exception_catch(name)
    def _run(self, question: str, context: str = "No context provided."):
        return (
            get_closed_knowledge_chain().invoke({"context": context, "question": question}).content
        )

    @tool_exception_catch(name)
    async def _arun(self, question: str, context: str = "No context provided."):
        chain = get_closed_knowledge_chain()
        chunks = [
            chunk
            async for chunk in chain.astream(
                {
                    "context": context,
                    "question": question,
                }
            )
        ]
        return "".join([chunk.content for chunk in chunks])
