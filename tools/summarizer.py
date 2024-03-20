from __future__ import annotations

from enum import Enum
from typing import Literal

from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

from config import config
from utils.pydantic import PydanticOutputParser

template = """核查过程: {history}
---
你是一名人类事实核查机构的编辑，你需要根据上面的事实核查信息，判断内容“{claim_text}”的真实性。\
{format_instructions}
"""


class SummarizerScheme(BaseModel):
    rank: Literal["真实", "虚假", "有待核实", "真假参半"] = Field(description="核查过程的结论")
    procedure: str = Field(description="中文，核查过程的过程")
    reference: list[tuple[str, str]] = Field(
        description="权威可靠来源的参考资料列表，以“(title, url)”的格式输出；如没有参考资料，返回空列表"
    )


parser = PydanticOutputParser(pydantic_object=SummarizerScheme)


prompt = PromptTemplate(
    template=template,
    input_variables=["claim_text", "history"],
).partial(format_instructions=parser.get_format_instructions())


def get_summarizer_chain():
    chain = prompt | ChatOpenAI(model_name=config.model_name, streaming=False) | parser
    return chain
