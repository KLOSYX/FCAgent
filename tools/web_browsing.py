from __future__ import annotations

import asyncio

from langchain.prompts import PromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import BaseTool
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_openai.chat_models import ChatOpenAI

from config import config
from utils.pydantic import PydanticOutputParser

SUMMARY_PROMPT = """Given the content of the web page, you need to extract title and summary \
from the page content:

page content: {page_content}

{format_instructions}
"""


class SummaryScheme(BaseModel):
    title: str = Field(description="Title of the web page.")
    summary: str = Field(description="Summary of the web page.")


parser = PydanticOutputParser(pydantic_object=SummaryScheme)
prompt = PromptTemplate(template=SUMMARY_PROMPT, input_variables=["page_content"]).partial(
    format_instructions=parser.get_format_instructions()
)
llm = ChatOpenAI(model_name=config.model_name, temperature=0.0)

summary_chain = prompt | llm | parser


async def get_web_content_from_url(urls: list[str]):
    loader = AsyncHtmlLoader(urls)
    docs = loader.load()
    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(docs)

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=4096,
        chunk_overlap=0,
    )
    splits = splitter.split_documents(docs_transformed)
    tasks = []
    for split in splits[: config.web_scrapy_max_splits]:
        tasks.append(summary_chain.ainvoke({"page_content": split.page_content}))
    results = await asyncio.gather(*tasks)
    return [{"title": x.title, "summary": x.summary} for x in results] if results else []


class WebBrowsingInput(BaseModel):
    urls: list[str] = Field(description="url of the web page.")


class WebBrowsingTool(BaseTool):
    name = "browse"
    cn_name = "网页浏览"
    is_multimodal: bool = False
    description = "Use this tool to browse urls for more details."
    args_schema: type[BaseModel] = WebBrowsingInput

    def _run(self, urls: list[str]) -> str:
        raise NotImplementedError

    async def _arun(self, urls: list[str]) -> str:
        web_content = await get_web_content_from_url(urls)
        return "\n".join(map(str, web_content)) + "\n"


if __name__ == "__main__":
    import pprint

    from pyrootutils import setup_root

    setup_root(".", dotenv=True)
    res = asyncio.run(
        get_web_content_from_url(
            ["https://www.piyao.org.cn/20231124/88da748b841b4be99374d2b651674d9e/c.html"],
        )
    )
    pprint.pprint(res)
