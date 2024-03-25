from __future__ import annotations

import asyncio
from typing import Any

from langchain.chains import create_extraction_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import BaseTool
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_openai.chat_models import ChatOpenAI
from pydantic import BaseModel, Field

from config import config

schema = {
    "properties": {
        "article_title": {"type": "string"},
        "article_summary": {"type": "string"},
    },
    "required": ["article_title", "article_summary"],
}


# async def extract(content: str, schema: dict, llm: Any):
#     res = await create_extraction_chain(schema=schema, llm=llm).ainvoke(content)
#     return res


async def get_web_content_from_url(urls: list[str]):
    llm = ChatOpenAI(model_name=config.model_name, temperature=0.0)
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
        tasks.append(
            create_extraction_chain(
                schema=schema,
                llm=llm,
            ).ainvoke(split.page_content)
        )
    results = await asyncio.gather(*tasks)
    return [x["text"] for x in results] if results else []


class WebBrowsingInput(BaseModel):
    indices: list[int] = Field(description="indices of web pages to browse")
    urls: list[str] = Field(description="urls of web", required=False)


class WebBrowsingTool(BaseTool):
    name = "browse"
    cn_name = "网页浏览"
    is_multimodal: bool = False
    description = "Use this tool to obtain content of web pages"
    args_schema: type[BaseModel] = WebBrowsingInput

    def _run(self, indices: list[int], urls: list[str]) -> str:
        # web_content = get_web_content_from_url([urls[i] for i in indices])
        # return "\n".join(map(str, web_content)) + "\n"
        raise NotImplementedError

    async def _arun(self, indices: list[int], urls: list[str]) -> str:
        web_content = await get_web_content_from_url([urls[i] for i in indices])
        return "\n".join(map(str, web_content)) + "\n"


if __name__ == "__main__":
    import pprint

    from pyrootutils import setup_root

    setup_root(".", dotenv=True)
    pprint.pprint(
        get_web_content_from_url(
            ["https://www.piyao.org.cn/20231124/88da748b841b4be99374d2b651674d9e/c.html"],
        ),
    )
