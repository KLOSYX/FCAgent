from __future__ import annotations

import base64
import os
from io import BytesIO

from langchain.tools import BaseTool
from langchain_core.messages import HumanMessage
from langchain_openai.chat_models import ChatOpenAI
from PIL import Image
from pydantic import BaseModel, Field
from pyrootutils import setup_root

root = setup_root(".", pythonpath=True, dotenv=True)

from config import config
from utils import tool_exception_catch

vlm = ChatOpenAI(
    model_name=config.vl_model_name,
    openai_api_base=config.vl_server_addr,
    openai_api_key=os.getenv("VLM_API_KEY"),
    temperature=1.0,
    streaming=True,
)


def load_tweet_content(image_name: str) -> str:
    image_content = Image.open(root / ".temp" / image_name)
    buffer = BytesIO()
    image_content.save(buffer, format="PNG")  # 可以选择其他格式,如 JPEG
    img_bytes = buffer.getvalue()

    # 将字节流编码为 Base64 字符串
    image_content = base64.b64encode(img_bytes).decode("utf-8")
    return image_content


class ImageQaScheme(BaseModel):
    image_name: str = Field(description="The name of the image.")
    question: str = Field(
        description="Should be the question about the tweet image.",
    )


class ImageQaTool(BaseTool):
    name = "ask_image"
    cn_name = "图像问答"
    is_multimodal: bool = True
    description = "Use this tool to ask any question about the tweet image content"
    args_schema: type[ImageQaScheme] = ImageQaScheme

    @tool_exception_catch(name)
    def _run(self, question: str, image_name: str) -> str:
        img_base64 = load_tweet_content(image_name)
        msg = vlm.invoke(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                        },
                    ]
                )
            ]
        )
        return msg.content

    @tool_exception_catch(name)
    async def _arun(self, question: str, image_name: str) -> str:
        img_base64 = load_tweet_content(image_name)
        chunk_iterator = vlm.astream(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                        },
                    ]
                )
            ]
        )
        chunks = [chunk async for chunk in chunk_iterator]
        return "".join(chunk.content for chunk in chunks)


if __name__ == "__main__":
    tool = ImageQaTool()
    # print(tool.invoke("1c3ff2e0.png"))
    import asyncio

    print(
        asyncio.run(
            tool.ainvoke(
                {
                    "question": "What is the content of the image?",
                    "image_name": "1c3ff2e0.png",
                }
            )
        )
    )
