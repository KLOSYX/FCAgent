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

template = """"请简要描述一下图像的内容以及图像中的文本信息。"""

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


class ImageScheme(BaseModel):
    image_name: str = Field(
        description="The name of the image.",
    )


class ImageComprehendingTool(BaseTool):
    name = "caption_image"
    cn_name = "图像描述"
    is_multimodal: bool = True
    description = "Use this tool to obtain text descriptions of tweet image content"
    args_schema: type[ImageScheme] = ImageScheme

    @tool_exception_catch(name)
    def _run(self, image_name: str) -> str:
        """use tweet summary as input.

        could be in English and Chinese.
        """
        img_base64 = load_tweet_content(image_name)
        msg = vlm.invoke(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": template},
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
    async def _arun(self, image_name: str) -> str:
        img_base64 = load_tweet_content(image_name)
        chunk_iterator = vlm.astream(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": template},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                        },
                    ]
                )
            ],
        )
        chunks = [chunk async for chunk in chunk_iterator]
        return "".join(chunk.content for chunk in chunks)


if __name__ == "__main__":
    tool = ImageComprehendingTool()
    # print(tool.invoke("1c3ff2e0.png"))
    import asyncio

    print(asyncio.run(tool.ainvoke("1c3ff2e0.png")))
