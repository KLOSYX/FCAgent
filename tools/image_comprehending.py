from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin

import aiohttp
import requests
from langchain.tools import BaseTool
from PIL import Image
from pydantic import BaseModel, Field
from pyrootutils import setup_root

from config import config
from utils.gpt4v import request_gpt4v

root = setup_root(".")

template = """"请简要描述一下图像的内容以及图像中的文本信息。"""


def get_vl_result(image: str) -> str:
    # 构造请求参数
    params = {"image": image, "text": template}
    # 发送POST请求
    response = requests.post(
        urljoin(config.vl_server_addr, "/vl"),
        data=params,
    )
    # 获取响应结果
    result = response.text
    return result


async def stream_get_vl_result(image: str):
    params = {"image": image, "text": template}
    url = urljoin(config.vl_server_addr, "/vl_stream")
    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=params) as response:
            # 使用iter_any方法逐块读取响应
            async for chunk in response.content.iter_any():
                yield chunk.decode("utf-8")


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

    def _run(self, image_name: str) -> str:
        """use tweet summary as input.

        could be in English and Chinese.
        """
        if config.vl_model_type == "gpt4v":
            resp = request_gpt4v(root / ".temp" / image_name, template) + "\n"
        else:
            image_content = load_tweet_content(image_name)
            resp = get_vl_result(image_content) + "\n"
        return resp

    async def _arun(self, image_name: str) -> str:
        image_content = load_tweet_content(image_name)
        res = ""
        if config.vl_model_type == "gpt4v":
            res = request_gpt4v(root / ".temp" / image_name, template) + "\n"
        else:
            async for token in stream_get_vl_result(image_content):
                res = token
                print("\r" + token, flush=True, end="")
        return res + "\n"
