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

root = setup_root(".")


def get_vl_result(image: str, text: str) -> str:
    # 构造请求参数
    params = {"image": image, "text": text}
    # 发送POST请求
    response = requests.post(
        urljoin(config.vl_server_addr, "/vl"),
        data=params,
    )
    # 获取响应结果
    result = response.text
    return result


async def stream_get_vl_result(image: str, text: str):
    params = {"image": image, "text": text}
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


class ImageQaScheme(BaseModel):
    image_name: str = Field(description="The name of the image.")
    question: str = Field(
        description="Should be the question about the tweet image.",
    )


class ImageQaTool(BaseTool):
    name = "ask_image"
    cn_name = "图像问答"
    description = "Use this tool to ask any question about the tweet image content"
    args_schema: type[ImageQaScheme] = ImageQaScheme

    def _run(self, question: str, image_name: str) -> str:
        image_content = load_tweet_content(image_name)
        return get_vl_result(image_content, question) + "\n"

    async def _arun(self, question: str, image_name: str) -> str:
        image_content = load_tweet_content(image_name)
        res = ""
        async for token in stream_get_vl_result(image_content, question):
            res = token
            print("\r" + token, flush=True, end="")
        return res + "\n"
