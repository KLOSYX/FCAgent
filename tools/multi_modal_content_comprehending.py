from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urljoin

import aiohttp
import requests
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from pyrootutils import setup_root

from config import config

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


def load_tweet_content(image_path: str) -> dict:
    with open(Path(image_path)) as f:
        tweet_content = json.loads(f.read())
    return tweet_content


class ImageScheme(BaseModel):
    image_path: str = Field(
        description="Should be the path of tweet image.",
    )


class ImageComprehendingTool(BaseTool):
    name = "caption_image"
    description = "Use this tool to obtain text descriptions of tweet image content"
    args_schema: type[ImageScheme] = ImageScheme

    def _run(self, image_path: str = str(root / ".temp" / "tweet_content.json")) -> str:
        """use tweet summary as input.

        could be in English and Chinese.
        """
        tweet_content = load_tweet_content(image_path)
        return get_vl_result(tweet_content["tweet_image"]) + "\n"

    async def _arun(self, image_path: str = str(root / ".temp" / "tweet_content.json")) -> str:
        tweet_content = load_tweet_content(image_path)
        res = ""
        async for token in stream_get_vl_result(tweet_content["tweet_image"]):
            res = token
            print("\r" + token, flush=True, end="")
        return res + "\n"
