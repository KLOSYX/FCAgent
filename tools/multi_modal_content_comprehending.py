from __future__ import annotations

import json
from urllib.parse import urljoin

import requests
from langchain.tools import BaseTool
from pyrootutils import setup_root

from config import Config

config = Config()
root = setup_root('.')

template = """"Please provide a brief description of the image content and textual information."""


def get_vl_result(image: str) -> str:
    # 构造请求参数
    params = {'image': image, 'text': template}
    # 发送POST请求
    response = requests.post(
        urljoin(config.vl_server_addr, '/vl'), data=params,
    )
    # 获取响应结果
    result = response.text
    return result


def load_tweet_content() -> dict:
    with open(root / '.temp' / 'tweet_content.json') as f:
        tweet_content = json.loads(f.read())
    return tweet_content


class ImageComprehendingTool(BaseTool):
    name = 'image_comprehending_tool'
    description = (
        'Use this tool to obtain text descriptions of tweet image content'
    )

    def _run(self, tweet_text_summary: str) -> str:
        """use tweet summary as input. could be in English and Chinese."""
        tweet_content = load_tweet_content()
        return get_vl_result(tweet_content['tweet_image'])

    def _arun(self, tweet_summary: str) -> list[str]:
        raise NotImplementedError('This tool does not support async')
