from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urljoin

import requests
from langchain.tools import BaseTool

from config import Config

config = Config()


def get_core_result(text: str, image: str) -> str:
    # 构造请求参数
    params = {'image': image, 'text': text}
    # 发送POST请求
    response = requests.post(
        urljoin(config.core_server_addr, '/core'), data=params,
    )
    # 获取响应结果
    result = response.json()
    return f"fake probability: {result['fake_prob']:.0%}\treal probability: {result['real_prob']:.0%}\n"


def load_image_content(image_path: str) -> dict:
    with open(Path(image_path)) as f:
        tweet_content = json.loads(f.read())
    return tweet_content


class FakeNewsDetectionTool(BaseTool):
    name = 'fnd_tool'
    description = (
        'use this tool to get machine learning model prediction whether a tweet is true/false. '
        'CANNOT be used as the only indicator. '
        'use the parameter `text` and `image_path` as input.'
    )

    def _run(self, text: str, image_path: str) -> str:
        """use tweet summary as input. could be in English and Chinese."""
        tweet_content = load_image_content(image_path)
        return get_core_result(text=text, image=tweet_content['tweet_image'])

    def _arun(self, text: str) -> list[str]:
        raise NotImplementedError('This tool does not support async')
