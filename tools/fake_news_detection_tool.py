from __future__ import annotations

import json
from urllib.parse import urljoin

import requests
from langchain.tools import BaseTool
from pyrootutils import setup_root

from config import Config

config = Config()
root = setup_root('.')


def get_core_result(text: str, image: str) -> str:
    # 构造请求参数
    params = {'image': image, 'text': text}
    # 发送POST请求
    response = requests.post(
        urljoin(config.core_server_addr, '/core'), data=params,
    )
    # 获取响应结果
    result = response.json()
    return f"fake probability: {result['fake_prob']:.0%}\treal probability: {result['real_prob']:.0%}"


def load_tweet_content() -> dict:
    with open(root / '.temp' / 'tweet_content.json') as f:
        tweet_content = json.loads(f.read())
    return tweet_content


class FakeNewsDetectionTool(BaseTool):
    name = 'fake news detection tool'
    description = (
        'use this tool to get machine learning model prediction whether a tweet is true/false.'
        'use the tweet text summary as input.'
    )

    def _run(self, tweet_text_summary: str) -> str:
        """use tweet summary as input. could be in English and Chinese."""
        tweet_content = load_tweet_content()
        return get_core_result(text=tweet_content['tweet_text'], image=tweet_content['tweet_image'])

    def _arun(self, tweet_summary: str) -> list[str]:
        raise NotImplementedError('This tool does not support async')
