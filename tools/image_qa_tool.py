from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urljoin

import requests
from langchain.tools import BaseTool
from pydantic import BaseModel
from pydantic import Field
from pyrootutils import setup_root

from config import Config

config = Config()
root = setup_root('.')


def get_vl_result(image: str, text: str) -> str:
    # 构造请求参数
    params = {'image': image, 'text': text}
    # 发送POST请求
    response = requests.post(
        urljoin(config.vl_server_addr, '/vl'), data=params,
    )
    # 获取响应结果
    result = response.text
    return result


def load_tweet_content(image_path: str) -> dict:
    with open(Path(image_path)) as f:
        tweet_content = json.loads(f.read())
    return tweet_content


class ImageQAScheme(BaseModel):
    question: str = Field(
        description='Should be the question of tweet image.',
    )
    image_path: str = Field(
        description='Should be the path of tweet image.',
        default=str(root / '.temp' / 'tweet_content.json'),
    )


class ImageQATool(BaseTool):
    name = 'image_qa_tool'
    description = (
        'Use this tool to ask any question about the tweet image content'
        'use parameter `question` as input'
    )
    args_schema: type[ImageQAScheme] = ImageQAScheme

    def _run(self, question: str, image_path: str) -> str:
        tweet_content = load_tweet_content(image_path)
        return get_vl_result(tweet_content['tweet_image'], question) + '\n'

    def _arun(self, question: str) -> str:
        raise NotImplementedError('This tool does not support async')
