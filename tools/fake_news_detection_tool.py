from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Any
from typing import Dict
from urllib.parse import urljoin

import requests
from langchain.tools import BaseTool
from PIL import Image
from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator
from pydantic import validator

from config import Config

config = Config()


def get_core_result(text: str, image: Image | str) -> str:
    if isinstance(image, str):
        image = Image.open(Path(image))
    buffer = BytesIO()
    image.save(buffer, format='JPEG')
    image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    # 构造请求参数
    params = {'image': image_data, 'text': text}
    # 发送POST请求
    response = requests.post(
        urljoin(config.core_server_addr, '/core'), data=params,
    )
    # 获取响应结果
    result = response.json()
    return f"fake probability: {result['fake_prob']:.0%}\treal probability: {result['real_prob']:.0%}"


class TweetContent(BaseModel):
    tweet_content: str = Field(
        description='The string must be in json format, containing the fields "tweet_text" and "tweet_image_path"',
    )

    @model_validator(mode='before')
    def validate_query(self) -> TweetContent:
        try:
            content = json.loads(self.tweet_content)
            assert 'tweet_text' in content and 'tweet_image_path' in content, \
                'missing field "tweet_text" or "tweet_image_path" in tweet_content'
        except Exception:
            raise ValueError('tweet_content is not a valid json string')
        return self


class FakeNewsDetectionTool(BaseTool):
    name = 'fake news detection tool'
    description = ("use this tool to get machine learning model prediction whether a tweet is true/false. \
                   The input string must be in json format, containing the fields 'tweet_text' and 'tweet_image_path'")
    args_schema = TweetContent

    def _run(self, tweet_content: str) -> str:
        tweet_content = tweet_content.replace("'", "\"")
        tweet_content = json.loads(tweet_content)
        return get_core_result(text=tweet_content['tweet_text'], image=tweet_content['tweet_image_path'])

    def _arun(self, query: str) -> list[str]:
        raise NotImplementedError('This tool does not support async')
