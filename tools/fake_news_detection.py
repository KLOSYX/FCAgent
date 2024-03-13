from __future__ import annotations

import base64
from io import BytesIO
from urllib.parse import urljoin

import requests
from langchain.tools import BaseTool
from PIL import Image
from pydantic import BaseModel, Field
from pyrootutils import setup_root

from config import config

root = setup_root(".")


def get_core_result(text: str, image: str) -> str:
    # 构造请求参数
    params = {"image": image, "text": text}
    # 发送POST请求
    response = requests.post(
        urljoin(config.core_server_addr, "/core"),
        data=params,
    )
    # 获取响应结果
    result = response.json()
    return f"fake probability: {result['fake_prob']:.0%}\treal probability: {result['real_prob']:.0%}\n"


def load_image_content(image_name: str) -> str:
    image_content = Image.open(root / ".temp" / image_name)
    buffer = BytesIO()
    image_content.save(buffer, format="PNG")  # 可以选择其他格式,如 JPEG
    img_bytes = buffer.getvalue()

    # 将字节流编码为 Base64 字符串
    image_content = base64.b64encode(img_bytes).decode("utf-8")
    return image_content


class FNDScheme(BaseModel):
    text: str = Field(description="Should be tweet text.")
    image_name: str = Field(description="Should be the tweet image name.")


class FakeNewsDetectionTool(BaseTool):
    name = "predict"
    cn_name = "预测模型"
    is_multimodal: bool = True
    description = (
        "use this tool to get machine learning model prediction whether a tweet is true/false. "
        "CANNOT be used as the only indicator. "
    )
    args_schema: type[FNDScheme] = FNDScheme

    def _run(self, text: str, image_name: str) -> str:
        """use tweet summary as input.

        could be in English and Chinese.
        """
        image_content = load_image_content(image_name)
        return get_core_result(text=text, image=image_content)

    async def _arun(self, text: str, image_name: str) -> str:
        """use tweet summary as input.

        could be in English and Chinese.
        """
        image_content = load_image_content(image_name)
        return get_core_result(text=text, image=image_content)
