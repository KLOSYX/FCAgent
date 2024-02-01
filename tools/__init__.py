from __future__ import annotations

__all__ = [
    "FakeNewsDetectionTool",
    "ImageComprehendingTool",
    "TOOL_LIST",
    "get_summarizer_chain",
]

from .fake_news_detection_tool import FakeNewsDetectionTool
from .image_qa_tool import ImageQATool
from .multi_modal_content_comprehending import ImageComprehendingTool
from .summarizer import get_summarizer_chain
from .web_scrapy import WebBrowsingTool

TOOL_LIST = [
    FakeNewsDetectionTool(),
    ImageComprehendingTool(),
    ImageQATool(),
    WebBrowsingTool(),
]
