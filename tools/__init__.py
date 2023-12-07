from __future__ import annotations

from langchain.agents import load_tools

__all__ = [
    'FakeNewsDetectionTool',
    'ImageComprehendingTool',
    'TOOL_LIST',
]

from .fake_news_detection_tool import FakeNewsDetectionTool
from .multi_modal_content_comprehending import ImageComprehendingTool

TOOL_LIST = [
    FakeNewsDetectionTool(),
    ImageComprehendingTool(),
]
