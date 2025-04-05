from __future__ import annotations

from utils import load_base_tools

__all__ = [
    "TOOL_LIST",
]

TOOL_LIST = load_base_tools("tools", except_classes=["FakeNewsDetectionTool"])
