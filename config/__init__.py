from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "config",
]


@dataclass(frozen=True)
class Config:
    core_server_addr: str = "http://10.26.128.30:8000"
    vl_server_addr: str = "http://10.26.128.30:8001"
    model_name: str = "gpt-4-0125-preview"
    search_engine: str = "bing"
    web_scrapy_max_splits: int = 3


config = Config()
