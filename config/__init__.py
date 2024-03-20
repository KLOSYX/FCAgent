from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "config",
]

from typing import Literal


@dataclass(frozen=True)
class Config:
    log_level: Literal["DEBUG", "INFO", "WARNING"] = "DEBUG"
    agent_type: Literal[
        "openai_tools", "react_json", "structured_chat", "shoggoth13_react_json"
    ] = "shoggoth13_react_json"
    core_server_addr: str = "http://10.26.128.30:8000"
    vl_model_type: Literal["local", "gpt4v"] = "gpt4v"
    vl_server_addr: str = "http://10.26.128.30:8001"
    model_name: str = "gpt-3.5-turbo"
    search_engine: Literal["bing", "google", "duckduckgo"] = "bing"
    web_scrapy_max_splits: int = 3


config = Config()
