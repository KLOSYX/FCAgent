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
        "openai_tools", "shoggoth13_react_json", "shoggoth13_react_json_cn"
    ] = "shoggoth13_react_json_cn"
    core_server_addr: str = "http://localhost:8001"
    vl_server_addr: str = "http://localhost:11434/v1"
    model_name: str = "qwen2.5:latest"
    vl_model_name: str = "minicpm-v:latest"
    search_engine: Literal["bing", "google", "duckduckgo"] = "google"
    rewrite_search_results: bool = False
    web_scrapy_max_splits: int = 3
    use_ocr: bool = False
    use_constrained_decoding: bool = False


config = Config()
