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
    core_server_addr: str = "http://localhost:8001"
    vl_model_type: Literal["local", "gpt4v"] = "gpt4v"
    vl_server_addr: str = "http://localhost:8002"
    model_name: str = "vllm"
    search_engine: Literal["bing", "google", "duckduckgo"] = "bing"
    web_scrapy_max_splits: int = 3
    use_ocr: bool = True
    use_constrained_decoding: bool = True


config = Config()
