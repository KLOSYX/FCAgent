from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

__all__ = [
    'Config',
]


@dataclass(frozen=True)
class Config:
    core_server_addr: str = 'http://10.26.128.30:8000'
    model_name: str = 'gpt-3.5-turbo'
