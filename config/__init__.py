from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

__all__ = [
    'Config',
]


@dataclass(frozen=True)
class Config:
    core_server_addr: str = field(
        default='http://10.26.128.30:8000', metadata={'description': '核心决策服务器的地址'})
