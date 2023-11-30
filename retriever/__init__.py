from __future__ import annotations
__all__ = [
    'get_closed_knowledge_chain',
    'get_qga_chain',
]

from .closed_book_knowledge import get_closed_knowledge_chain
from .query_generation_agent import get_qga_chain
