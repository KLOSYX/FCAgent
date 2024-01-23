from __future__ import annotations

__all__ = [
    "get_closed_knowledge_chain",
    "get_qga_chain",
    "get_wiki_result",
    "ClosedBookTool",
    "WikipediaTool",
    "get_web_searcher",
    "WebSearchTool",
    "RETRIEVER_LIST",
]

from .closed_book_knowledge import ClosedBookTool, get_closed_knowledge_chain
from .query_generation_agent import get_qga_chain
from .web_search import WebSearchTool, get_web_searcher
from .wiki_knowledge import WikipediaTool, get_wiki_result

RETRIEVER_LIST = [
    ClosedBookTool(),
    WikipediaTool(),
    WebSearchTool(),
]
