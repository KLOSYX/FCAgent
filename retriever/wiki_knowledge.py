from __future__ import annotations

from urllib.parse import urljoin

import requests
from langchain.tools import BaseTool

from config import config


def get_wiki_result(key_words: str) -> list[str]:
    params = {"key_words": key_words, "top_k": 3}
    response = requests.post(
        urljoin(config.core_server_addr, "/wiki"),
        data=params,
    )
    result = response.json()
    return result


class WikipediaTool(BaseTool):
    name = "ask_wikipedia"
    cn_name = "维基百科"
    description = "use this tool when you need to retrieve knowledge from Wikipedia. \
    note that knowledge may be out of date, but it is certainly correct. \
    the query MUST be in English."

    def _run(self, query: str) -> str:
        """use string 'query' as input.

        must be English.
        """
        return (
            "\n".join(f"{i}. {s}" for i, s in enumerate(get_wiki_result(key_words=query))) + "\n"
        )

    async def _arun(self, query: str) -> str:
        """use string 'query' as input.

        must be English.
        """
        return (
            "\n".join(f"{i}. {s}" for i, s in enumerate(get_wiki_result(key_words=query))) + "\n"
        )
