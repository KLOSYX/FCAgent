from __future__ import annotations

from urllib.parse import urljoin

import requests
from langchain.tools import BaseTool

from config import Config

config = Config()


def get_wiki_result(key_words: str) -> list[str]:
    params = {'key_words': key_words, 'top_k': 3}
    response = requests.post(
        urljoin(config.core_server_addr, '/wiki'), data=params,
    )
    result = response.json()
    return result


class WikipediaTool(BaseTool):
    name = 'Wikipedia tool'
    description = ('use this tool when you need to retrieve knowledge from Wikipedia. \
                    note that knowledge may be out of date, but it is certainly correct. \
                   the query MUST be in English.')

    def _run(self, query: str) -> list[str]:
        """use string 'query' as input. must be English."""
        return get_wiki_result(key_words=query)

    def _arun(self, query: str) -> list[str]:
        raise NotImplementedError('This tool does not support async')
