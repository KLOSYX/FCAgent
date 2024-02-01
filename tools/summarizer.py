from __future__ import annotations

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

template = """推文: {claim_text}

信息: {history}

你是一名事实核查机构的编辑，你需要根据上面的信息撰写一篇简短的事实核查新闻，并需要给出你对于社交媒体信息真实性的判断。\
你的输出必须包括两个部分，格式如下：
- 评价：（真实/虚假/无法确定）
- 点评：（你的事实核查新闻，详尽分点叙述，保持专业理性的风格）
现在请按照上面的格式输出，请用中文和markdown格式输出：
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["claim_text", "history"],
)


def get_summarizer_chain():
    chain = prompt | ChatOpenAI(temperature=0.7, streaming=True)
    return chain
