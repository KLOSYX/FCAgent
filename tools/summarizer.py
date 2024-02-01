from __future__ import annotations

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

from config import config

template = """推文: {claim_text}

信息: {history}

你是一名事实核查机构的编辑，你需要根据上面的信息撰写一篇简短的事实核查新闻，并需要给出你对于社交媒体信息真实性的判断。\
你的输出必须包括两个部分，格式如下：
- 评价：（真实/虚假/无法确定）
- 点评：（你的事实核查新闻，详尽分点叙述，保持专业理性的风格。你需要隐藏工具的名称和具体使用细节，你引用的证据来源只能是Wikipedia或者网站链接。）
- 参考资料：（列出你的所有参考资料来源。填“没有参考资料”，如果你找不到合适的参考资料。）
现在请按照上面的格式输出，请用中文和markdown格式输出：
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["claim_text", "history"],
)


def get_summarizer_chain():
    chain = prompt | ChatOpenAI(model_name=config.model_name, temperature=0.7, streaming=True)
    return chain
