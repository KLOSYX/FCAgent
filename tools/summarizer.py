from __future__ import annotations

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

template = """claim: {claim_text}

info: {history}

你是一名事实核查机构的编辑，你需要根据上面的信息总结一段简短的事实核查信息，并需要给出你对于社交媒体信息真实性的判断。\
你的输出必须包括两个部分，格式如下：
- 评分：（真实/虚假/无法确定）
- 点评：（你的打分理由）
现在请按照上面的格式输出，请用中文和markdown格式输出：
"""

prompt = PromptTemplate(
    template=template,
    input_variables=['claim_text', 'history'],
)


def get_summarizer_chain():
    chain = prompt | ChatOpenAI(temperature=.7)
    return chain
