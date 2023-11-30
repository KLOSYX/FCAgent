from __future__ import annotations

import base64
from io import BytesIO
from urllib.parse import urljoin

import gradio as gr
import requests
from dotenv import load_dotenv
from PIL import Image

from config import Config
from fact_checker import get_fact_checker_chain
from retriever import get_closed_knowledge_chain
from retriever import get_qga_chain

load_dotenv()
config = Config()


def get_core_result(text: str, image: Image) -> dict:
    buffer = BytesIO()
    image.save(buffer, format='JPEG')
    image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    # 构造请求参数
    params = {'image': image_data, 'text': text}
    # 发送POST请求
    response = requests.post(
        urljoin(config.core_server_addr, '/core'), data=params,
    )
    # 获取响应结果
    result = response.json()
    return result


def get_wiki_result(key_words: str) -> dict:
    params = {'key_words': key_words}
    response = requests.post(
        urljoin(config.core_server_addr, '/wiki'), data=params,
    )
    result = response.json()
    return result


def list_to_markdown(lst):
    markdown = ''
    for item in lst:
        markdown += f'- {item}\n'
    return markdown


def inference(raw_image, claim):
    # TODO
    if not raw_image or not claim:
        return '图像和文本不能为空！'
    history = ''
    # 模型决策 TODO
    stage1 = '# 模型决策中'
    core_result = get_core_result(claim, raw_image)
    history += stage1 + '\n' + \
        f"- real probability: {core_result['real_prob']:.0%}\n- fake probability: {core_result['fake_prob']:.0%}" + '\n'
    yield history

    # 知识获取 TODO
    closed_knowledge_chain = get_closed_knowledge_chain()
    stage2 = '# 知识获取中'
    closed_knowledge: list[str] = closed_knowledge_chain.invoke(
        {
            'text_input': claim,
            'image_caption': 'Not provided',
            'fake_prob': '{:.0%}'.format(core_result['fake_prob']),
            'real_prob': '{:.0%}'.format(core_result['real_prob']),
        },
    ).knowledges
    history += stage2 + '\n' + list_to_markdown(closed_knowledge) + '\n'
    key_words = get_qga_chain().invoke({'text_input': claim, 'image_caption': 'Not provided'})[-1].replace(
        'query: ',
        '',
    )
    wiki_knowledge: list[dict] = get_wiki_result(key_words)
    history += f'> keywords: {key_words}\n' + \
        list_to_markdown(wiki_knowledge) + '\n'
    yield history

    # LLM事实核查 TODO
    stage3 = '# 结论'
    fact_checker = get_fact_checker_chain()
    result = fact_checker.invoke({
        'claim': claim,
        'image_caption': 'Not provided.',
        'ai_knowledge': closed_knowledge,
        'wiki_knowledge': wiki_knowledge,
        'fake_prob': '{:.0%}'.format(core_result['fake_prob']),
        'real_prob': '{:.0%}'.format(core_result['real_prob']),
    })
    history += stage3 + '\n' + \
        f'- The tweet is {result.conclusion}.\n- The reason is as followed: {result.reason}'
    yield history


if __name__ == '__main__':
    inputs = [
        gr.Image(type='pil', interactive=True, label='Image'),
        gr.Textbox(lines=2, label='Claim', interactive=True),
    ]
    outputs = gr.Markdown(label='输出')

    title = 'fcsys'
    description = 'fcsys'
    article = 'fcsys'

    gr.Interface(
        inference, inputs, outputs, title=title,
        description=description, article=article,
    ).launch()
