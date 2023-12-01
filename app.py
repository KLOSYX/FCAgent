from __future__ import annotations

import base64
from io import BytesIO
from shutil import rmtree
from urllib.parse import urljoin

import gradio as gr
import requests
from PIL import Image
from pyrootutils import setup_root

from config import Config
from fact_checker import get_fact_checker_agent
from fact_checker import get_fact_checker_chain
from retriever import get_closed_knowledge_chain
from retriever import get_qga_chain
from retriever import get_web_searcher
from retriever import get_wiki_result
from retriever import WebSearchTool

root = setup_root('.', pythonpath=True, dotenv=True)
config = Config()


def list_to_markdown(lst):
    markdown = ''
    for item in lst:
        markdown += f'- {item}\n'
    return markdown


# def inference(raw_image, claim):
#     # TODO
#     if not raw_image or not claim:
#         return '图像和文本不能为空！'
#     history = ''
#     # 模型决策 TODO
#     stage1 = '# 模型决策中'
#     core_result = get_core_result(claim, raw_image)
#     history += stage1 + '\n' + \
#         f"- real probability: {core_result['real_prob']:.0%}\n- fake probability: {core_result['fake_prob']:.0%}" + '\n'
#     yield history
#
#     # 知识获取 TODO
#     closed_knowledge_chain = get_closed_knowledge_chain()
#     stage2 = '# 知识获取中'
#     closed_knowledge: list[str] = closed_knowledge_chain.invoke(
#         {
#             'text_input': claim,
#             'image_caption': 'Not provided',
#             'fake_prob': '{:.0%}'.format(core_result['fake_prob']),
#             'real_prob': '{:.0%}'.format(core_result['real_prob']),
#         },
#     ).knowledges
#     # closed knowledge
#     history += stage2 + '\n' + list_to_markdown(closed_knowledge) + '\n'
#     key_words = get_qga_chain().invoke({'text_input': claim, 'image_caption': 'Not provided'})[-1].replace(
#         'query: ',
#         '',
#     )
#     # wiki knowledge
#     wiki_knowledge: list[str] = get_wiki_result(key_words)
#     history += f'> keywords: {key_words}\n' + \
#                list_to_markdown(wiki_knowledge) + '\n'
#     # web knowledge
#     web_knowledge: str = get_web_searcher().run(key_words)
#     history += '- ' + web_knowledge + '\n'
#     yield history
#
#     # LLM事实核查 TODO
#     stage3 = '# 结论'
#     fact_checker = get_fact_checker_chain()
#     result = fact_checker.invoke({
#         'claim': claim,
#         'image_caption': 'Not provided.',
#         'ai_knowledge': closed_knowledge,
#         'wiki_knowledge': wiki_knowledge,
#         'web_knowledge': web_knowledge,
#         'fake_prob': '{:.0%}'.format(core_result['fake_prob']),
#         'real_prob': '{:.0%}'.format(core_result['real_prob']),
#     })
#     history += stage3 + '\n' + \
#         f'- The tweet is {result.conclusion}.\n- The reason is as followed: {result.reason}'
#     yield history

def inference(raw_image: None | Image, claim: None | str):
    if not raw_image or not claim:
        return '图像和文本不能为空！'
    tmp_dir = root / '.temp'
    if not tmp_dir.exists():
        tmp_dir.mkdir(parents=True)
    raw_image.save(tmp_dir / 'image.png')
    agent = get_fact_checker_agent()
    response = agent.stream(
        {'tweet_text': claim, 'tweet_image_path': str(tmp_dir / 'image.png')},
    )
    partial_message = ''
    for chunk in response:
        partial_message = partial_message + \
            '\n'.join([str(msg.content) for msg in chunk['messages']]) + '\n'
        message_list = [
            msg for msg in partial_message.replace(
                '\n\n', '\n',
            ).split('\n') if msg.strip()
        ]
        yield list_to_markdown(message_list)


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
    ).queue().launch()
