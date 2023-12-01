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
