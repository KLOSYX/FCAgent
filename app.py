from __future__ import annotations

import base64
import json
from io import BytesIO
from typing import Any

import gradio as gr
from pyrootutils import setup_root

from config import Config
from fact_checker import get_fact_checker_agent
from retriever import RETRIEVER_LIST
from tools import get_summarizer_chain
from tools import TOOL_LIST

root = setup_root('.', pythonpath=True, dotenv=True)
config = Config()

tool_map = {x.name: x for x in TOOL_LIST}
retriever_map = {x.name: x for x in RETRIEVER_LIST}


def list_to_markdown(lst):
    markdown = ''
    for item in lst:
        markdown += f'- {item}\n'
    return markdown


def inference(raw_image: Any, claim: str, selected_tools: list[str], selected_retrievers: list[str]):
    if not raw_image or not claim:
        return '图像和文本不能为空！'
    tmp_dir = root / '.temp'
    if not tmp_dir.exists():
        tmp_dir.mkdir(parents=True)
    buffer = BytesIO()
    raw_image.save(buffer, format='JPEG')
    image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    with open(tmp_dir / 'tweet_content.json', 'w') as f:
        f.write(
            json.dumps(
                {
                    'tweet_text': claim,
                    'tweet_image': image_data,
                }, ensure_ascii=False,
            ),
        )
    all_tools = [tool_map[x] for x in selected_tools] + \
                [retriever_map[x] for x in selected_retrievers]
    agent = get_fact_checker_agent(all_tools)
    response = agent.stream(
        {
            'tweet_text': claim,
            'tweet_image_path': str(tmp_dir / 'tweet_content.json'),
        },
    )
    partial_message = ''
    for chunk in response:
        partial_message = partial_message + \
            '\n'.join([str(msg.content) for msg in chunk['messages']]) + '\n'
        yield partial_message
    summarizer = get_summarizer_chain()
    summarization = summarizer.invoke(
        {
            'claim_text': claim,
            'history': partial_message,
        },
    ).content
    yield partial_message + '\n---\n' + summarization


if __name__ == '__main__':
    inputs = [
        gr.Image(type='pil', interactive=True, label='Image'),
        gr.Textbox(lines=2, label='Claim', interactive=True),
        gr.Checkboxgroup(
            list(tool_map.keys()), value=list(
                tool_map.keys(),
            ), label='Tools',
        ),
        gr.Checkboxgroup(
            list(retriever_map.keys()), value=list(
                retriever_map.keys(),
            ), label='Retriever',
        ),
    ]
    outputs = gr.Markdown(label='Output', sanitize_html=False)

    title = 'FCAgent'
    description = 'This project is designed to provide a Large Language Model (LLM)-based agent for verifying \
    multimodal social media posts by analyzing both image and text content. It leverages a suite of Python tools \
    and models to assess the authenticity of tweets and comprehend the content within images associated with tweets. \
    The system is built with a focus on modularity, allowing for easy expansion or modification of its capabilities.'
    article = 'FCAgent'

    gr.Interface(
        inference, inputs, outputs, title=title,
        description=description, article=article,
    ).queue().launch()
