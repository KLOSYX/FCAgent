from __future__ import annotations

import base64
import json
from io import BytesIO
from typing import Any

import gradio as gr
from PIL import Image
from pyrootutils import setup_root

from config import Config
from fact_checker import get_fact_checker_agent

root = setup_root('.', pythonpath=True, dotenv=True)
config = Config()


def list_to_markdown(lst):
    markdown = ''
    for item in lst:
        markdown += f'- {item}\n'
    return markdown


def inference(raw_image: Any, claim: str):
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
    agent = get_fact_checker_agent()
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
