from __future__ import annotations

import asyncio
import json
from typing import Any

import gradio as gr
from loguru import logger
from pyrootutils import setup_root

from config import config
from fact_checker import get_fact_checker_agent
from retriever import RETRIEVER_LIST
from tools import TOOL_LIST
from tools.summarizer import SummarizerScheme
from utils import generate_filename_from_image

root = setup_root(".", pythonpath=True, dotenv=True)
logger.level(config.log_level)

tool_map = {x.cn_name: x for x in TOOL_LIST}
retriever_map = {x.cn_name: x for x in RETRIEVER_LIST}

# initialize ocr
if config.use_ocr:
    import os

    from paddleocr import PaddleOCR

    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    ocr = PaddleOCR(use_angle_cls=True, lang="ch")
else:
    ocr = None


def format_markdown(text: str) -> str:
    text = text.replace("```", "")
    return text


async def inference(
    raw_image: Any, claim: str, selected_tools: list[str], selected_retrievers: list[str]
):
    global ocr
    tmp_dir = root / ".temp"
    if not tmp_dir.exists():
        tmp_dir.mkdir(parents=True)
    if raw_image is not None:
        image_name = generate_filename_from_image(raw_image) + ".png"
        raw_image.save(f"{tmp_dir}/{image_name}")
    else:
        image_name = "No image"

    all_tools = [tool_map[x] for x in selected_tools] + [
        retriever_map[x] for x in selected_retrievers
    ]
    if raw_image is None:
        all_tools = list(filter(lambda x: not x.is_multimodal, all_tools))

    agent = get_fact_checker_agent(all_tools, ocr)
    partial_message = ""
    ended = False
    on_tool = False
    current_tool_streaming_out = ""
    async for event in agent.astream_events(
        {
            "input": {
                "tweet_text": claim,
                "tweet_image_name": image_name,
            }
        },
        version="v1",
    ):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                if not on_tool:
                    partial_message += content
                    yield format_markdown(partial_message)
                else:
                    current_tool_streaming_out += content
                    yield format_markdown(partial_message + "\n\n" + current_tool_streaming_out)
        elif kind == "on_tool_start":
            on_tool = True
            content = f"\n\n> 调用工具：{event['name']}\t输入: {event['data'].get('input')}\n\n"
            partial_message += content
            yield format_markdown(partial_message)
        elif kind == "on_tool_end":
            on_tool = False
            current_tool_streaming_out = ""
            tool_output = event["data"].get("output")
            if tool_output:
                tool_output = tool_output.replace("\n", "\t")
                partial_message += f"\n\n> 工具输出：{tool_output}\n\n"
            yield format_markdown(partial_message)
        elif kind == "on_parser_end" and isinstance(event["data"].get("output"), SummarizerScheme):
            ended = True
            result: SummarizerScheme = event["data"].get("output")
            partial_message += "\n\n---\n\n"
            partial_message += (
                f"- 结论：{result.rank.value}\n- 过程：{result.procedure}\n- 参考：{result.reference}\n"
            )
            yield format_markdown(partial_message)
        elif not ended and not current_tool_streaming_out:
            yield format_markdown(partial_message + "\n\nPlease wait...")


if __name__ == "__main__":
    inputs = [
        gr.Image(type="pil", interactive=True, label="图像"),
        gr.Textbox(lines=2, label="文本", interactive=True),
        gr.Checkboxgroup(
            list(tool_map.keys()),
            value=list(
                tool_map.keys(),
            ),
            label="工具选择",
        ),
        gr.Checkboxgroup(
            list(retriever_map.keys()),
            value=list(
                retriever_map.keys(),
            ),
            label="知识库选择",
        ),
    ]
    outputs = gr.Markdown(label="输出", sanitize_html=False)

    title = "多模态失序信息检测原型系统"
    description = "该系统提供一个基于大型语言模型(LLM)的代理，用于通过分析图像和文本内容来验证多模态社交媒体信息的真实性。\
    系统接受文本和图像作为输入，并可以灵活组合不同的工具/知识库。 系统将输出真实性判断以及事实核查信息。"
    article = "多模态失序信息检测原型系统"

    gr.Interface(
        inference,
        inputs,
        outputs,
        allow_flagging="never",
        title=title,
        description=description,
        article=article,
        submit_btn="提交",
        stop_btn="停止",
        clear_btn="清除",
    ).queue(max_size=10).launch(server_name="localhost", server_port=7860, ssl_verify=False)
