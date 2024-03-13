from __future__ import annotations

import hashlib
from io import BytesIO
from typing import Any

import gradio as gr
from pyrootutils import setup_root

from fact_checker import get_fact_checker_agent
from retriever import RETRIEVER_LIST
from tools import TOOL_LIST
from tools.summarizer import get_summarizer_chain

root = setup_root(".", pythonpath=True, dotenv=True)

tool_map = {x.cn_name: x for x in TOOL_LIST}
retriever_map = {x.cn_name: x for x in RETRIEVER_LIST}


def list_to_markdown(lst):
    markdown = ""
    for item in lst:
        markdown += f"- {item}\n"
    return markdown


def generate_filename_from_image(image):
    # 创建一个BytesIO对象，用于保存图像的二进制数据
    img_byte_arr = BytesIO()
    # 将图像保存到BytesIO对象中（这里以PNG格式为例）
    image.save(img_byte_arr, format="PNG")
    # 获取图像的二进制数据
    img_byte_arr = img_byte_arr.getvalue()
    # 使用sha256哈希算法
    hasher = hashlib.md5()
    # 更新哈希值
    hasher.update(img_byte_arr)
    # 获取十六进制格式的哈希值
    hash_value = hasher.hexdigest()[:8]
    # 根据需要添加文件扩展名（这里以.png为例）
    return str(hash_value)


async def inference(
    raw_image: Any, claim: str, selected_tools: list[str], selected_retrievers: list[str]
):
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
        all_tools = list(filter(lambda x: "image" not in x.name.lower(), all_tools))
    all_tool_names = [t.name for t in all_tools]
    agent = get_fact_checker_agent(all_tools)
    partial_message = ""
    async for event in agent.astream_events(
        {
            "input": {
                "tweet_text": claim,
                "tweet_image_name": image_name if raw_image is not None else "No image.",
            }
        },
        version="v1",
    ):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                partial_message += content
                if partial_message.endswith("```"):
                    partial_message += "\n"
                yield partial_message
        elif kind == "on_tool_start":
            content = f"\n\n> 调用工具：{event['name']}\t输入: {event['data'].get('input')}\n\n"
            partial_message += content
            yield partial_message
        elif kind == "on_tool_end":
            partial_message += f"\n\n> 工具输出：{event['data'].get('output')}\n\n"
            yield partial_message
    partial_message += "\n\n---\n\n"
    summarizer = get_summarizer_chain()
    async for chunk in summarizer.astream(
        {
            "claim_text": claim,
            "history": partial_message,
        },
    ):
        partial_message += chunk.content
        yield partial_message


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
    ).queue().launch(server_name="0.0.0.0", server_port=7860, ssl_verify=False)
