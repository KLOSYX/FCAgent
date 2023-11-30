from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import gradio as gr
from dotenv import load_dotenv

load_dotenv()
chat = ChatOpenAI(temperature=.7)


def inference(raw_image, claim):
    # TODO
    if not raw_image or not claim:
        return "图像和文本不能为空！"
    messages = [SystemMessage(content="你是一个事实核查机器人"), HumanMessage(content=claim)]
    history = ""
    # 模型决策 TODO
    stage1 = "模型决策中"
    history += stage1 + "\n"
    yield history

    # 知识获取 TODO
    stage2 = "知识获取中"
    history += stage2 + "\n"
    yield history

    # LLM事实核查 TODO
    for msg in chat.stream(messages):
        history += msg.content
        yield history


if __name__ == "__main__":
    inputs = [
        gr.Image(type='pil', interactive=True, label="Image"),
        gr.Textbox(lines=2, label="Claim", interactive=True),
    ]
    outputs = gr.Textbox(label="输出", lines=3)

    title = "fcsys"
    description = "fcsys"
    article = "fcsys"

    gr.Interface(inference, inputs, outputs, title=title, description=description, article=article).launch()
