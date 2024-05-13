## 概览

本项目旨在提供一个基于大语言模型（LLM）的代理，通过分析图像和文本内容来验证多模态社交媒体帖子。它利用一套 Python 工具和模型来评估推文的真实性，并理解与推文相关的图片内容。该系统的构建注重模块化，便于扩展或修改其功能。

## 结构

该项目分为几个模块，每个模块负责核查过程的不同功能：

- `app.py`: 主应用程序文件，用于设置 Gradio 界面，供用户输入图像和文本进行验证。
- `config`: 配置文件夹，主要配置在`__init__.py`文件中。
- `fact_checker`: 包含Agent构建的主要逻辑及其实现。主要逻辑在`__init__.py`中，`get_agent.py`包含了一些辅助函数。
- `retriever`: 一套从各种来源检索信息的工具，包括大语言模型、网络搜索和维基百科，以及一个统一所有搜索逻辑的聚合搜索工具。
- `tools`: 包括失序信息检测和多模态内容理解工具。

## 依赖

要求Python >= 3.11。所有依赖项都列在 "requirements.txt "中。要安装，请在终端运行以下命令：

```bash
pip install -r requirements.txt
```

## 配置

在 `config/__init__.py` 文件中设置适当的值来配置系统。下面提供了一个配置示例：

```python
# Example configuration
log_level: Literal["DEBUG", "INFO", "WARNING"] = "DEBUG"
agent_type: Literal[
    "openai_tools", "shoggoth13_react_json", "shoggoth13_react_json_cn"
] = "shoggoth13_react_json_cn"
core_server_addr: str = "http://localhost:8001"  # Wikipedia知识库以及事实核查模型的接口地址
vl_model_type: Literal["local", "gpt4v"] = "gpt4v"  # VLM类型，本地部署或利用gpt4v
vl_server_addr: str = "http://localhost:8002"  # 如果VLM类型为本地部署，则需要设置服务器地址
model_name: str = "deepseek-chat"
search_engine: Literal["bing", "google", "duckduckgo"] = "bing"
rewrite_search_results: bool = False  # 是否使用llm重写搜索结果
web_scrapy_max_splits: int = 3  # 网页切分的块数
use_ocr: bool = True  # 是否使用OCR识别
use_constrained_decoding: bool = True  # 是否使用约束解码，仅支持vllm的接口
```

## 事实核查模块（fact_checker/__init__.py）

本模块定义了事实核查机器人的工作流程，包括一个基于大语言模型的 Agent，以及一系列用于分析和验证文本内容和图片的工具执行器。工作流程包括初始化 Agent、运行 Agent、执行工具以及决定是否继续或结束执行。此外，还定义了一个状态图来管理整个流程。

## Retrievers

retriever负责从不同来源获取信息，继承于Langchain的BaseTool类：

- `web_search.py`： 使用搜索引擎搜索网络的工具。
- `wikipedia.py`：从维基百科获取知识的工具。
- `ask_llm.py`：从大语言模型处获取知识的工具。
- `__init__.py`：定义检索器功能模块并注册所有工具。

## Tools

其中包含若干个主要工具：

- `fake_news_detection_tool.py`： 通过分析推文文本检测失序信息。
- `image_comprehending.py`：处理包含图片的推文内容，生成文字说明。
- `image_qa.py`: 处理图片测验工具的功能。
- `summarizer.py`：实现摘要文本生成器的功能。
- `web_browsing.py`: 从网页中获取内容的工具模块。
- `__init__.py`: 注册所有工具。

## utils/__init__.py

该模块定义了一个 `load_base_tools` 函数，用于加载指定目录中所有继承自 `BaseTool` 的类实例。用于自动注册所有的retriever以及tool，而无需在增加新的工具时手动添加。

## 用法

要运行系统，请执行以下命令：

```bash
python app.py
```

这将建立一个 Gradio 界面，用户可以输入推文的图片和文字，系统将返回验证结果。

## 贡献

欢迎为该项目做出贡献。请确保遵循现有的代码结构，并记录您所做的任何更改或添加。有关编码标准和贡献指南，请参阅 `CONTRIBUTING.md` 文件。

## License

This project is licensed under the [Apache-2.0 license](https://github.com/KLOSYX/fcsys/blob/main/LICENSE). Please see the LICENSE file for more details.

## 联系

如有任何疑问或支持，请通过我们的 [GitHub Issues](https://github.com/KLOSYX/fcsys/issues) 页面提交问题。
