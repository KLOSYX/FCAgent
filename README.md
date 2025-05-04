## Overview

This project is designed to provide a Large Language Model (LLM)-based agent for verifying multimodal social media posts by analyzing both image and text content. It leverages a suite of Python tools and models to assess the authenticity of tweets and comprehend the content within images associated with tweets. The system is built with a focus on modularity, allowing for easy expansion or modification of its capabilities.

## Structure

The project is structured into several modules, each responsible for a different aspect of the verification process:

- `app.py`: The main application file that sets up a Gradio interface for users to input images and text for verification.
- `config`: Configuration settings for the project, including server addresses and model names.
- `fact_checker`: Contains the `FactChecker` class and a method to get a fact-checking agent.
- `retriever`: A set of tools for retrieving information from various sources, including closed book knowledge, query generation, web search, and Wikipedia.
- `tools`: Includes tools for fake news detection and multi-modal content comprehension.

## Dependencies

All dependencies are listed in `requirements.txt`. To install, run the following command in your terminal:

```bash
pip install -r requirements.txt
```

## Configuration

Configure the system by setting the appropriate values in the `config/__init__.py` file. This includes the core server address, virtual learning server address, and the model name. An example configuration is provided below:

```python
# Example configuration
vl_server_addr = "http://localhost:8001"
model_name = "gpt-4"
```

## Fact-checking module (fact_checker/__init__.py)

This module defines the workflow of a fact-checking bot, including an Agent that uses OpenAI tools and a series of tool executors for analyzing and verifying textual content and pictures. The workflow includes initializing the Agent, running the Agent, executing the tools, and determining whether to continue or end the execution. Also, a statechart is defined to manage the whole process.

## Retrievers

The retrievers are responsible for fetching information from different sources:

- `closed_book_knowledge.py`: Searches for knowledge within the closed book knowledge base.
- `web_search.py`: Searches the web using the DuckDuckGo search engine.
- `wikipedia.py`: tool for retrieving knowledge from Wikipedia.
- `__init__.py`: defines the retriever function module and loads the base tools

## Tools

Two main tools are included:

- `fake_news_detection_tool.py`: Detects fake news by analyzing tweet text.
- `image_comprehending.py`: processes the content of tweets containing images to generate text descriptions.
- `image_qa.py`: handles the functionality of the image quiz tool.
- `summarizer.py`: implements the functionality of the summary text generator.
- `web_browsing.py`: tool module for fetching content from web pages.
- `__init__.py`: loads the list of base tools and defines the public interface.

### Tools module (utils/__init__.py)

This module defines a `load_base_tools` function that loads all instances of classes inherited from `BaseTool` in the specified directory.

## Usage

To run the system, execute the following command:

```bash
python app.py
```

This will set up a Gradio interface where users can input the image and text of a tweet, and the system will return the verification results.

## Contributing

Contributions to the project are welcome. Please ensure to follow the existing code structure and document any changes or additions you make. For coding standards and contribution guidelines, refer to the `CONTRIBUTING.md` file.

## License

This project is licensed under the [Apache-2.0 license](https://github.com/KLOSYX/fcsys/blob/main/LICENSE). Please see the LICENSE file for more details.

## Contact

For any queries or support, please submit an issue through our [GitHub Issues](https://github.com/KLOSYX/fcsys/issues) page.
