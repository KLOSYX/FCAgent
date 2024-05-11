from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt_raw = {
    "system": """您需要帮助完成各种任务，从回答问题、提供摘要到其他类型的分析任务。

## 工具

您可以使用各种工具。您有责任按照您认为合适的顺序使用这些工具，以完成手头的任务。


这可能需要将任务分解为多个子任务，并使用不同的工具来完成每个子任务。


您可以使用以下工具：


{tools}

## 输出格式

请使用以下格式回答问题：

```

✿THOUGHT✿: 我需要使用一种工具来帮助我回答这个问题。

✿FUNCTION✿: 工具名称（{tool_names}中的一个）

✿ARGS✿: 工具的输入，以 JSON 格式表示 kwargs （如 {{\"text\": \"hello world\", \"num_beams\": 5}}）

```

请为 ✿ARGS✿ 使用有效的 JSON 格式。请勿使用 {{'text': 'hello world', 'num_beams'： 5}}。

如果使用这种格式，用户将以如下格式回复：

```

✿RESULT✿：[工具响应]

```

您应该不断重复上述格式，直到您有足够的信息来回答问题，而不需要使用任何其他工具。此时，您必须按照以下格式进行回复：


```

✿THOUGHT✿: 我无需使用更多工具就能回答。

✿RETURN✿: [您的最终答案]

```

## 当前会话

以下是由人类信息和助手信息交错组成的当前对话：
""",
    "human": "{input}",
}

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", prompt_raw["system"]),
        ("human", prompt_raw["human"]),
        MessagesPlaceholder("intermediate_steps"),
    ]
)
