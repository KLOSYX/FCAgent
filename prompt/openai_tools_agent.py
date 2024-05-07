from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt_raw = {"system": "You are a helpful assistant", "human": "{input}\n\n请用“核查结束：（你的结论）”作为结尾。"}

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", prompt_raw["system"]),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", prompt_raw["human"]),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)
