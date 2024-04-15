from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt_raw = {"system": "You are a helpful assistant", "human": "{input}"}

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", prompt_raw["system"]),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", prompt_raw["human"]),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)
