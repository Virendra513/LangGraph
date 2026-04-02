from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import os

load_dotenv()

client = InferenceClient(api_key=os.environ["HF_TOKEN"])

def invoke_model(messages: list[dict], model: str = "openai/gpt-oss-20b:groq", temperature: float = 0.7, max_tokens: int = 50, top_p: float = 0.9):
    if not messages:
        raise ValueError("messages cannot be empty")
    return client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True
    )

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]

def message_to_dict(msg: BaseMessage) -> dict:
    role_map = {"human": "user", "ai": "assistant", "system": "system"}
    role = role_map.get(getattr(msg, "type", ""), "system")
    return {"role": role, "content": msg.content}

# ── Node: streams to console, returns full message to state ────
def chat_node(state: ChatState):
    messages_for_model = [
        m if isinstance(m, dict) else message_to_dict(m)
        for m in state["messages"]
    ]

    stream = invoke_model(messages_for_model)

    full_response = ""
    #print("\nAssistant: ", end="", flush=True)

    for chunk in stream:
        delta = chunk.choices[0].delta
        content = delta.content if delta and delta.content else ""
        if content:
            #print(content, end="", flush=True)
            full_response += content

    #print()  # newline after stream ends

    #  Return the complete message — LangGraph stores it in memory cleanly
    return {
        "messages": [{"role": "assistant", "content": full_response}]
    }

# ── Graph ──────────────────────────────────────────────────────
checkpointer = InMemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

cb = graph.compile(checkpointer=checkpointer)

CONFIG = {"configurable": {"thread_id": "demo-thread"}}

'''
# ── Multi-turn chat loop ───────────────────────────────────────
print("Chat started. Type 'quit' to exit.\n")

while True:
    user_input = input("You: ").strip()
    if not user_input or user_input.lower() == "quit":
        break

    cb.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config=CONFIG,
    )
'''