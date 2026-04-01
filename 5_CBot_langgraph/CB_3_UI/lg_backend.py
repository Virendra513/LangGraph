from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import os 
load_dotenv()

client = InferenceClient(
    api_key=os.environ['HF_TOKEN'],
)


def invoke_model(messages, model="openai/gpt-oss-20b:groq"):
    if not messages:
        raise ValueError("messages cannot be empty")

    completion = client.chat.completions.create(
        model=model,
        messages=messages,  #  expects list[dict]
    )

    return completion.choices[0].message.content

class ChatState(TypedDict):
    # store JSON-serializable dicts in memory
    messages: Annotated[list[dict], add_messages]

# dict -> BaseMessage
def dict_to_message(msg: dict) -> BaseMessage:
    role = msg["role"]
    content = msg["content"]
    if role == "user":
        return HumanMessage(content=content)
    elif role == "assistant":
        return AIMessage(content=content)
    else:
        return BaseMessage(content=content)

# BaseMessage -> dict
def message_to_dict(msg: BaseMessage) -> dict:
    if hasattr(msg, "type"):
        role = "user" if msg.type == "human" else "assistant" if msg.type == "ai" else "system"
    else:
        role = "system"
    return {"role": role, "content": msg.content}



def chat_node(state: ChatState):
    messages_obj = []

    # Ensure each item is dict first
    for m in state["messages"]:
        if isinstance(m, dict):
            role = m["role"]
            content = m["content"]
            if role == "user":
                messages_obj.append(HumanMessage(content=content))
            elif role == "assistant":
                messages_obj.append(AIMessage(content=content))
            else:
                messages_obj.append(BaseMessage(content=content))
        elif isinstance(m, BaseMessage):
            # Already BaseMessage, just append
            messages_obj.append(m)
        else:
            raise TypeError(f"Unexpected message type: {type(m)}")

    # Convert messages_obj -> dicts for model
    messages_for_model = [
        {"role": "user" if m.type=="human" else "assistant", "content": m.content}
        for m in messages_obj
    ]

    # Call model
    response_text = invoke_model(messages_for_model)

    # Wrap in BaseMessage
    response_msg = AIMessage(content=response_text)

    # Return JSON-serializable dict
    return {
        "messages": [
            {"role": "assistant", "content": response_msg.content}
        ]
    }

# Checkpointer
checkpointer = InMemorySaver() 

# Construct graph
graph = StateGraph(ChatState)
graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

cb= graph.compile(checkpointer=checkpointer)