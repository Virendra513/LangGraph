from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import os
import json
import asyncio
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

llm = InferenceClient(api_key=os.environ["HF_TOKEN"])

# MCP Client for local FastMCP Server
mcp_client = MultiServerMCPClient(
    {
        "Demo Server": {
            "transport": "stdio",
            "command": "E:\\repo\\langG\\langG\\Scripts\\python.exe",
            "args": [
                "-u",
                "E:\\repo\\mcp\\local_server\\expense-tracker-mcp-server\\main_demo_server.py"
            ],
        },
        "Fetch Server": {
            "transport": "streamable_http",
            "url": "https://remote.mcpservers.org/fetch/mcp"
        }
    }
)


# ── Helpers ────────────────────────────────────────────────────────────────────

async def get_formatted_tools() -> tuple[list, list]:
    """
    Returns two lists:
      - lc_tools   : raw LangChain tool objects  (used to call tools)
      - api_tools  : OpenAI-schema dicts          (sent to the LLM)
    """
    lc_tools = await mcp_client.get_tools()

    api_tools = []
    for t in lc_tools:
        if isinstance(t, dict):
            params = t.get("input_schema", {})
            name   = t.get("name", "")
            desc   = t.get("description", "")
        else:
            name  = t.name
            desc  = t.description
            schema = t.args_schema
            # args_schema can be a Pydantic model class OR already a plain dict
            if isinstance(schema, dict):
                params = schema
            elif hasattr(schema, "schema"):
                params = schema.schema()        # Pydantic v1
            elif hasattr(schema, "model_json_schema"):
                params = schema.model_json_schema()  # Pydantic v2
            else:
                params = {}

        api_tools.append({
            "type": "function",
            "function": {
                "name":        name,
                "description": desc,
                "parameters":  params,
            }
        })

    return lc_tools, api_tools


def unwrap_mcp_content(content) -> str:
    """
    MCP tool results often come back as a list of typed blocks, e.g.
      [{'type': 'text', 'text': '3', 'id': '...'}]
    Unwrap to a plain string so the LLM receives clean text.
    """
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", str(block)))
            else:
                parts.append(str(block))
        return " ".join(parts)
    return str(content) if content else ""


async def message_to_dict(msg) -> dict:
    """Convert a LangChain BaseMessage to the dict the HF API expects."""
    if isinstance(msg, dict):
        return msg

    role_map = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}
    role = role_map.get(getattr(msg, "type", ""), "user")

    # ToolMessage MUST include tool_call_id — without it the API rejects the request
    if isinstance(msg, ToolMessage):
        return {
            "role":         "tool",
            "tool_call_id": msg.tool_call_id,
            "content":      unwrap_mcp_content(msg.content),
        }

    # AIMessage with tool_calls — preserve the tool_calls payload for history replay
    if isinstance(msg, AIMessage) and msg.additional_kwargs.get("tool_calls"):
        return {
            "role":       "assistant",
            "content":    msg.content or "",
            "tool_calls": msg.additional_kwargs["tool_calls"],
        }

    return {"role": role, "content": msg.content or ""}


async def retrieve_all_threads(checkpointer) -> list[str]:
    """Return every thread_id stored in the checkpointer."""
    return list({
        cp.config["configurable"]["thread_id"]
        for cp in checkpointer.list(None)
    })


# ── Graph builder ──────────────────────────────────────────────────────────────

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


async def build_graph(checkpointer):
    lc_tools, api_tools = await get_formatted_tools()

    # Map tool name → callable LangChain tool (for manual dispatch)
    tool_map = {
        (t["name"] if isinstance(t, dict) else t.name): t
        for t in lc_tools
    }

    # ── Node: call LLM with tools ──────────────────────────────────────────────
    async def chat_node(state: ChatState):
        messages_for_model = [
            await message_to_dict(m) for m in state["messages"]
        ]

        # Stream the response; collect both text and tool_calls
        stream = llm.chat.completions.create(
            model="openai/gpt-oss-120b:novita",
            messages=messages_for_model,
            tools=api_tools,           # ✅ tools now sent to the model
            stream=True,
            max_tokens=1024,
            temperature=0.7,
            top_p=0.9,
        )

        full_text   = ""
        tool_calls  = {}          # index → {id, name, arguments}

        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            # Accumulate text content
            if getattr(delta, "content", None):
                full_text += delta.content

            # Accumulate tool_call deltas (streamed in pieces)
            for tc in getattr(delta, "tool_calls", None) or []:
                idx = tc.index
                if idx not in tool_calls:
                    tool_calls[idx] = {"id": tc.id, "name": "", "arguments": ""}
                if tc.function.name:
                    tool_calls[idx]["name"] += tc.function.name
                if tc.function.arguments:
                    tool_calls[idx]["arguments"] += tc.function.arguments

        # Build the assistant message
        if tool_calls:
            # Return an AIMessage that carries the tool_call requests
            calls_payload = [
                {
                    "id":       v["id"],
                    "type":     "function",
                    "function": {"name": v["name"], "arguments": v["arguments"]},
                }
                for v in tool_calls.values()
            ]
            return {
                "messages": [
                    AIMessage(
                        content=full_text,
                        additional_kwargs={"tool_calls": calls_payload},
                    )
                ]
            }

        return {"messages": [{"role": "assistant", "content": full_text}]}

    # ── Node: execute tool calls ───────────────────────────────────────────────
    async def tool_node(state: ChatState):
        last_msg = state["messages"][-1]

        # Extract tool_calls from the last AI message
        raw_calls = (
            last_msg.additional_kwargs.get("tool_calls", [])
            if hasattr(last_msg, "additional_kwargs")
            else []
        )

        results = []
        for tc in raw_calls:
            name      = tc["function"]["name"]
            args      = json.loads(tc["function"]["arguments"] or "{}")
            call_id   = tc["id"]

            tool = tool_map.get(name)
            if tool is None:
                output = f"Error: tool '{name}' not found."
            else:
                try:
                    # LangChain tools use .ainvoke for async
                    output = await tool.ainvoke(args)
                except Exception as e:
                    output = f"Error calling {name}: {e}"

            results.append(
                ToolMessage(content=str(output), tool_call_id=call_id)
            )

        return {"messages": results}

    # ── Routing: does the last message contain tool calls? ─────────────────────
    def should_use_tools(state: ChatState) -> str:
        last = state["messages"][-1]
        has_calls = (
            hasattr(last, "additional_kwargs")
            and bool(last.additional_kwargs.get("tool_calls"))
        )
        return "tools" if has_calls else END

    # ── Assemble the graph ─────────────────────────────────────────────────────
    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools",     tool_node)

    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", should_use_tools)
    graph.add_edge("tools", "chat_node")

    return graph.compile(checkpointer=checkpointer)


# ── Entry point ────────────────────────────────────────────────────────────────

async def main():
    async with AsyncSqliteSaver.from_conn_string("cb.db") as checkpointer:
        cb     = await build_graph(checkpointer)
        CONFIG = {"configurable": {"thread_id": "demo-thread2"}}

        response = await cb.ainvoke(
            {"messages": [{"role": "user", "content": "go to website google.com bring first 5 words "}]},
            config=CONFIG,
        )
        print(response)


if __name__ == "__main__":
    asyncio.run(main())