"""Offline runnable LangGraph agent loop recorded and replayed with rhinotape.

A two-step agent (model asks for a tool, tool runs, model answers) is recorded against a
mock OpenAI upstream, then replayed with no upstream at all. The replay is checked with
`assert_tool_sequence`, `assert_no_tool_loop` and `assert_cost_under`.

Needs the optional packages used by the agent:
    pip install langgraph langchain-openai

Run with:
    .venv/bin/python examples/langgraph_agent.py
"""

from __future__ import annotations

import json

import httpx
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from rhinotape import MemoryCassetteStore, TapeMode
from rhinotape.integrations import TapeConfig, make_httpx_client


@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"21C in {city}"


def _mock_openai(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    if any(m["role"] == "tool" for m in body["messages"]):
        message: dict[str, object] = {"role": "assistant", "content": "It is 21C in Cairo."}
    else:
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Cairo"}'},
                }
            ],
        }
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-demo",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        },
    )


def run_agent(cfg: TapeConfig, upstream: httpx.BaseTransport, mode: TapeMode) -> str:
    with cfg.session("langgraph_agent", mode=mode, patch_httpx=False) as session:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key="test-key",
            http_client=make_httpx_client(session, upstream=upstream),
        ).bind_tools([get_weather])

        def call_model(state: MessagesState) -> dict[str, list[object]]:
            return {"messages": [llm.invoke(state["messages"])]}

        graph = StateGraph(MessagesState)
        graph.add_node("agent", call_model)
        graph.add_node("tools", ToolNode([get_weather]))
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", tools_condition)
        graph.add_edge("tools", "agent")
        result = graph.compile().invoke({"messages": [("user", "Weather in Cairo?")]})

        session.assert_call_count(2)
        session.assert_tool_sequence(["get_weather"])
        session.assert_no_tool_loop(max_consecutive=2)
        session.assert_cost_under(0.05)
        return str(result["messages"][-1].content)


def main() -> None:
    cfg = TapeConfig(store=MemoryCassetteStore())

    answer = run_agent(cfg, httpx.MockTransport(_mock_openai), TapeMode.RECORD)
    print(f"recorded: {answer}")

    # Replay with an upstream that fails every request: nothing may reach it.
    offline = httpx.MockTransport(lambda request: httpx.Response(500))
    replayed = run_agent(cfg, offline, TapeMode.REPLAY)
    assert replayed == answer
    print(f"replayed offline: {replayed}")


if __name__ == "__main__":
    main()
