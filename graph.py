from __future__ import annotations

import os
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from tools import ALL_TOOLS

load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


SYSTEM_PROMPT = """You are Buraq, an intelligent Gmail assistant.
You must reason step by step using tools when needed.

Available capabilities:
- Read or search inbox emails
- Retrieve grounded project knowledge from the Lab 2 vector database
- Find urgent alerts and deadlines
- Draft emails for user review
- Send a reviewed draft only after explicit approval

Rules:
1. If the answer depends on inbox data, deadlines, meeting notes, or old emails, use a tool.
2. Use search_knowledge_base for questions about grounded project memory.
3. Never claim an email was sent unless send_reviewed_email succeeded.
4. Draft first, then wait for explicit approval before sending.
5. If no tool is needed, answer directly and clearly.
"""


def get_bound_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is required to run the live LangGraph agent.")

    try:
        from langchain_groq import ChatGroq
    except ImportError as exc:
        raise RuntimeError(
            "langchain_groq is not installed. Install project dependencies before running the live agent."
        ) from exc

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0,
    )
    return llm.bind_tools(ALL_TOOLS)


def agent_node_factory(model=None):
    def agent_node(state: AgentState) -> AgentState:
        llm = model or get_bound_llm()
        messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        response = llm.invoke(messages)
        return {"messages": [response]}

    return agent_node


tool_node = ToolNode(tools=ALL_TOOLS)


def router(state: AgentState) -> Literal["tools", "__end__"]:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return "__end__"


def build_graph(model=None):
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node_factory(model=model))
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        router,
        {
            "tools": "tools",
            "__end__": END,
        },
    )
    workflow.add_edge("tools", "agent")
    return workflow.compile()


buraq_graph = build_graph()


def chat(user_message: str, history: list[BaseMessage] | None = None, model=None) -> str:
    if history is None:
        history = []

    graph = buraq_graph if model is None else build_graph(model=model)
    result = graph.invoke({"messages": [*history, HumanMessage(content=user_message)]})
    last_message = result["messages"][-1]
    if isinstance(last_message, AIMessage):
        return last_message.content
    return str(last_message.content)


__all__ = [
    "AgentState",
    "SYSTEM_PROMPT",
    "tool_node",
    "router",
    "build_graph",
    "buraq_graph",
    "chat",
]
