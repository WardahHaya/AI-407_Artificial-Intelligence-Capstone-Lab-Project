# agent/multi_agent_graph.py
# Multi-Agent LangGraph for Buraq
# Supervisor → decides path → Researcher and/or Analyst
# Smart routing saves 60-70% tokens vs running both agents every time

import os
import logging
from typing import Annotated, Literal
from typing_extensions import TypedDict

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

from agent.agents_config import (
    RESEARCHER_PERSONA,
    ANALYST_PERSONA,
    HANDOFF_SIGNAL,
)
from agent.tools import ALL_TOOLS

load_dotenv()

# ══════════════════════════════════════════
#  LOGGING SETUP
# ══════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler("collaboration_trace.log", mode="w"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BuraqMultiAgent")


# ══════════════════════════════════════════
#  AGENT STATE
# ══════════════════════════════════════════
class MultiAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    active_agent: str        # "supervisor", "researcher", "analyst", "direct"
    research_output: str     # what Researcher found
    final_answer: str        # final response to user
    route: str               # supervisor decision: "direct", "researcher_only", "full_pipeline"


# ══════════════════════════════════════════
#  LLM
# ══════════════════════════════════════════
def get_llm():
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0,
    )


# ══════════════════════════════════════════
#  HELPER — extract draft from messages
# ══════════════════════════════════════════
def extract_draft_from_messages(messages: list) -> str:
    for msg in reversed(messages):
        content = ""
        if isinstance(msg, ToolMessage):
            content = msg.content or ""
        elif isinstance(msg, AIMessage):
            content = msg.content or ""
        if "Draft Email" in content and "To:" in content and "Subject:" in content:
            return content
    return ""


# ══════════════════════════════════════════
#  SUPERVISOR NODE
#  Reads the query and decides which path:
#  - direct: simple chat, no tools needed
#  - researcher_only: just read/search emails
#  - full_pipeline: research + write/summarize
#  Only costs 1 small LLM call
# ══════════════════════════════════════════
def supervisor_node(state: MultiAgentState) -> MultiAgentState:
    logger.info("=" * 60)
    logger.info("SUPERVISOR — Classifying query...")
    logger.info("=" * 60)

    last_user_msg = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break

    llm = get_llm()
    classification = llm.invoke([
        SystemMessage(content=(
            "You are a routing supervisor for Buraq, an email assistant. "
            "Classify the user query into exactly one of these three routes:\n\n"
            "1. DIRECT — Simple conversation, greetings, questions about Buraq itself, "
            "or anything that does NOT require reading emails or writing emails. "
            "Examples: 'hello', 'what can you do', 'thanks'\n\n"
            "2. RESEARCHER_ONLY — Query only needs reading, searching, or retrieving "
            "email data. No writing, drafting, or summarizing needed. "
            "Examples: 'check my inbox', 'any emails from Ahmad', "
            "'did anyone reply', 'check spam'\n\n"
            "3. FULL_PIPELINE — Query needs both research AND writing/synthesis. "
            "Includes drafting emails, email summaries, daily digests, "
            "or any task requiring both data gathering and content creation. "
            "Examples: 'draft an email', 'summarize my emails', "
            "'write a reply to Ahmad', 'what needs my attention today'\n\n"
            "Reply with ONLY one word: DIRECT, RESEARCHER_ONLY, or FULL_PIPELINE."
        )),
        HumanMessage(content=last_user_msg)
    ])

    route_text = classification.content.strip().upper()
    if "DIRECT" in route_text:
        route = "direct"
    elif "RESEARCHER_ONLY" in route_text:
        route = "researcher_only"
    else:
        route = "full_pipeline"

    logger.info(f"SUPERVISOR — Route decided: {route.upper()}")

    return {
        "messages": state["messages"],
        "active_agent": "supervisor",
        "research_output": state.get("research_output", ""),
        "final_answer": state.get("final_answer", ""),
        "route": route,
    }


# ══════════════════════════════════════════
#  DIRECT NODE
#  Handles simple queries without any tools
#  Saves tokens by skipping both agents
# ══════════════════════════════════════════
def direct_node(state: MultiAgentState) -> MultiAgentState:
    logger.info("=" * 60)
    logger.info("DIRECT — Answering without agents...")
    logger.info("=" * 60)

    llm = get_llm()
    response = llm.invoke(
        [SystemMessage(content=(
            "You are Buraq, an intelligent Gmail assistant for Wardah Haya. "
            "Answer the user's question directly and helpfully. "
            "Be concise and friendly."
        ))] + state["messages"]
    )

    logger.info(f"DIRECT response: {response.content[:200]}...")
    return {
        "messages": [response],
        "active_agent": "direct",
        "research_output": "",
        "final_answer": response.content,
        "route": state.get("route", "direct"),
    }


# ══════════════════════════════════════════
#  AGENT A — RESEARCHER NODE
# ══════════════════════════════════════════
def researcher_node(state: MultiAgentState) -> MultiAgentState:
    logger.info("=" * 60)
    logger.info("AGENT A (RESEARCHER) — Starting data gathering...")
    logger.info("=" * 60)

    llm = get_llm()
    researcher_llm = llm.bind_tools(RESEARCHER_PERSONA["tools"])
    system_msg = SystemMessage(content=RESEARCHER_PERSONA["backstory"])
    response = researcher_llm.invoke([system_msg] + state["messages"])

    logger.info(f"RESEARCHER response: {response.content[:300]}...")

    research_output = ""
    if HANDOFF_SIGNAL in (response.content or ""):
        parts = response.content.split(HANDOFF_SIGNAL)
        research_output = parts[1].strip() if len(parts) > 1 else response.content
        logger.info("RESEARCHER — Handoff signal detected.")
        logger.info(f"RESEARCHER — Output: {research_output[:200]}...")

    return {
        "messages": [response],
        "active_agent": "researcher",
        "research_output": research_output,
        "final_answer": state.get("final_answer", ""),
        "route": state.get("route", ""),
    }


# ══════════════════════════════════════════
#  AGENT B — ANALYST NODE
# ══════════════════════════════════════════
def analyst_node(state: MultiAgentState) -> MultiAgentState:
    logger.info("=" * 60)
    logger.info("AGENT B (ANALYST) — Starting synthesis...")
    logger.info("=" * 60)

    llm = get_llm()
    analyst_llm = llm.bind_tools(ANALYST_PERSONA["tools"])

    research_context = state.get("research_output", "")
    analyst_system = (
        ANALYST_PERSONA["backstory"]
        + (
            f"\n\nResearcher findings:\n{research_context}\n\n"
            f"Use this to give a clear, helpful answer."
            if research_context else ""
        )
        + (
            "\n\nCRITICAL: When draft_email tool returns a draft, "
            "copy the ENTIRE draft verbatim into your response. "
            "Never just say 'please review' without showing the full email."
        )
    )

    response = analyst_llm.invoke(
        [SystemMessage(content=analyst_system)] + state["messages"]
    )

    logger.info(f"ANALYST response: {response.content[:300]}...")

    return {
        "messages": [response],
        "active_agent": "analyst",
        "research_output": state.get("research_output", ""),
        "final_answer": response.content or "",
        "route": state.get("route", ""),
    }


# ══════════════════════════════════════════
#  TOOL NODES
# ══════════════════════════════════════════
researcher_tool_node = ToolNode(tools=RESEARCHER_PERSONA["tools"])
analyst_tool_node = ToolNode(tools=ANALYST_PERSONA["tools"])


# ══════════════════════════════════════════
#  SUPERVISOR ROUTER
#  Runs after supervisor_node
# ══════════════════════════════════════════
def supervisor_router(
    state: MultiAgentState,
) -> Literal["direct", "researcher", "analyst"]:
    route = state.get("route", "full_pipeline")
    if route == "direct":
        logger.info("SUPERVISOR ROUTER → direct")
        return "direct"
    elif route == "researcher_only":
        logger.info("SUPERVISOR ROUTER → researcher only")
        return "researcher"
    else:
        logger.info("SUPERVISOR ROUTER → full pipeline (researcher first)")
        return "researcher"


# ══════════════════════════════════════════
#  RESEARCHER ROUTER
# ══════════════════════════════════════════
def researcher_router(
    state: MultiAgentState,
) -> Literal["researcher_tools", "analyst", "__end__"]:
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        logger.info("ROUTER — Researcher has tool calls, executing...")
        return "researcher_tools"

    if HANDOFF_SIGNAL in (last_message.content or ""):
        logger.info("ROUTER — Handoff detected, routing to Analyst...")
        return "analyst"

    # researcher_only route — end after research
    if state.get("route") == "researcher_only":
        logger.info("ROUTER — Researcher only route, ending...")
        return "__end__"

    # full_pipeline but no handoff signal — still go to analyst
    logger.info("ROUTER — Routing to Analyst...")
    return "analyst"


# ══════════════════════════════════════════
#  ANALYST ROUTER
# ══════════════════════════════════════════
def analyst_router(
    state: MultiAgentState,
) -> Literal["analyst_tools", "__end__"]:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        logger.info("ROUTER — Analyst has tool calls, executing...")
        return "analyst_tools"
    logger.info("ROUTER — Analyst gave final answer, ending.")
    return "__end__"


# ══════════════════════════════════════════
#  BUILD THE GRAPH
# ══════════════════════════════════════════
def build_multi_agent_graph():
    graph = StateGraph(MultiAgentState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("direct", direct_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("researcher_tools", researcher_tool_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("analyst_tools", analyst_tool_node)

    graph.set_entry_point("supervisor")

    graph.add_conditional_edges(
        source="supervisor",
        path=supervisor_router,
        path_map={
            "direct": "direct",
            "researcher": "researcher",
            "analyst": "analyst",
        }
    )

    graph.add_edge("direct", END)

    graph.add_conditional_edges(
        source="researcher",
        path=researcher_router,
        path_map={
            "researcher_tools": "researcher_tools",
            "analyst": "analyst",
            "__end__": END,
        }
    )
    graph.add_edge("researcher_tools", "researcher")

    graph.add_conditional_edges(
        source="analyst",
        path=analyst_router,
        path_map={
            "analyst_tools": "analyst_tools",
            "__end__": END,
        }
    )
    graph.add_edge("analyst_tools", "analyst")

    return graph.compile()


buraq_multi_agent = build_multi_agent_graph()


# ══════════════════════════════════════════
#  Public chat function
# ══════════════════════════════════════════
def chat(user_message: str, history: list = None) -> str:
    if history is None:
        history = []

    logger.info(f"USER: {user_message}")

    input_messages = history + [HumanMessage(content=user_message)]

    result = buraq_multi_agent.invoke({
        "messages": input_messages,
        "active_agent": "supervisor",
        "research_output": "",
        "final_answer": "",
        "route": "",
    })

    final = result.get("final_answer", "").strip()
    all_messages = result.get("messages", [])

    # Ensure draft emails are always shown
    if "Draft Email" not in final:
        draft = extract_draft_from_messages(all_messages)
        if draft:
            final = draft + ("\n\n" + final if final else "")

    if not final:
        final = all_messages[-1].content if all_messages else "Sorry, something went wrong."

    logger.info(f"FINAL ANSWER TO USER: {final[:300]}...")
    logger.info("=" * 60)

    return final


# ══════════════════════════════════════════
#  CLI Test
# ══════════════════════════════════════════
if __name__ == "__main__":
    print("\nBuraq AI — Smart Multi-Agent Mode")
    print("Fly Above Your Inbox")
    print("=" * 40)
    print("Supervisor routes → Direct / Researcher / Full Pipeline")
    print("Type 'exit' to quit\n")

    history = []
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        if not user_input:
            continue

        print("\nBuraq: thinking...\n")
        response = chat(user_input, history)
        print(f"Buraq: {response}\n")
        print("-" * 40)

        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=response))
