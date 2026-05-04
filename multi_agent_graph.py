from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from agents_config import ANALYST_PERSONA, HANDOFF_SIGNAL, RESEARCHER_PERSONA

load_dotenv()

TRACE_LOG_PATH = Path("collaboration_trace.log")


class MultiAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    active_agent: str
    route: str
    research_output: str
    final_answer: str


def get_trace_logger(log_path: Path | str = TRACE_LOG_PATH, reset: bool = False) -> logging.Logger:
    logger = logging.getLogger("BuraqMultiAgent")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if reset:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s | %(message)s")
        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def describe_tool_calls(message: AIMessage) -> str:
    tool_calls = getattr(message, "tool_calls", None) or []
    if not tool_calls:
        return "no tool calls"
    parts = []
    for call in tool_calls:
        parts.append(f"{call['name']}({call['args']})")
    return "; ".join(parts)


def get_live_llm(bind_tools=None):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is required to run the live multi-agent graph.")

    try:
        from langchain_groq import ChatGroq
    except ImportError as exc:
        raise RuntimeError(
            "langchain_groq is not installed. Install project dependencies before running the live multi-agent graph."
        ) from exc

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0,
    )
    if bind_tools:
        return llm.bind_tools(bind_tools)
    return llm


def extract_draft_from_messages(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, ToolMessage) and "Draft Email" in (message.content or ""):
            return message.content
        if isinstance(message, AIMessage) and "Draft Email" in (message.content or ""):
            return message.content
    return ""


def get_last_tool_message(messages: list[BaseMessage]) -> ToolMessage | None:
    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            return message
    return None


def supervisor_node_factory(model=None, logger: logging.Logger | None = None):
    trace_logger = logger or get_trace_logger()

    def supervisor_node(state: MultiAgentState) -> MultiAgentState:
        trace_logger.info("SUPERVISOR | Reviewing user request.")
        last_user_message = next(
            (message.content for message in reversed(state["messages"]) if isinstance(message, HumanMessage)),
            "",
        )

        if model is None:
            llm = get_live_llm()
            response = llm.invoke(
                [
                    SystemMessage(
                        content=(
                            "Classify the request as DIRECT, RESEARCHER_ONLY, or FULL_PIPELINE.\n"
                            "DIRECT = no tools needed.\n"
                            "RESEARCHER_ONLY = only evidence gathering.\n"
                            "FULL_PIPELINE = research plus writing or synthesis.\n"
                            "Reply with one label only."
                        )
                    ),
                    HumanMessage(content=last_user_message),
                ]
            )
        else:
            response = model.invoke(state["messages"])

        route_text = (response.content or "").strip().upper()
        if "DIRECT" in route_text:
            route = "direct"
        elif "RESEARCHER_ONLY" in route_text:
            route = "researcher_only"
        else:
            route = "full_pipeline"

        trace_logger.info("SUPERVISOR | Route selected: %s", route)
        return {
            "messages": state["messages"],
            "active_agent": "supervisor",
            "route": route,
            "research_output": state.get("research_output", ""),
            "final_answer": state.get("final_answer", ""),
        }

    return supervisor_node


def direct_node_factory(model=None, logger: logging.Logger | None = None):
    trace_logger = logger or get_trace_logger()

    def direct_node(state: MultiAgentState) -> MultiAgentState:
        trace_logger.info("DIRECT | Answering without specialist handoff.")
        llm = model or get_live_llm()
        response = llm.invoke(
            [
                SystemMessage(content="You are Buraq. Answer directly and concisely without calling tools."),
                *state["messages"],
            ]
        )
        trace_logger.info("DIRECT | Final response prepared.")
        return {
            "messages": [response],
            "active_agent": "direct",
            "route": state.get("route", "direct"),
            "research_output": state.get("research_output", ""),
            "final_answer": response.content or "",
        }

    return direct_node


def researcher_node_factory(model=None, logger: logging.Logger | None = None):
    trace_logger = logger or get_trace_logger()

    def researcher_node(state: MultiAgentState) -> MultiAgentState:
        trace_logger.info("RESEARCHER | Starting grounded evidence gathering.")
        llm = model or get_live_llm(bind_tools=RESEARCHER_PERSONA["restricted_tools"])
        response = llm.invoke([SystemMessage(content=RESEARCHER_PERSONA["backstory"]), *state["messages"]])

        if getattr(response, "tool_calls", None):
            trace_logger.info("RESEARCHER | Tool request: %s", describe_tool_calls(response))
        else:
            trace_logger.info("RESEARCHER | Internal summary: %s", (response.content or "").strip())

        research_output = state.get("research_output", "")
        if HANDOFF_SIGNAL in (response.content or ""):
            research_output = response.content.split(HANDOFF_SIGNAL, 1)[1].strip()
            trace_logger.info("RESEARCHER | Handoff ready for Analyst.")

        return {
            "messages": [response],
            "active_agent": "researcher",
            "route": state.get("route", ""),
            "research_output": research_output,
            "final_answer": state.get("final_answer", ""),
        }

    return researcher_node


def analyst_node_factory(model=None, logger: logging.Logger | None = None):
    trace_logger = logger or get_trace_logger()

    def analyst_node(state: MultiAgentState) -> MultiAgentState:
        trace_logger.info("ANALYST | Starting synthesis from Researcher handoff.")
        llm = model or get_live_llm(bind_tools=ANALYST_PERSONA["restricted_tools"])

        research_context = state.get("research_output", "")
        system_prompt = ANALYST_PERSONA["backstory"]
        if research_context:
            system_prompt += f"\n\nResearcher handoff:\n{research_context}"

        response = llm.invoke([SystemMessage(content=system_prompt), *state["messages"]])
        if getattr(response, "tool_calls", None):
            trace_logger.info("ANALYST | Tool request: %s", describe_tool_calls(response))
        else:
            trace_logger.info("ANALYST | Final synthesis: %s", (response.content or "").strip())

        return {
            "messages": [response],
            "active_agent": "analyst",
            "route": state.get("route", ""),
            "research_output": research_context,
            "final_answer": response.content or "",
        }

    return analyst_node


def researcher_tool_node_factory(logger: logging.Logger | None = None):
    trace_logger = logger or get_trace_logger()
    tool_node = ToolNode(tools=RESEARCHER_PERSONA["restricted_tools"])

    def researcher_tools(state: MultiAgentState) -> dict[str, list[BaseMessage]]:
        trace_logger.info("RESEARCHER_TOOLS | Executing restricted toolset.")
        result = tool_node.invoke(state)
        for message in result["messages"]:
            if isinstance(message, ToolMessage):
                trace_logger.info("RESEARCHER_TOOLS | Tool result:\n%s", message.content)
        return result

    return researcher_tools


def analyst_tool_node_factory(logger: logging.Logger | None = None):
    trace_logger = logger or get_trace_logger()
    tool_node = ToolNode(tools=ANALYST_PERSONA["restricted_tools"])

    def analyst_tools(state: MultiAgentState) -> dict[str, list[BaseMessage]]:
        trace_logger.info("ANALYST_TOOLS | Executing restricted toolset.")
        result = tool_node.invoke(state)
        for message in result["messages"]:
            if isinstance(message, ToolMessage):
                trace_logger.info("ANALYST_TOOLS | Tool result:\n%s", message.content)
        return result

    return analyst_tools


def supervisor_router(state: MultiAgentState) -> Literal["direct", "researcher", "analyst"]:
    route = state.get("route", "full_pipeline")
    if route == "direct":
        return "direct"
    if route == "researcher_only":
        return "researcher"
    return "researcher"


def researcher_router(state: MultiAgentState) -> Literal["researcher_tools", "analyst", "__end__"]:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "researcher_tools"
    if HANDOFF_SIGNAL in (last_message.content or ""):
        return "analyst"
    if state.get("route") == "researcher_only":
        return "__end__"
    return "analyst"


def analyst_router(state: MultiAgentState) -> Literal["analyst_tools", "__end__"]:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "analyst_tools"
    return "__end__"


def build_multi_agent_graph(
    supervisor_model=None,
    direct_model=None,
    researcher_model=None,
    analyst_model=None,
    logger: logging.Logger | None = None,
):
    trace_logger = logger or get_trace_logger()
    workflow = StateGraph(MultiAgentState)

    workflow.add_node("supervisor", supervisor_node_factory(model=supervisor_model, logger=trace_logger))
    workflow.add_node("direct", direct_node_factory(model=direct_model, logger=trace_logger))
    workflow.add_node("researcher", researcher_node_factory(model=researcher_model, logger=trace_logger))
    workflow.add_node("researcher_tools", researcher_tool_node_factory(logger=trace_logger))
    workflow.add_node("analyst", analyst_node_factory(model=analyst_model, logger=trace_logger))
    workflow.add_node("analyst_tools", analyst_tool_node_factory(logger=trace_logger))

    workflow.set_entry_point("supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        supervisor_router,
        {
            "direct": "direct",
            "researcher": "researcher",
            "analyst": "analyst",
        },
    )
    workflow.add_edge("direct", END)
    workflow.add_conditional_edges(
        "researcher",
        researcher_router,
        {
            "researcher_tools": "researcher_tools",
            "analyst": "analyst",
            "__end__": END,
        },
    )
    workflow.add_edge("researcher_tools", "researcher")
    workflow.add_conditional_edges(
        "analyst",
        analyst_router,
        {
            "analyst_tools": "analyst_tools",
            "__end__": END,
        },
    )
    workflow.add_edge("analyst_tools", "analyst")
    return workflow.compile()


def chat(
    user_message: str,
    history: list[BaseMessage] | None = None,
    supervisor_model=None,
    direct_model=None,
    researcher_model=None,
    analyst_model=None,
    logger: logging.Logger | None = None,
) -> str:
    if history is None:
        history = []

    trace_logger = logger or get_trace_logger()
    trace_logger.info("USER | %s", user_message)

    graph = build_multi_agent_graph(
        supervisor_model=supervisor_model,
        direct_model=direct_model,
        researcher_model=researcher_model,
        analyst_model=analyst_model,
        logger=trace_logger,
    )
    result = graph.invoke(
        {
            "messages": [*history, HumanMessage(content=user_message)],
            "active_agent": "supervisor",
            "route": "",
            "research_output": "",
            "final_answer": "",
        }
    )

    final_answer = (result.get("final_answer") or "").strip()
    if "Draft Email" not in final_answer:
        draft = extract_draft_from_messages(result.get("messages", []))
        if draft:
            final_answer = f"{draft}\n\n{final_answer}".strip()

    if not final_answer:
        final_answer = result["messages"][-1].content

    trace_logger.info("FINAL | %s", final_answer)
    return final_answer


class ScriptedSupervisorModel:
    def invoke(self, messages):
        return AIMessage(content="FULL_PIPELINE")


class ScriptedResearcherModel:
    def invoke(self, messages):
        last_tool_message = get_last_tool_message(messages)
        if last_tool_message is None:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_knowledge_base",
                        "args": {
                            "query": "updated resume interview",
                            "department": "careers",
                            "top_k": 1,
                        },
                        "id": "research_call_1",
                        "type": "tool_call",
                    }
                ],
            )

        return AIMessage(
            content=(
                f"{HANDOFF_SIGNAL}\n"
                "- A recruiter email was found in the grounded knowledge base.\n"
                "- Sender: Talent Team <careers@neuralbridge.ai>.\n"
                "- Request: upload the updated resume before May 6 and confirm interview availability.\n"
                "- Recommended next step: draft a professional reply confirming the resume will be shared tonight."
            )
        )


class ScriptedAnalystModel:
    def invoke(self, messages):
        last_tool_message = get_last_tool_message(messages)
        if last_tool_message is None or "Draft Email" not in (last_tool_message.content or ""):
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "draft_email",
                        "args": {
                            "to": "Talent Team <careers@neuralbridge.ai>",
                            "subject": "Updated resume for interview",
                            "context": (
                                "Thank them for the update, confirm that I will send my updated resume tonight, "
                                "and mention that I am available for the interview."
                            ),
                            "tone": "professional",
                        },
                        "id": "analyst_call_1",
                        "type": "tool_call",
                    }
                ],
            )

        return AIMessage(
            content=(
                "I checked the grounded recruiter evidence and prepared the reply below.\n\n"
                f"{last_tool_message.content}"
            )
        )


def run_collaboration_demo(log_path: Path | str = TRACE_LOG_PATH) -> str:
    trace_logger = get_trace_logger(log_path=log_path, reset=True)
    demo_query = (
        "Check whether any recruiter asked for my updated resume and then draft a professional reply "
        "saying I will send it tonight."
    )

    final_answer = chat(
        user_message=demo_query,
        supervisor_model=ScriptedSupervisorModel(),
        researcher_model=ScriptedResearcherModel(),
        analyst_model=ScriptedAnalystModel(),
        logger=trace_logger,
    )
    return final_answer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Buraq's Lab 4 multi-agent graph.")
    parser.add_argument(
        "--demo-trace",
        action="store_true",
        help="Run the deterministic collaboration demo and write collaboration_trace.log.",
    )
    parser.add_argument(
        "--chat",
        help="Optional live chat query to run with the real multi-agent graph.",
    )
    args = parser.parse_args()

    if args.chat:
        print(chat(args.chat))
        return

    final_answer = run_collaboration_demo()
    print(final_answer)


if __name__ == "__main__":
    main()

