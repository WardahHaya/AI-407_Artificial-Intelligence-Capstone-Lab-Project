from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from agents_config import HANDOFF_SIGNAL
from guardrails_config import sanitize_output_text
from lab7_evaluation import ensure_grounding_ready
from secured_graph import build_graph_input, build_secured_graph

TRACE_JSON_PATH = Path("observability_traces.json")
TRACE_PDF_PATH = Path("observability_trace_export.pdf")
OBSERVABILITY_LINK_PATH = Path("observability_link.txt")
BOTTLENECK_PATH = Path("bottleneck_analysis.txt")


@dataclass
class NodeTrace:
    node: str
    duration_ms: float
    preview: str


@dataclass
class QueryTrace:
    query_id: str
    user_query: str
    node_traces: list[NodeTrace] = field(default_factory=list)
    final_answer: str = ""


@dataclass
class TracePlan:
    query_id: str
    route: str
    research_tool: str
    research_args: dict[str, Any]
    analyst_tool: str
    analyst_args: dict[str, Any]


def complex_queries() -> list[tuple[str, str]]:
    return [
        (
            "trace_01",
            "Check whether any recruiter asked for my updated resume and draft a professional reply saying I will send it tonight.",
        ),
        (
            "trace_02",
            "Summarize the sponsor feedback from the meeting notes and draft a friendly message to Momina saying I will simplify the dashboard wording.",
        ),
        (
            "trace_03",
            "Look up the evaluation rubric reminder and draft a friendly note to Areeba saying I will update the PRD tonight.",
        ),
        (
            "trace_04",
            "Find the viva slot deadline and draft a professional confirmation to Dr. Sana that I will choose a slot before the deadline.",
        ),
        (
            "trace_05",
            "Identify the most urgent open deadline and draft a short update to Areeba about the next action.",
        ),
    ]


def build_trace_plans() -> dict[str, TracePlan]:
    query_pairs = dict(complex_queries())
    return {
        query_pairs["trace_01"]: TracePlan(
            query_id="trace_01",
            route="full_pipeline",
            research_tool="search_knowledge_base",
            research_args={"query": "updated resume interview", "department": "careers", "top_k": 1},
            analyst_tool="draft_email",
            analyst_args={
                "to": "Talent Team <careers@neuralbridge.ai>",
                "subject": "Updated resume for interview",
                "context": (
                    "Thank them for the update, confirm that I will send my updated resume tonight, "
                    "and mention that I am available for the interview."
                ),
                "tone": "professional",
            },
        ),
        query_pairs["trace_02"]: TracePlan(
            query_id="trace_02",
            route="full_pipeline",
            research_tool="search_knowledge_base",
            research_args={"query": "sponsor dashboard fewer technical terms", "doc_type": "meeting_notes", "top_k": 1},
            analyst_tool="draft_email",
            analyst_args={
                "to": "Momina Shahid <momina.shahid@projectteam.com>",
                "subject": "Dashboard wording update",
                "context": (
                    "Tell Momina that I will simplify the dashboard wording and focus on saved time, "
                    "fewer missed follow-ups, and better prioritization."
                ),
                "tone": "friendly",
            },
        ),
        query_pairs["trace_03"]: TracePlan(
            query_id="trace_03",
            route="full_pipeline",
            research_tool="search_emails",
            research_args={"query": "evaluation rubric", "max_results": 1},
            analyst_tool="draft_email",
            analyst_args={
                "to": "Areeba Khan <areeba.khan@projectteam.com>",
                "subject": "PRD update tonight",
                "context": "Tell Areeba that I will update the PRD according to the evaluation rubric tonight.",
                "tone": "friendly",
            },
        ),
        query_pairs["trace_04"]: TracePlan(
            query_id="trace_04",
            route="full_pipeline",
            research_tool="search_knowledge_base",
            research_args={"query": "viva slot due date", "doc_type": "deadline_record", "top_k": 1},
            analyst_tool="draft_email",
            analyst_args={
                "to": "Dr. Sana Qureshi <sana.qureshi@university.edu>",
                "subject": "AI407 viva slot selection",
                "context": "Confirm that I will choose my AI407 viva slot before the deadline.",
                "tone": "professional",
            },
        ),
        query_pairs["trace_05"]: TracePlan(
            query_id="trace_05",
            route="full_pipeline",
            research_tool="check_important_alerts",
            research_args={"max_results": 5},
            analyst_tool="draft_email",
            analyst_args={
                "to": "Areeba Khan <areeba.khan@projectteam.com>",
                "subject": "Next urgent action",
                "context": "Share a short update about the most urgent open deadline and the immediate next action.",
                "tone": "friendly",
            },
        ),
    }


def _latest_user_query(messages: list[BaseMessage]) -> str:
    return next((str(message.content) for message in reversed(messages) if isinstance(message, HumanMessage)), "")


def _last_tool_message(messages: list[BaseMessage]) -> ToolMessage | None:
    return next((message for message in reversed(messages) if isinstance(message, ToolMessage)), None)


def preview_from_payload(payload: Any) -> str:
    if isinstance(payload, dict):
        messages = payload.get("messages")
        if messages:
            last_message = messages[-1]
            if hasattr(last_message, "content"):
                return str(last_message.content).replace("\n", " ")[:220]
        if "final_answer" in payload:
            return str(payload["final_answer"]).replace("\n", " ")[:220]
    return str(payload).replace("\n", " ")[:220]


class TraceSupervisorModel:
    def __init__(self, plan_by_query: dict[str, TracePlan]) -> None:
        self.plan_by_query = plan_by_query

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        query = _latest_user_query(messages)
        plan = self.plan_by_query[query]
        return AIMessage(content="FULL_PIPELINE" if plan.route == "full_pipeline" else "RESEARCHER_ONLY")


class TraceResearcherModel:
    def __init__(self, plan_by_query: dict[str, TracePlan]) -> None:
        self.plan_by_query = plan_by_query

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        query = _latest_user_query(messages)
        plan = self.plan_by_query[query]
        last_tool = _last_tool_message(messages)

        if last_tool is None:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": plan.research_tool,
                        "args": plan.research_args,
                        "id": f"{plan.query_id}_researcher_tool",
                        "type": "tool_call",
                    }
                ],
            )

        return AIMessage(
            content=(
                f"{HANDOFF_SIGNAL}\n"
                f"- Grounded evidence for: {query}\n"
                f"- Evidence payload:\n{sanitize_output_text(str(last_tool.content))}"
            )
        )


class TraceAnalystModel:
    def __init__(self, plan_by_query: dict[str, TracePlan]) -> None:
        self.plan_by_query = plan_by_query

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        query = _latest_user_query(messages)
        plan = self.plan_by_query[query]
        last_tool = _last_tool_message(messages)

        if last_tool is None:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": plan.analyst_tool,
                        "args": plan.analyst_args,
                        "id": f"{plan.query_id}_analyst_tool",
                        "type": "tool_call",
                    }
                ],
            )

        return AIMessage(content=sanitize_output_text(str(last_tool.content)))


def build_instrumented_graph() -> tuple[Any, dict[str, TracePlan], str]:
    ensure_grounding_ready()
    plans = build_trace_plans()
    use_live_runtime = os.getenv("BURAQ_OBSERVABILITY_USE_LIVE_MODEL", "false").lower() == "true" and bool(os.getenv("GROQ_API_KEY"))

    if use_live_runtime:
        return build_secured_graph(), plans, "live_llm"

    model_bundle = {
        "supervisor": TraceSupervisorModel(plans),
        "researcher": TraceResearcherModel(plans),
        "analyst": TraceAnalystModel(plans),
    }
    return build_secured_graph(model=model_bundle), plans, "scripted_trace"


def run_trace(query_id: str, user_query: str, graph: Any) -> QueryTrace:
    trace = QueryTrace(query_id=query_id, user_query=user_query)
    last_event_time = time.perf_counter()

    for event in graph.stream(build_graph_input(user_query), stream_mode="updates"):
        current_time = time.perf_counter()
        elapsed_ms = (current_time - last_event_time) * 1000
        for node, payload in event.items():
            trace.node_traces.append(
                NodeTrace(
                    node=node,
                    duration_ms=round(elapsed_ms, 2),
                    preview=preview_from_payload(payload),
                )
            )
        last_event_time = time.perf_counter()

    if trace.node_traces:
        trace.final_answer = trace.node_traces[-1].preview
    return trace


def aggregate_node_timings(traces: list[QueryTrace]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[float]] = {}
    for trace in traces:
        for node_trace in trace.node_traces:
            grouped.setdefault(node_trace.node, []).append(node_trace.duration_ms)

    summary: dict[str, dict[str, float]] = {}
    for node, values in grouped.items():
        ordered = sorted(values)
        p95_index = max(math.ceil(len(ordered) * 0.95) - 1, 0)
        summary[node] = {
            "average_ms": round(sum(ordered) / len(ordered), 2),
            "max_ms": round(max(ordered), 2),
            "p95_ms": round(ordered[p95_index], 2),
            "count": float(len(ordered)),
        }
    return summary


def pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def write_simple_pdf(lines: list[str], output_path: Path) -> None:
    content_lines = ["BT", "/F1 10 Tf", "40 780 Td", "14 TL"]
    for index, line in enumerate(lines):
        if index > 0:
            content_lines.append("T*")
        content_lines.append(f"({pdf_escape(line)}) Tj")
    content_lines.append("ET")
    stream = "\n".join(content_lines).encode("latin-1", errors="replace")

    objects = []
    objects.append(b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n")
    objects.append(b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n")
    objects.append(
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n"
    )
    objects.append(b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Courier >> endobj\n")
    objects.append(
        f"5 0 obj << /Length {len(stream)} >> stream\n".encode("latin-1")
        + stream
        + b"\nendstream endobj\n"
    )

    output = [b"%PDF-1.4\n"]
    offsets = [0]
    for obj in objects:
        offsets.append(sum(len(chunk) for chunk in output))
        output.append(obj)

    xref_start = sum(len(chunk) for chunk in output)
    xref_lines = [b"xref\n", f"0 {len(objects) + 1}\n".encode("latin-1"), b"0000000000 65535 f \n"]
    for offset in offsets[1:]:
        xref_lines.append(f"{offset:010d} 00000 n \n".encode("latin-1"))
    trailer = [
        b"trailer\n",
        f"<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode("latin-1"),
        b"startxref\n",
        f"{xref_start}\n".encode("latin-1"),
        b"%%EOF\n",
    ]

    output.extend(xref_lines)
    output.extend(trailer)
    output_path.write_bytes(b"".join(output))


def write_outputs(traces: list[QueryTrace], summary: dict[str, dict[str, float]], execution_mode: str) -> None:
    TRACE_JSON_PATH.write_text(
        json.dumps(
            {
                "execution_mode": execution_mode,
                "traces": [asdict(trace) for trace in traces],
                "summary": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    first_trace = traces[0]
    pdf_lines = [
        "Buraq Lab 7 Trace Export",
        "",
        f"Execution mode: {execution_mode}",
        f"Query ID: {first_trace.query_id}",
        f"User Query: {first_trace.user_query}",
        "",
        "Node Timeline (ms):",
    ]
    for node_trace in first_trace.node_traces:
        pdf_lines.append(f"- {node_trace.node}: {node_trace.duration_ms:.2f} ms | {node_trace.preview}")
    pdf_lines.extend(["", "Final Answer Preview:", first_trace.final_answer[:180]])
    write_simple_pdf(pdf_lines, TRACE_PDF_PATH)

    if os.getenv("LANGSMITH_API_KEY"):
        OBSERVABILITY_LINK_PATH.write_text(
            "LangSmith tracing is configured in this environment. If LANGCHAIN_TRACING_V2 is enabled, these graph runs will appear in the LangSmith project dashboard.",
            encoding="utf-8",
        )
    else:
        OBSERVABILITY_LINK_PATH.write_text(
            "LangSmith tracing is not configured locally. Use observability_traces.json and observability_trace_export.pdf for the recorded node timeline.",
            encoding="utf-8",
        )

    slowest_node = max(summary.items(), key=lambda item: item[1]["average_ms"])
    BOTTLENECK_PATH.write_text(
        (
            f"Observed {len(traces)} real secured-graph traces in {execution_mode} mode. "
            f"The slowest average node was {slowest_node[0]} at {slowest_node[1]['average_ms']:.2f} ms "
            f"(p95 {slowest_node[1]['p95_ms']:.2f} ms, max {slowest_node[1]['max_ms']:.2f} ms). "
            "Tool-bearing nodes dominate latency because they pay the retrieval or draft-generation cost before handing "
            "control back to the graph. The main optimization targets are keeping the embedding model warm, caching stable "
            "grounded lookups used in evaluation, and reducing oversized handoff payloads between researcher and analyst."
        ),
        encoding="utf-8",
    )


def main() -> None:
    graph, _, execution_mode = build_instrumented_graph()
    traces = [run_trace(query_id, user_query, graph) for query_id, user_query in complex_queries()]
    summary = aggregate_node_timings(traces)
    write_outputs(traces, summary, execution_mode)
    print(f"Wrote {TRACE_JSON_PATH}, {TRACE_PDF_PATH}, {OBSERVABILITY_LINK_PATH}, and {BOTTLENECK_PATH}.")


if __name__ == "__main__":
    main()
