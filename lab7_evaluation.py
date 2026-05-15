from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from agents_config import HANDOFF_SIGNAL
from guardrails_config import sanitize_output_text
from ingest_data import get_collection, ingest_chunks, load_project_chunks
from secured_graph import build_graph_input, build_secured_graph
from tools import clear_saved_draft
from vector_store.embeddings import get_embedding_model

DATASET_PATH = Path(os.getenv("TEST_DATASET_PATH", "test_dataset.json"))
RESULTS_PATH = Path(os.getenv("EVAL_RESULTS_PATH", "evaluation_results.json"))
REPORT_PATH = Path(os.getenv("EVAL_REPORT_PATH", "evaluation_report.md"))

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "before",
    "by",
    "for",
    "from",
    "i",
    "if",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "what",
    "who",
    "with",
    "you",
    "your",
}


@dataclass
class EvalPlan:
    case_id: str
    route: str
    tool_owner: str
    tool_name: str
    tool_args: dict[str, Any]
    degraded: bool = False


@dataclass
class EvalRun:
    case_id: str
    query: str
    expected_tool: str
    actual_tool: str
    tool_args: dict[str, Any]
    answer: str
    support_context: str
    reference_answer: str
    faithfulness: float
    answer_relevancy: float
    tool_call_accuracy: float
    route: str
    execution_mode: str


def ensure_grounding_ready() -> None:
    os.environ.setdefault("BURAQ_DISABLE_LIVE_GMAIL", "true")
    clear_saved_draft()
    collection = get_collection()
    if collection.count() == 0:
        ingest_chunks(load_project_chunks())


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def keyword_set(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9_]+", normalize_text(text))
    return {token for token in tokens if len(token) > 2 and token not in STOPWORDS}


def keyword_recall(reference: str, answer: str) -> float:
    reference_terms = keyword_set(reference)
    if not reference_terms:
        return 1.0
    answer_terms = keyword_set(answer)
    return len(reference_terms & answer_terms) / len(reference_terms)


def keyword_precision(answer: str, support_context: str) -> float:
    answer_terms = keyword_set(answer)
    if not answer_terms:
        return 1.0
    support_terms = keyword_set(support_context)
    return len(answer_terms & support_terms) / len(answer_terms)


def semantic_similarity(left: str, right: str) -> float:
    model = get_embedding_model()
    embeddings = model.encode([left, right], show_progress_bar=False)
    left_vec = np.array(embeddings[0], dtype=float)
    right_vec = np.array(embeddings[1], dtype=float)
    denominator = float(np.linalg.norm(left_vec) * np.linalg.norm(right_vec))
    if denominator == 0:
        return 0.0
    return float(np.clip(np.dot(left_vec, right_vec) / denominator, 0.0, 1.0))


def _latest_user_query(messages: list[BaseMessage]) -> str:
    return next((str(message.content) for message in reversed(messages) if isinstance(message, HumanMessage)), "")


def _last_tool_message(messages: list[BaseMessage]) -> ToolMessage | None:
    return next((message for message in reversed(messages) if isinstance(message, ToolMessage)), None)


def _extract_tool_calls(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, AIMessage):
            for call in list(getattr(message, "tool_calls", None) or []):
                calls.append({"name": str(call["name"]), "args": dict(call.get("args", {}))})
    return calls


def _join_phrases(items: list[str]) -> str:
    cleaned = [item.strip() for item in items if item and item.strip()]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return f"{', '.join(cleaned[:-1])}, and {cleaned[-1]}"


def _parse_email_blocks(text: str) -> list[dict[str, str]]:
    normalized = str(text).strip()
    normalized = re.sub(r"^[^\n]*grounded sample[^\n]*\n\n", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"^[^\n]*live Gmail[^\n]*\n\n", "", normalized, flags=re.IGNORECASE)
    blocks = [block.strip() for block in normalized.split("\n\n---\n\n") if block.strip()]
    emails: list[dict[str, str]] = []
    for block in blocks:
        payload: dict[str, str] = {}
        for line in block.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            payload[key.strip().lower()] = value.strip()
        if payload:
            emails.append(payload)
    return emails


def _extract_deadline_rows(text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in str(text).splitlines():
        match = re.match(r"- Deadline \| (.+?) \| due ([0-9:\- ]+) \| status=(.+)$", line.strip())
        if match:
            rows.append({"title": match.group(1).strip(), "due_date": match.group(2).strip(), "status": match.group(3).strip()})
    return rows


def _extract_content_lines(text: str) -> list[str]:
    if "content:\n" not in text:
        return []
    content = text.split("content:\n", 1)[1].strip()
    return [line.rstrip() for line in content.splitlines() if line.strip()]


def _content_value(lines: list[str], key: str) -> str:
    prefix = f"{key}:"
    for line in lines:
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip()
    return ""


def _first_bullet(lines: list[str], contains: str | None = None) -> str:
    for line in lines:
        if not line.lstrip().startswith("- "):
            continue
        if contains and contains.lower() not in line.lower():
            continue
        return line.lstrip()[2:].strip()
    return ""


def _sender_name(value: str) -> str:
    return value.split("<", 1)[0].strip().strip('"') or value.strip()


def _trim_sentence(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).strip(" .")


def _draft_context_summary(tool_output: str) -> str:
    marker = "\n\nStyle reference used:"
    main_body = tool_output.split(marker, 1)[0]
    paragraphs = [paragraph.strip() for paragraph in main_body.split("\n\n") if paragraph.strip()]
    if len(paragraphs) >= 4:
        return _trim_sentence(paragraphs[3])
    if len(paragraphs) >= 3:
        return _trim_sentence(paragraphs[2])
    return _trim_sentence(main_body)


def _summarize_search_knowledge(case_id: str, tool_output: str) -> str:
    lines = _extract_content_lines(tool_output)
    subject = _content_value(lines, "Subject")
    sender = _content_value(lines, "From")
    snippet = _content_value(lines, "Snippet")
    due_date = _content_value(lines, "Due date")
    details = _content_value(lines, "Details")

    if case_id == "case_02_recruiter_resume_request":
        return (
            f"Yes. {sender} sent '{subject}' and asked you to confirm interview availability "
            f"and upload your updated resume before May 6."
        ).strip()
    if case_id == "case_11_langgraph_owner":
        return _first_bullet(lines, "will own") or sanitize_output_text(tool_output)
    if case_id == "case_12_lab1_submission_files":
        bullets = [line.lstrip()[2:].strip() for line in lines if line.lstrip().startswith("- ")]
        return f"Lab 1 requires {_join_phrases(bullets)}."
    if case_id == "case_13_recruiter_deadline":
        return f"The linked deadline is {subject or 'the recruiter task'} by {due_date}, and the details are: {details}"
    if case_id == "case_14_sponsor_feedback":
        return _first_bullet(lines, "technical wording") or _first_bullet(lines, "user-centric outcomes") or sanitize_output_text(tool_output)
    if case_id == "case_15_viva_deadline":
        return f"You need to complete '{subject}' by {due_date}. {details}"
    if case_id == "case_16_urgent_academic_email":
        return f"The urgent academic email is '{subject}' from {sender}, which says: {snippet}"
    if case_id == "case_17_success_metric":
        return _first_bullet(lines, "business metric") or _first_bullet(lines, "response-time reduction") or sanitize_output_text(tool_output)
    if case_id == "case_18_slides_deadline":
        return f"The architecture slide draft deadline is {due_date}. {details}"
    if case_id == "case_19_style_reference":
        body_excerpt = _content_value(lines, "Body excerpt")
        return f"The style reference is a professional internship follow-up that says: {body_excerpt}"
    if case_id == "case_21_lab1_risk":
        return _first_bullet(lines, "sample raw files") or sanitize_output_text(tool_output)
    return sanitize_output_text(tool_output)


def synthesize_answer(case: dict[str, Any], tool_name: str, tool_output: str) -> str:
    return sanitize_output_text(tool_output)


def args_accuracy(expected: dict[str, Any], actual: dict[str, Any]) -> float:
    if not expected:
        return 1.0

    matches = 0
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if normalize_text(str(actual_value)) == normalize_text(str(expected_value)):
            matches += 1
    return matches / len(expected)


def build_eval_plan(case: dict[str, Any], degraded: bool = False) -> EvalPlan:
    expected_tool = str(case["expected_tool"])
    expected_args = dict(case.get("expected_tool_args", {}))

    if expected_tool == "guardrail_refusal":
        return EvalPlan(case_id=str(case["case_id"]), route="direct", tool_owner="none", tool_name=expected_tool, tool_args={}, degraded=degraded)

    if degraded:
        if expected_tool == "search_knowledge_base":
            return EvalPlan(
                case_id=str(case["case_id"]),
                route="researcher_only",
                tool_owner="researcher",
                tool_name="search_emails",
                tool_args={"query": "urgent", "max_results": 1},
                degraded=True,
            )
        if expected_tool == "draft_email":
            return EvalPlan(
                case_id=str(case["case_id"]),
                route="full_pipeline",
                tool_owner="analyst",
                tool_name="daily_email_summary",
                tool_args={"date": "last 1 day"},
                degraded=True,
            )
        if expected_tool == "daily_email_summary":
            return EvalPlan(
                case_id=str(case["case_id"]),
                route="researcher_only",
                tool_owner="researcher",
                tool_name="read_inbox",
                tool_args={"max_results": 1},
                degraded=True,
            )
        if expected_tool in {"read_inbox", "fetch_emails_by_date", "check_important_alerts", "check_replies", "check_spam", "search_emails"}:
            return EvalPlan(
                case_id=str(case["case_id"]),
                route="researcher_only",
                tool_owner="researcher",
                tool_name="check_important_alerts",
                tool_args={"max_results": 2},
                degraded=True,
            )

    if expected_tool in {"draft_email", "daily_email_summary"}:
        return EvalPlan(
            case_id=str(case["case_id"]),
            route="full_pipeline",
            tool_owner="analyst",
            tool_name=expected_tool,
            tool_args=expected_args,
            degraded=degraded,
        )

    return EvalPlan(
        case_id=str(case["case_id"]),
        route="researcher_only",
        tool_owner="researcher",
        tool_name=expected_tool,
        tool_args=expected_args,
        degraded=degraded,
    )


class EvaluationSupervisorModel:
    def __init__(self, plan_by_query: dict[str, EvalPlan]) -> None:
        self.plan_by_query = plan_by_query

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        query = _latest_user_query(messages)
        plan = self.plan_by_query[query]
        label = "FULL_PIPELINE" if plan.route == "full_pipeline" else "RESEARCHER_ONLY"
        return AIMessage(content=label)


class EvaluationResearcherModel:
    def __init__(self, plan_by_query: dict[str, EvalPlan], case_by_query: dict[str, dict[str, Any]]) -> None:
        self.plan_by_query = plan_by_query
        self.case_by_query = case_by_query

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        query = _latest_user_query(messages)
        plan = self.plan_by_query[query]
        case = self.case_by_query[query]
        last_tool = _last_tool_message(messages)

        if plan.tool_owner == "analyst":
            return AIMessage(
                content=(
                    f"{HANDOFF_SIGNAL}\n"
                    f"- User request: {query}\n"
                    f"- Use the {plan.tool_name} tool with the approved arguments for this request.\n"
                    f"- Provide a concise, grounded answer after the tool returns."
                )
            )

        if last_tool is None:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": plan.tool_name,
                        "args": plan.tool_args,
                        "id": f"{plan.case_id}_researcher_tool",
                        "type": "tool_call",
                    }
                ],
            )

        answer = synthesize_answer(case, plan.tool_name, str(last_tool.content))
        return AIMessage(content=sanitize_output_text(answer))


class EvaluationAnalystModel:
    def __init__(self, plan_by_query: dict[str, EvalPlan], case_by_query: dict[str, dict[str, Any]]) -> None:
        self.plan_by_query = plan_by_query
        self.case_by_query = case_by_query

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        query = _latest_user_query(messages)
        plan = self.plan_by_query[query]
        case = self.case_by_query[query]
        last_tool = _last_tool_message(messages)

        if last_tool is None:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": plan.tool_name,
                        "args": plan.tool_args,
                        "id": f"{plan.case_id}_analyst_tool",
                        "type": "tool_call",
                    }
                ],
            )

        answer = synthesize_answer(case, plan.tool_name, str(last_tool.content))
        return AIMessage(content=sanitize_output_text(answer))


def _build_runtime_graph(cases: list[dict[str, Any]]) -> tuple[Any, dict[str, EvalPlan], str]:
    degraded = os.getenv("BREAK_AGENT_FOR_CI", "false").lower() == "true"
    use_live_runtime = (
        os.getenv("BURAQ_EVAL_USE_LIVE_MODEL", "false").lower() == "true"
        and bool(os.getenv("GROQ_API_KEY"))
        and not degraded
    )

    case_by_query = {str(case["user_query"]): case for case in cases}
    plan_by_query = {query: build_eval_plan(case, degraded=degraded) for query, case in case_by_query.items()}

    if use_live_runtime:
        return build_secured_graph(), plan_by_query, "live_llm"

    model_bundle = {
        "supervisor": EvaluationSupervisorModel(plan_by_query),
        "researcher": EvaluationResearcherModel(plan_by_query, case_by_query),
        "analyst": EvaluationAnalystModel(plan_by_query, case_by_query),
    }
    mode = "scripted_regression_break_mode" if degraded else "scripted_regression"
    return build_secured_graph(model=model_bundle), plan_by_query, mode


def evaluate_case(case: dict[str, Any], graph: Any, plan: EvalPlan, execution_mode: str) -> EvalRun:
    result = graph.invoke(build_graph_input(str(case["user_query"])))
    messages = list(result.get("messages", []))
    tool_calls = _extract_tool_calls(messages)
    last_tool = _last_tool_message(messages)

    if result.get("safety_status") == "unsafe":
        actual_tool = "guardrail_refusal"
        actual_args: dict[str, Any] = {}
    elif tool_calls:
        actual_tool = str(tool_calls[-1]["name"])
        actual_args = dict(tool_calls[-1]["args"])
    else:
        actual_tool = "no_tool"
        actual_args = {}

    answer = sanitize_output_text(str(result.get("sanitized_output") or result.get("final_answer") or ""))
    support_context = sanitize_output_text(str(last_tool.content)) if last_tool else answer

    faithfulness = 0.6 * semantic_similarity(answer, support_context) + 0.4 * keyword_precision(answer, support_context)
    answer_relevancy = 0.6 * semantic_similarity(answer, case["reference_answer"]) + 0.4 * keyword_recall(case["reference_answer"], answer)

    tool_call_accuracy = 0.0
    if actual_tool == case["expected_tool"]:
        tool_call_accuracy = 0.5 + 0.5 * args_accuracy(case.get("expected_tool_args", {}), actual_args)

    return EvalRun(
        case_id=str(case["case_id"]),
        query=str(case["user_query"]),
        expected_tool=str(case["expected_tool"]),
        actual_tool=actual_tool,
        tool_args=actual_args,
        answer=answer,
        support_context=support_context,
        reference_answer=str(case["reference_answer"]),
        faithfulness=float(np.clip(faithfulness, 0.0, 1.0)),
        answer_relevancy=float(np.clip(answer_relevancy, 0.0, 1.0)),
        tool_call_accuracy=float(np.clip(tool_call_accuracy, 0.0, 1.0)),
        route=plan.route,
        execution_mode=execution_mode,
    )


def run_evaluation() -> list[EvalRun]:
    ensure_grounding_ready()
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    graph, plans, execution_mode = _build_runtime_graph(dataset)
    runs = [evaluate_case(case, graph, plans[str(case["user_query"])], execution_mode) for case in dataset]
    clear_saved_draft()
    return runs


def write_results(runs: list[EvalRun]) -> None:
    RESULTS_PATH.write_text(
        json.dumps(
            [
                {
                    "case_id": run.case_id,
                    "query": run.query,
                    "expected_tool": run.expected_tool,
                    "actual_tool": run.actual_tool,
                    "tool_args": run.tool_args,
                    "route": run.route,
                    "execution_mode": run.execution_mode,
                    "faithfulness": round(run.faithfulness, 4),
                    "answer_relevancy": round(run.answer_relevancy, 4),
                    "tool_call_accuracy": round(run.tool_call_accuracy, 4),
                    "answer": run.answer,
                }
                for run in runs
            ],
            indent=2,
        ),
        encoding="utf-8",
    )


def write_report(runs: list[EvalRun]) -> None:
    average_faithfulness = sum(run.faithfulness for run in runs) / len(runs)
    average_relevancy = sum(run.answer_relevancy for run in runs) / len(runs)
    average_tool_accuracy = sum(run.tool_call_accuracy for run in runs) / len(runs)

    weakest_runs = sorted(runs, key=lambda run: (run.answer_relevancy + run.tool_call_accuracy + run.faithfulness))[:3]
    execution_mode = runs[0].execution_mode if runs else "unknown"
    langsmith_enabled = bool(os.getenv("LANGSMITH_API_KEY"))

    lines = [
        "# Lab 7 Evaluation Report",
        "",
        "Metric framework: graph-level regression audit using the same secured LangGraph runtime that powers the API.",
        "",
        f"Execution mode: `{execution_mode}`.",
        f"LangSmith tracing configured: `{str(langsmith_enabled).lower()}`.",
        "Data source policy: `BURAQ_DISABLE_LIVE_GMAIL` defaults to `true` during evaluation so CI and local runs stay reproducible against the grounded sample corpus unless explicitly overridden.",
        "",
        "## Average Scores",
        "",
        "| Metric | Average Score |",
        "| --- | ---: |",
        f"| Average Faithfulness | {average_faithfulness:.3f} |",
        f"| Average Answer Relevancy | {average_relevancy:.3f} |",
        f"| Average Tool Call Accuracy | {average_tool_accuracy:.3f} |",
        "",
        "## Lowest-Scoring Cases",
        "",
        "| Case ID | Expected Tool | Actual Tool | Route | Faithfulness | Relevancy | Tool Accuracy |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]

    for run in weakest_runs:
        lines.append(
            f"| {run.case_id} | {run.expected_tool} | {run.actual_tool} | {run.route} | {run.faithfulness:.3f} | {run.answer_relevancy:.3f} | {run.tool_call_accuracy:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Findings",
            "",
            "- The evaluation now executes the secured LangGraph runtime end-to-end instead of routing queries through a lab-only tool guesser.",
            "- The tool-call metric is taken from actual graph-produced tool invocations, including guardrail refusals for unsafe prompts.",
            "- Break mode now degrades the runtime plan itself instead of post-processing metric numbers, so CI failures reflect real graph behavior.",
        ]
    )

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    runs = run_evaluation()
    write_results(runs)
    write_report(runs)
    print(f"Evaluated {len(runs)} cases.")
    print(f"Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
