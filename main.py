from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from ingest_data import get_collection, ingest_chunks, load_project_chunks
from multi_agent_graph import ScriptedAnalystModel, ScriptedResearcherModel, ScriptedSupervisorModel
from runtime_services import (
    ensure_runtime_dirs,
    init_schedule_db,
    list_scheduled_emails,
    list_stored_files,
    save_uploaded_bytes,
    start_scheduler_thread,
    stop_scheduler_thread,
)
from schema import ApprovalDecisionRequest, ApprovalStateResponse, ChatRequest, ChatResponse, ManualApprovalRequest
from secured_graph import (
    apply_approval_decision_sync,
    build_graph_input,
    build_secured_graph,
    get_graph_config,
    inspect_thread_state_sync,
    snapshot_to_response,
    stage_manual_action_sync,
)
from tools import deliver_email_message

CHECKPOINT_DB_PATH = Path(os.getenv("CHECKPOINT_DB_PATH", "checkpoint_db.sqlite"))


class UnavailableRuntimeModel:
    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        raise RuntimeError(
            "No live model is configured for the API. Set GROQ_API_KEY, or explicitly opt into the limited "
            "demo bundle with BURAQ_ENABLE_DEMO_MODEL=true."
        )


class DirectDemoModel:
    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        latest_user_message = next(
            (message.content for message in reversed(messages) if isinstance(message, HumanMessage)),
            "",
        )
        return AIMessage(
            content=(
                "Demo mode is enabled. The deterministic bundle can walk the recruiter grounding flow, but broader "
                f"requests still need GROQ_API_KEY.\n\nLatest request: {latest_user_message}"
            )
        )


class RateLimitExceeded(RuntimeError):
    def __init__(self, retry_after: int) -> None:
        super().__init__("Rate limit exceeded.")
        self.retry_after = retry_after


class InMemoryRateLimiter:
    def __init__(self, limit: int, window_seconds: int) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        now = time.monotonic()
        with self._lock:
            queue = self._events[key]
            while queue and now - queue[0] >= self.window_seconds:
                queue.popleft()
            if len(queue) >= self.limit:
                retry_after = max(1, int(self.window_seconds - (now - queue[0])))
                raise RateLimitExceeded(retry_after=retry_after)
            queue.append(now)


def _resolve_runtime_model_bundle() -> tuple[dict[str, Any] | Any | None, str]:
    if os.getenv("GROQ_API_KEY"):
        return None, "live"

    if os.getenv("BURAQ_ENABLE_DEMO_MODEL", "false").lower() == "true":
        return (
            {
                "supervisor": ScriptedSupervisorModel(),
                "direct": DirectDemoModel(),
                "researcher": ScriptedResearcherModel(),
                "analyst": ScriptedAnalystModel(),
            },
            "demo",
        )

    return UnavailableRuntimeModel(), "unavailable"


def _graph_config(thread_id: str) -> dict[str, dict[str, str]]:
    return get_graph_config(thread_id)


def _chunk_to_text(chunk: Any) -> str:
    if isinstance(chunk, dict):
        parts: list[str] = []
        for value in chunk.values():
            if isinstance(value, dict) and value.get("messages"):
                message = value["messages"][-1]
                content = getattr(message, "content", "")
                if content:
                    parts.append(str(content))
            elif hasattr(value, "content"):
                content = getattr(value, "content", "")
                if content:
                    parts.append(str(content))
        return "\n".join(part for part in parts if part).strip()
    return str(chunk).strip()


def _ensure_checkpoint_parent_exists() -> None:
    CHECKPOINT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def _ensure_grounding_ready(max_attempts: int = 10) -> None:
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            collection = get_collection()
            if collection.count() == 0:
                ingest_chunks(load_project_chunks())
            return
        except Exception as exc:
            last_error = exc
            if attempt == max_attempts:
                break
            time.sleep(3)

    raise RuntimeError(f"Grounding store startup failed after {max_attempts} attempts: {last_error}")


def _auth_required() -> bool:
    return bool(os.getenv("BURAQ_API_KEY", "").strip())


def _get_client_key(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    if forwarded_for:
        return forwarded_for
    if request.client and request.client.host:
        return request.client.host
    return "unknown-client"


def create_app(model=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        chosen_model, mode = _resolve_runtime_model_bundle() if model is None else (model, "custom")

        _ensure_checkpoint_parent_exists()
        ensure_runtime_dirs()
        init_schedule_db()
        await asyncio.to_thread(_ensure_grounding_ready)
        async with AsyncSqliteSaver.from_conn_string(str(CHECKPOINT_DB_PATH)) as saver:
            if not hasattr(saver.conn, "is_alive"):
                saver.conn.is_alive = lambda: True
            app.state.checkpointer = saver
            app.state.graph = build_secured_graph(model=chosen_model, checkpointer=saver)
            app.state.model_mode = mode
            app.state.api_key = os.getenv("BURAQ_API_KEY", "").strip()
            app.state.rate_limiter = InMemoryRateLimiter(
                limit=int(os.getenv("BURAQ_RATE_LIMIT_COUNT", "60")),
                window_seconds=int(os.getenv("BURAQ_RATE_LIMIT_WINDOW_SECONDS", "60")),
            )
            stop_event, thread = start_scheduler_thread(deliver_email_message)
            app.state.scheduler_stop_event = stop_event
            app.state.scheduler_thread = thread
            yield
            stop_scheduler_thread(app.state.scheduler_stop_event, app.state.scheduler_thread)

    app = FastAPI(
        title="Buraq Agent API",
        version="2.0.0",
        description="FastAPI wrapper around the secured multi-agent LangGraph runtime with checkpointed HITL approval.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def security_middleware(request: Request, call_next):
        if request.url.path != "/health":
            api_key = app.state.api_key
            if api_key:
                provided = request.headers.get("x-api-key", "").strip()
                if provided != api_key:
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "Missing or invalid X-API-Key header."},
                    )

            try:
                app.state.rate_limiter.check(_get_client_key(request))
            except RateLimitExceeded as exc:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded."},
                    headers={"Retry-After": str(exc.retry_after)},
                )

        return await call_next(request)

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {
            "status": "ok",
            "mode": app.state.model_mode,
            "auth_required": _auth_required(),
            "rate_limit_count": app.state.rate_limiter.limit,
            "rate_limit_window_seconds": app.state.rate_limiter.window_seconds,
        }

    @app.get("/uploads")
    async def uploads() -> dict[str, object]:
        return {"files": list_stored_files()}

    @app.get("/scheduled")
    async def scheduled() -> dict[str, object]:
        return {"items": list_scheduled_emails()}

    @app.post("/upload")
    async def upload(file: UploadFile = File(...)) -> dict[str, object]:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        stored = save_uploaded_bytes(file.filename or "upload.bin", content, area="uploads")
        return stored

    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest) -> ChatResponse:
        config = _graph_config(str(request.thread_id))
        try:
            existing_snapshot = await app.state.graph.aget_state(config)
            existing_state = snapshot_to_response(existing_snapshot)
            if existing_state["status"] == "awaiting_approval":
                response_state = existing_state
            else:
                await app.state.graph.ainvoke(build_graph_input(request.message), config)
                snapshot = await app.state.graph.aget_state(config)
                response_state = snapshot_to_response(snapshot)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Agent execution failed: {exc}") from exc

        return ChatResponse(
            thread_id=str(request.thread_id),
            message_id=str(uuid4()),
            final_answer=str(response_state["final_answer"]),
            status=str(response_state["status"]),
            pending_action=response_state.get("pending_action"),
        )

    @app.post("/stream")
    async def stream(request: ChatRequest) -> StreamingResponse:
        async def event_stream():
            response_id = str(uuid4())
            config = _graph_config(str(request.thread_id))
            yield f"event: metadata\ndata: {json.dumps({'thread_id': str(request.thread_id), 'message_id': response_id})}\n\n"
            try:
                existing_snapshot = await app.state.graph.aget_state(config)
                existing_state = snapshot_to_response(existing_snapshot)
                if existing_state["status"] != "awaiting_approval":
                    async for chunk in app.state.graph.astream(
                        build_graph_input(request.message),
                        config,
                        stream_mode="updates",
                    ):
                        text = _chunk_to_text(chunk)
                        if not text:
                            continue
                        payload = {"status": "streaming", "chunk": text}
                        yield f"event: chunk\ndata: {json.dumps(payload)}\n\n"
            except RuntimeError as exc:
                payload = {"status": "error", "detail": str(exc)}
                yield f"event: error\ndata: {json.dumps(payload)}\n\n"
                return
            except Exception as exc:
                payload = {"status": "error", "detail": f"Agent stream failed: {exc}"}
                yield f"event: error\ndata: {json.dumps(payload)}\n\n"
                return

            snapshot = await app.state.graph.aget_state(config)
            response_state = snapshot_to_response(snapshot)
            payload = {
                "status": response_state["status"],
                "final_answer": response_state["final_answer"],
                "pending_action": response_state.get("pending_action"),
            }
            yield f"event: done\ndata: {json.dumps(payload)}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.get("/approval/{thread_id}", response_model=ApprovalStateResponse)
    async def approval_state(thread_id: str) -> ApprovalStateResponse:
        try:
            state = await asyncio.to_thread(inspect_thread_state_sync, CHECKPOINT_DB_PATH, thread_id)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Could not inspect approval state: {exc}") from exc

        return ApprovalStateResponse(
            thread_id=str(thread_id),
            status=str(state["status"]),
            final_answer=str(state["final_answer"]),
            pending_action=state.get("pending_action"),
            next_nodes=list(state.get("next_nodes", [])),
        )

    @app.post("/approval/decision", response_model=ApprovalStateResponse)
    async def approval_decision(request: ApprovalDecisionRequest) -> ApprovalStateResponse:
        edited_fields = {
            "to": request.edited_to,
            "subject": request.edited_subject,
            "body": request.edited_body,
            "send_at": request.edited_send_at,
            "attachment_ref": request.edited_attachment_ref,
        }
        try:
            state = await asyncio.to_thread(
                apply_approval_decision_sync,
                CHECKPOINT_DB_PATH,
                str(request.thread_id),
                request.decision,
                edited_fields,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Could not apply approval decision: {exc}") from exc

        return ApprovalStateResponse(
            thread_id=str(request.thread_id),
            status=str(state["status"]),
            final_answer=str(state["final_answer"]),
            pending_action=state.get("pending_action"),
            next_nodes=list(state.get("next_nodes", [])),
        )

    @app.post("/approval/manual", response_model=ApprovalStateResponse)
    async def manual_review(request: ManualApprovalRequest) -> ApprovalStateResponse:
        pending_action = {
            "action_type": request.action_type,
            "source_tool": "manual_compose",
            "to": request.to,
            "subject": request.subject,
            "body": request.body,
        }
        if request.send_at:
            pending_action["send_at"] = request.send_at
        if request.attachment_ref:
            pending_action["attachment_ref"] = request.attachment_ref

        try:
            state = await asyncio.to_thread(
                stage_manual_action_sync,
                CHECKPOINT_DB_PATH,
                str(request.thread_id),
                pending_action,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Could not stage manual approval: {exc}") from exc

        return ApprovalStateResponse(
            thread_id=str(request.thread_id),
            status=str(state["status"]),
            final_answer=str(state["final_answer"]),
            pending_action=state.get("pending_action"),
            next_nodes=list(state.get("next_nodes", [])),
        )

    return app


app = create_app()


__all__ = ["app", "create_app"]
