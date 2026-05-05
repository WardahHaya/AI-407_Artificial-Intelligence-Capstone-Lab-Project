from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from ingest_data import get_collection, ingest_chunks, load_project_chunks
from schema import ChatRequest, ChatResponse
from secured_graph import build_secured_graph

CHECKPOINT_DB_PATH = Path(os.getenv("CHECKPOINT_DB_PATH", "checkpoint_db.sqlite"))


class ApiDemoModel:
    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        latest_user_message = next(
            (message.content for message in reversed(messages) if isinstance(message, HumanMessage)),
            "",
        )
        if "repeat that in one sentence" in latest_user_message.lower():
            return AIMessage(
                content=(
                    "Talent Team asked for your updated resume before May 6 and wants your interview availability."
                )
            )

        last_tool_message = next((message for message in reversed(messages) if isinstance(message, ToolMessage)), None)
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
                        "id": "api_demo_tool_call",
                        "type": "tool_call",
                    }
                ],
            )

        return AIMessage(
            content=(
                "I found the recruiter request in grounded memory. "
                "Talent Team asked for the updated resume before May 6 and requested interview availability."
            )
        )


def _graph_input(message: str) -> dict[str, object]:
    return {
        "messages": [HumanMessage(content=message)],
        "safety_status": "safe",
        "guardrail_reason": "",
        "sanitized_output": "",
    }


def _graph_config(thread_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": str(thread_id)}}


def _status_from_result(result: dict[str, Any]) -> str:
    if result.get("safety_status") == "unsafe":
        return "blocked"
    return "completed"


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
            import time

            time.sleep(3)

    raise RuntimeError(f"Grounding store startup failed after {max_attempts} attempts: {last_error}")


def create_app(model=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        chosen_model = model
        if chosen_model is None and not os.getenv("GROQ_API_KEY"):
            chosen_model = ApiDemoModel()

        _ensure_checkpoint_parent_exists()
        await asyncio.to_thread(_ensure_grounding_ready)
        async with AsyncSqliteSaver.from_conn_string(str(CHECKPOINT_DB_PATH)) as saver:
            if not hasattr(saver.conn, "is_alive"):
                saver.conn.is_alive = lambda: True
            app.state.checkpointer = saver
            app.state.graph = build_secured_graph(model=chosen_model, checkpointer=saver)
            app.state.model_mode = "demo" if chosen_model is not None else "live"
            yield

    app = FastAPI(
        title="Buraq Agent API",
        version="1.0.0",
        description="FastAPI wrapper around the LangGraph-based Buraq Gmail agent.",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "mode": app.state.model_mode}

    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest) -> ChatResponse:
        try:
            result = await app.state.graph.ainvoke(_graph_input(request.message), _graph_config(str(request.thread_id)))
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Agent execution failed: {exc}") from exc

        final_answer = str(result.get("sanitized_output") or result["messages"][-1].content)
        return ChatResponse(
            thread_id=str(request.thread_id),
            message_id=str(uuid4()),
            final_answer=final_answer,
            status=_status_from_result(result),
        )

    @app.post("/stream")
    async def stream(request: ChatRequest) -> StreamingResponse:
        async def event_stream():
            response_id = str(uuid4())
            yield f"event: metadata\ndata: {json.dumps({'thread_id': str(request.thread_id), 'message_id': response_id})}\n\n"
            try:
                async for chunk in app.state.graph.astream(
                    _graph_input(request.message),
                    _graph_config(str(request.thread_id)),
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

            state_snapshot = await app.state.graph.aget_state(_graph_config(str(request.thread_id)))
            values = state_snapshot.values or {}
            final_answer = str(values.get("sanitized_output") or values.get("messages", [])[-1].content)
            payload = {"status": "completed", "final_answer": final_answer}
            yield f"event: done\ndata: {json.dumps(payload)}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app


app = create_app()


__all__ = ["app", "create_app"]
