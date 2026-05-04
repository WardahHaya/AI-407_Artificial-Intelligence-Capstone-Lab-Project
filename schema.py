from __future__ import annotations

from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    message: str = Field(min_length=1, description="User message to send to the agent.")
    thread_id: str | UUID = Field(description="Conversation thread identifier used by the LangGraph checkpointer.")


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: str = Field(description="Conversation thread identifier echoed back to the client.")
    message_id: str = Field(default_factory=lambda: str(uuid4()), description="Server-generated response identifier.")
    final_answer: str = Field(description="The final answer returned by the agent.")
    status: Literal["completed", "blocked", "error"] = Field(description="Current status of the request.")
