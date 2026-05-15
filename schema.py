from __future__ import annotations

from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class PendingActionSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    action_type: Literal["send_email", "schedule_email"]
    source_tool: str = Field(description="The originating tool or workflow that staged the outbound action.")
    to: str = Field(description="Recipient for the outbound action.")
    subject: str = Field(description="Subject line for the outbound action.")
    body: str = Field(description="Body text that will be sent or scheduled.")
    send_at: str | None = Field(default=None, description="Scheduled send timestamp when action_type is schedule_email.")
    attachment_ref: str | None = Field(default=None, description="Managed storage reference for an attachment, if any.")


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    message: str = Field(min_length=1, description="User message to send to the agent.")
    thread_id: str | UUID = Field(description="Conversation thread identifier used by the LangGraph checkpointer.")


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: str = Field(description="Conversation thread identifier echoed back to the client.")
    message_id: str = Field(default_factory=lambda: str(uuid4()), description="Server-generated response identifier.")
    final_answer: str = Field(description="The final answer returned by the agent.")
    status: Literal["completed", "blocked", "awaiting_approval", "error"] = Field(
        description="Current status of the request."
    )
    pending_action: PendingActionSummary | None = Field(
        default=None,
        description="Structured outbound action awaiting human review, when status is awaiting_approval.",
    )


class ApprovalStateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: str = Field(description="Conversation thread identifier echoed back to the client.")
    status: Literal["completed", "blocked", "awaiting_approval", "error"] = Field(
        description="Current state of the thread."
    )
    final_answer: str = Field(description="Most recent user-visible response associated with the thread.")
    pending_action: PendingActionSummary | None = Field(
        default=None,
        description="Structured outbound action awaiting human review, if one exists.",
    )
    next_nodes: list[str] = Field(default_factory=list, description="Pending graph nodes for the current checkpoint.")


class ApprovalDecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    thread_id: str | UUID = Field(description="Conversation thread identifier for the paused outbound action.")
    decision: Literal["approve", "cancel"] = Field(description="Reviewer decision for the pending outbound action.")
    edited_to: str | None = Field(default=None, description="Optional replacement recipient.")
    edited_subject: str | None = Field(default=None, description="Optional replacement subject.")
    edited_body: str | None = Field(default=None, description="Optional replacement email body.")
    edited_send_at: str | None = Field(default=None, description="Optional replacement schedule timestamp.")
    edited_attachment_ref: str | None = Field(default=None, description="Optional replacement attachment reference.")


class ManualApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    thread_id: str | UUID = Field(description="Conversation thread identifier to associate with the manual review.")
    action_type: Literal["send_email", "schedule_email"] = Field(description="Outbound action type to stage for review.")
    to: str = Field(description="Recipient for the outbound action.")
    subject: str = Field(description="Subject line for the outbound action.")
    body: str = Field(description="Body text for the outbound action.")
    send_at: str | None = Field(default=None, description="Scheduled timestamp when action_type is schedule_email.")
    attachment_ref: str | None = Field(default=None, description="Optional managed storage attachment reference.")
