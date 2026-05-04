from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

STANDARD_REFUSAL = (
    "I cannot help with requests that try to bypass safety rules, hide destructive instructions, "
    "or expose internal project details. I can still help with safe inbox, deadline, grounding, "
    "and draft-review tasks."
)

FORBIDDEN_RULES = {
    "persona_bypass": {
        "reason": "Persona-bypass jailbreak attempt detected.",
        "patterns": [
            r"\bpretend you are\b",
            r"\bdo anything now\b",
            r"\bdan\b",
            r"\bwithout rules\b",
            r"\bdeveloper mode\b",
            r"\bunrestricted mode\b",
        ],
    },
    "instruction_hijacking": {
        "reason": "Instruction-hijacking language detected.",
        "patterns": [
            r"\bignore (all )?(previous|prior) instructions\b",
            r"\bforget (all )?(previous|prior) instructions\b",
            r"\boverride the system prompt\b",
            r"\bdisregard your safety\b",
        ],
    },
    "payload_smuggling": {
        "reason": "Hidden or destructive payload request detected.",
        "patterns": [
            r"\bsecretly\b.*\b(delete|drop|erase|wipe|exfiltrate|leak)\b",
            r"\b(delete|drop|erase|wipe)\b.*\b(database|db|checkpoint|sqlite|vector store)\b",
            r"\b(drop table|rm -rf|truncate table|delete from)\b",
            r"\bpoem\b.*\b(delete|drop|erase)\b",
            r"\bencode\b.*\b(delete|drop|erase|leak)\b",
        ],
    },
}

OBVIOUS_OFF_TOPIC_PATTERNS = [
    r"\bweather\b",
    r"\brecipe\b",
    r"\bfootball\b",
    r"\bmovie\b",
    r"\bjoke\b",
    r"\bquiz\b",
]

DOMAIN_KEYWORDS = {
    "email",
    "emails",
    "inbox",
    "reply",
    "recruiter",
    "resume",
    "interview",
    "draft",
    "send",
    "message",
    "deadline",
    "meeting",
    "project",
    "urgent",
    "spam",
    "summary",
    "follow-up",
    "knowledge base",
    "grounded",
}

RAW_METADATA_KEYS = (
    "ID",
    "doc_type",
    "department",
    "priority_level",
    "source_file",
    "last_updated",
)


class PromptValidationInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    prompt: str = Field(description="The latest user prompt entering the secured graph.")

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Empty prompts are not allowed.")

        violation_reason = detect_policy_violation(value)
        if violation_reason:
            raise ValueError(violation_reason)

        return value


class GuardrailAssessment(BaseModel):
    status: Literal["SAFE", "UNSAFE"]
    reason: str
    matched_rule: str | None = None


def detect_policy_violation(prompt: str) -> str | None:
    normalized = prompt.lower()

    for rule_name, rule in FORBIDDEN_RULES.items():
        for pattern in rule["patterns"]:
            if re.search(pattern, normalized, flags=re.IGNORECASE | re.DOTALL):
                return f"{rule_name}: {rule['reason']}"

    if is_obviously_off_topic(normalized):
        return "off_topic: This request is outside Buraq's email and grounded-workflow scope."

    return None


def is_obviously_off_topic(normalized_prompt: str) -> bool:
    if any(keyword in normalized_prompt for keyword in DOMAIN_KEYWORDS):
        return False

    return any(re.search(pattern, normalized_prompt, flags=re.IGNORECASE) for pattern in OBVIOUS_OFF_TOPIC_PATTERNS)


def assess_prompt(prompt: str) -> GuardrailAssessment:
    try:
        PromptValidationInput(prompt=prompt)
    except ValidationError as exc:
        reason = exc.errors()[0]["msg"]
        if reason.startswith("Value error, "):
            reason = reason.replace("Value error, ", "", 1)
        matched_rule = reason.split(":", 1)[0] if ":" in reason else None
        return GuardrailAssessment(status="UNSAFE", reason=reason, matched_rule=matched_rule)

    return GuardrailAssessment(status="SAFE", reason="Prompt is safe for the agent.", matched_rule=None)


def build_standard_refusal(reason: str) -> str:
    return f"{STANDARD_REFUSAL}\nReason: {reason}"


def sanitize_output_text(text: str) -> str:
    sanitized = text

    sanitized = re.sub(
        r"(?im)^(?:ID|doc_type|department|priority_level|source_file|last_updated):[^\n]*\n?",
        "",
        sanitized,
    )
    sanitized = re.sub(r"(?im)^content:\s*\n?", "", sanitized)

    sanitized = re.sub(
        r"[A-Za-z]:\\(?:[^\\\n]+\\)*[^\\\n]+",
        "[internal path redacted]",
        sanitized,
    )
    sanitized = re.sub(
        r"(?<!\w)(?:\./|\.\./)?(?:Initial_Data|agent|vector_store)/[^\s]+",
        "[internal source redacted]",
        sanitized,
    )
    sanitized = re.sub(
        r"\b(?:checkpoint_db\.sqlite|token\.pickle|credentials\.json|\.draft_cache\.pkl)\b",
        "[internal file redacted]",
        sanitized,
    )

    sanitized = re.sub(r"\n{3,}", "\n\n", sanitized).strip()
    if not sanitized:
        return "Grounded evidence was found, but internal metadata was removed before display."
    return sanitized


__all__ = [
    "GuardrailAssessment",
    "PromptValidationInput",
    "RAW_METADATA_KEYS",
    "STANDARD_REFUSAL",
    "assess_prompt",
    "build_standard_refusal",
    "sanitize_output_text",
]
