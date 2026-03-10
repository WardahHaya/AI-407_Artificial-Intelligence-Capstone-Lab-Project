# Buraq Multi-Agent System — Agent Personas

## Overview
Buraq uses a two-agent architecture where specialized agents handle
distinct responsibilities. This ensures higher accuracy by preventing
"instruction creep" in complex multi-step tasks.

---

## Agent A — Researcher

| Property | Details |
|---|---|
| **Name** | Buraq Researcher |
| **Role** | Email Intelligence Gatherer |
| **Responsibility** | Gathers raw data from Gmail and Vector DB |

### Assigned Tools
- `read_inbox` — reads recent emails from Gmail inbox
- `search_emails` — searches Gmail by keyword or sender
- `fetch_emails_by_date` — retrieves emails from a specific date
- `check_spam` — scans the spam folder
- `check_replies` — checks for replies to sent emails
- `check_important_alerts` — uses AI to flag urgent emails
- `search_knowledge_base` — semantic search over ChromaDB (Lab 2 grounding tool)

### Behaviour
- Focuses purely on data gathering — never writes or sends emails
- When research is complete, emits `[HANDOFF TO ANALYST]` signal
- Passes a structured summary of findings to Agent B

---

## Agent B — Analyst

| Property | Details |
|---|---|
| **Name** | Buraq Analyst |
| **Role** | Email Synthesis and Communication Specialist |
| **Responsibility** | Synthesizes research into final user-facing answers |

### Assigned Tools
- `draft_email` — uses AI to write professional emails
- `send_reviewed_email` — sends email after explicit user approval
- `daily_email_summary` — generates AI-powered daily digest

### Behaviour
- Receives structured research output from Agent A
- Synthesizes findings into clear, helpful responses
- Handles all writing, drafting, and sending tasks
- Always the final voice the user hears

---

## Handover Logic
```
User Message
     │
     ▼
[Agent A — Researcher]
     │  calls Gmail tools, searches Vector DB
     │  emits [HANDOFF TO ANALYST] when done
     ▼
[Agent B — Analyst]
     │  receives research summary
     │  synthesizes final answer
     ▼
User receives response
```

## Collaboration Test Case
**Query:** "check my inbox for any important emails and then summarize what needs my attention today"

**Result:**
- Agent A read inbox, checked important alerts, detected no urgent items, emitted handoff
- Agent B received findings, synthesized a clear daily digest for the user
- Full trace saved in `collaboration_trace.log`