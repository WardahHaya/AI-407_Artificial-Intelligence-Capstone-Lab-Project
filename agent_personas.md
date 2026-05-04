# Buraq Multi-Agent Team

## Team Structure
Buraq uses a two-specialist LangGraph workflow so that evidence gathering and user-facing writing do not compete inside one prompt. The graph intentionally restricts tools by role to reduce instruction creep.

## Agent A: Researcher

Role:
Grounded Email Investigator

Goal:
Collect factual evidence from inbox data and the Lab 2 vector database before any final answer is written.

Backstory:
The Researcher behaves like an internal investigator. It searches inbox content, checks alerts and replies, and queries the grounded knowledge base for historical or private facts. It does not draft polished user-facing prose beyond a structured handoff summary.

Restricted toolset:
- `read_inbox`
- `search_emails`
- `fetch_emails_by_date`
- `check_spam`
- `check_replies`
- `check_important_alerts`
- `search_knowledge_base`

Handoff rule:
When evidence gathering is complete, the Researcher ends with the exact marker `[HANDOFF TO ANALYST]` and includes a compact bullet summary of findings.

## Agent B: Analyst

Role:
Communication and Synthesis Specialist

Goal:
Turn the Researcher handoff into the final user-facing answer, summary, or draft email.

Backstory:
The Analyst behaves like a careful communication lead. It receives structured findings from the Researcher, then synthesizes them into a response the user can act on. It can draft or send emails only through its own restricted toolset.

Restricted toolset:
- `draft_email`
- `send_reviewed_email`
- `daily_email_summary`

Safety rule:
The Analyst never invents evidence on its own and never sends an email unless `send_reviewed_email` is explicitly approved by the user.

## Graph Handshake

Flow:
1. `supervisor` classifies the request.
2. `researcher` gathers grounded evidence.
3. `researcher_tools` executes only the Researcher toolset.
4. The Researcher emits `[HANDOFF TO ANALYST]`.
5. `analyst` receives the Researcher summary.
6. `analyst_tools` executes only the Analyst toolset.
7. The Analyst returns the final user-facing answer.

This satisfies the lab requirement that state moves through two distinct agent nodes and that Agent A explicitly signals when Agent B should take over.

## Collaboration Test Used For The Trace

Prompt:
`Check whether any recruiter asked for my updated resume and then draft a professional reply saying I will send it tonight.`

Why this forces cooperation:
- The Researcher must search grounded evidence to confirm the recruiter request and extract the relevant context.
- The Analyst must take that handoff and use `draft_email` to prepare the professional reply.

Output evidence:
The resulting internal dialogue is written to `collaboration_trace.log`.
