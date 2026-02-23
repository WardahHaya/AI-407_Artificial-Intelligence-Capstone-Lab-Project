# PRD — Buraq: Intelligent Email Agent

> **Course:** AI407 — Artificial Intelligence Capstone
> **Lab:** 01 — Problem Framing & Agentic Architecture
> **Industry Vertical:** Productivity / Communication Technology
> **Framework:** LangGraph + LangChain
> **Tagline:** Fly Above Your Inbox

---

## 1. Problem Statement

### The Bottleneck

Knowledge workers — students, professionals, and teams — spend an average of **2–3 hours per day** managing email. The core bottleneck is not sending or receiving emails. It is the **cognitive overhead** of managing them:

- Scanning hundreds of messages to find what is actually important
- Re-writing similar emails from scratch repeatedly
- Forgetting to follow up on sent emails that received no reply
- Missing deadlines or action items buried inside long threads
- Manually downloading and opening attachments one by one
- Losing important old emails because keyword search is too limited

No single LLM prompt can solve this. The process requires **multi-step perception, reasoning, and execution** across a live email API, a semantic knowledge base, a file system, and a scheduler — exactly the problem an agentic architecture is designed for.

### What Buraq Solves

Buraq is a **conversational Gmail agent** that connects to the user's Gmail account. The user interacts entirely in natural language through a chat interface. The agent perceives their inbox, reasons about what action is needed, executes the right tool, and responds — without the user ever opening Gmail.

---

## 2. The Agentic Boundary

### Perceive
The agent extracts data from multiple live sources:
- Gmail API — inbox, sent, spam, attachments, thread history
- ChromaDB Vector Store — semantically indexed past emails and documents
- Firebase Firestore — user profiles, chat history, scheduled email queue
- Firebase Storage — uploaded and received file attachments

### Reason
The agent uses a LangGraph ReAct (Reason + Act) loop to:
- Classify user intent from natural language input
- Decide which tool(s) to invoke and in what sequence
- Evaluate tool results and determine the next step
- Loop until the task is fully complete or ask the user for clarification

### Execute
The agent calls Python functions that interact with the real world:
- Read, search, draft, send, and schedule emails via Gmail API
- Ingest new emails into the Vector Store for semantic retrieval
- Upload files to Firebase Storage and attach them to outgoing emails
- Run a background scheduler that delivers emails at user-specified times
- Generate AI-written summaries, digests, and intelligent alerts

---

## 3. User Personas

### Primary — The Overwhelmed Student (Hamza, 22)
- **Context:** Final year BS AI student managing emails from professors, group members, and internship recruiters on Gmail
- **Pain:** Misses deadlines buried in emails, spends 45 minutes daily just reading and responding
- **Goal:** Ask "anything urgent today?" and get a clear prioritised answer in seconds
- **Expected Behaviour:** Agent reads inbox, flags deadlines, drafts replies for review, sends on approval

### Secondary — The Early-Career Professional (Aisha, 26)
- **Context:** Works at a startup, Gmail is her primary communication channel
- **Pain:** Forgets to follow up, spends significant time writing professional emails from scratch
- **Goal:** Type "write a follow-up to the client about the proposal" and have it done
- **Expected Behaviour:** Agent drafts email, shows for review, sends only after explicit approval

### Tertiary — The Small Business Owner (Usman, 38)
- **Context:** Receives 100+ emails per day, needs oversight without a dedicated assistant
- **Pain:** Important client emails get buried under newsletters and spam
- **Goal:** Receive a morning digest — what is important, what needs reply, what can be ignored
- **Expected Behaviour:** Agent generates prioritised daily summary with clear action items

---

## 4. Tool & Data Inventory

### Knowledge Sources

| Source | Purpose |
|---|---|
| ChromaDB Vector Store | Semantic search over indexed past emails — finds emails by meaning not just keyword |
| Firebase Firestore | User data, OAuth tokens, chat history, scheduled email queue |
| Firebase Storage | Files the user wants to send as attachments; received attachment storage |
| Gmail Thread History | Retrieved per-conversation to maintain context for drafting replies |
| User Writing Style Profile | Inferred from sent emails, stored in Firestore, used to personalise AI drafts |

### Action Tools

| Tool | Description |
|---|---|
| `read_inbox()` | Fetch and format recent emails from Gmail inbox |
| `search_emails(query)` | Search by keyword, sender, subject, or date range |
| `fetch_emails_by_date(date)` | Return all emails received on a specific date |
| `draft_email(to, subject, context, tone)` | AI writes email body, returns draft for review — never sends automatically |
| `send_reviewed_email(confirmed)` | Sends cached draft ONLY after explicit user approval |
| `send_email_with_attachment(to, subject, body, file_url)` | Sends email with file attached from Firebase Storage |
| `schedule_email(to, subject, body, send_at)` | Queues email in Firestore for future delivery by scheduler daemon |
| `download_attachments()` | Downloads received files, stores in Firebase Storage |
| `daily_email_summary(date)` | AI-generated digest of all emails for a given day |
| `check_spam()` | Reports suspicious emails from the spam folder |
| `check_replies(hours_back)` | Detects if anyone has replied to the user's sent emails |
| `check_important_alerts()` | Scans inbox and flags urgent items and upcoming deadlines |
| `search_knowledge_base(query)` | Semantic search over ChromaDB Vector Store of past emails |
| `ingest_emails_to_vector_store()` | Embeds and indexes new emails into ChromaDB for future retrieval |
| `upload_file_to_storage(file)` | Uploads a local file to Firebase Storage for use as email attachment |

---

## 5. System Architecture (LangGraph Focus)

```
+------------------------------------------------------------+
|                    STREAMLIT CHAT UI                       |
|     chat input | file upload | attachment viewer           |
|     chat history panel | Streamlit Community Cloud        |
+---------------------------+--------------------------------+
                            |
                            v
+------------------------------------------------------------+
|                    FASTAPI BACKEND                         |
|     Google OAuth callback | file routing                  |
|     scheduler daemon | session management | Render        |
+---------------------------+--------------------------------+
                            |
                            v
+------------------------------------------------------------+
|                 LANGGRAPH ReAct AGENT                      |
|                                                            |
|   [START]                                                  |
|      |                                                     |
|      v                                                     |
|  [agent_node] <----------------------------------+         |
|      |  LLM reasons over full state             |         |
|      v                                           |         |
|  [router]                                        |         |
|      |                                           |         |
|      +-- tool_calls? YES --> [tool_node] ---------+        |
|      |                                                     |
|      +-- final answer? ---> [END]                         |
|                                                            |
|  State: {messages, user_id, draft_cache, tool_results}   |
+------+-------------------------+---------------------------+
       |                         |
+------v---------+   +-----------v-----------------------------+
|  GMAIL API     |   |        KNOWLEDGE & DATA                 |
|                |   |                                         |
|  Read / Send   |   |  ChromaDB  -- semantic email search     |
|  Search        |   |  Firestore -- users, tokens, queue      |
|  Attachments   |   |  Firebase Storage -- files              |
|  Google OAuth2 |   |  Groq LLM  -- reasoning & drafting      |
+----------------+   +-----------------------------------------+
```

---

## 6. Success Metrics

| Metric | Target |
|---|---|
| Agent response time | Under 5 seconds for read and search operations |
| Draft approval rate | User accepts AI draft without revision in 70% of cases |
| Semantic search accuracy | Correct email in top 3 results for 90% of test queries |
| Scheduler delivery accuracy | 100% of emails sent within 1 minute of scheduled time |
| Daily digest coverage | Zero missed important emails reported in user testing |
| Autonomous resolution rate | Agent handles 85% of requests without asking for clarification |

---

## 7. Tech Stack

| Component | Technology |
|---|---|
| Agent Orchestration | LangGraph |
| Tool Framework | LangChain + Pydantic |
| LLM | Groq API (llama3-70b-8192) |
| Vector Store | ChromaDB + sentence-transformers |
| Email Provider | Gmail API (Google Cloud Console) |
| Auth | Firebase Auth + Google OAuth2 |
| Database | Firebase Firestore |
| File Storage | Firebase Storage |
| Backend | FastAPI + Python |
| Frontend | Streamlit |
| Deployment | Streamlit Community Cloud + Render |
| IDE | VS Code |

---

## 8. Why This Requires an Agent

A single LLM prompt cannot solve this because:

1. Reading an inbox requires a **live Gmail API call** — LLM has no real-time email access
2. Draft and send are **two steps separated by human review** — requiring stateful memory between turns
3. Finding a past email by meaning requires a **Vector Store** — keyword search alone fails
4. Scheduling requires a **background daemon** running independently of the conversation
5. The full workflow (read → understand → draft → review → send) requires **4+ sequential tool calls**

This is precisely the class of problem LangGraph's stateful ReAct loop is designed for.

---

*Prepared for AI407 Capstone | Wardah Haya*
