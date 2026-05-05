# Buraq - Intelligent Gmail Agent
### Fly Above Your Inbox

**AI407 Artificial Intelligence Capstone | Wardah Haya**

---

## What is Buraq?
Buraq is an AI-powered Gmail agent. Users interact with their email
entirely through natural language. No need to open Gmail.

## Tech Stack
- LangGraph + LangChain (Agent framework)
- ChromaDB (Vector Store)
- Groq API / llama3-70b (LLM)
- Gmail API (Email provider)
- Firebase (Auth + Database + Storage)
- Streamlit (UI)
- FastAPI (Backend)

## Lab Progress
- [x] Lab 1 - Problem Framing & Architecture
- [x] Lab 2 - Vector Store (ChromaDB)
- [x] Lab 3 - LangGraph ReAct Agent
- [x] Lab 4 - Multi-Agent Orchestration
- [x] Lab 5 - Persistence & HITL
- [x] Lab 6 - Security Guardrails & Jailbreaking
- [x] Lab 7 - Evaluation & Observability
- [x] Lab 8 - FastAPI API Layer
- [x] Lab 9 - Docker Packaging
- [x] Lab 10 - CI/CD Quality Gate
- [x] Lab 11 - Drift Monitoring & Feedback Loops

## Quick Runbook
- Start the API locally: `python -m uvicorn main:app --host 127.0.0.1 --port 8000`
- Start the feedback UI: `streamlit run app.py`
- Run the evaluation gate locally: `python run_eval.py`
- Simulate a CI-breaking quality regression: `$env:BREAK_AGENT_FOR_CI='true'; python run_eval.py`
- Seed demo feedback rows for the Lab 11 loop: `python seed_feedback_demo.py --reset`
- Analyze collected thumbs-down feedback: `python analyze_feedback.py`
- Generate the Lab 9 Docker log after Docker Desktop is installed: `powershell -ExecutionPolicy Bypass -File .\generate_docker_build_log.ps1`
- Packaging and CI/CD report: `packaging_and_cicd_report.md`
- Versioned CI thresholds: `eval_thresholds.json`
