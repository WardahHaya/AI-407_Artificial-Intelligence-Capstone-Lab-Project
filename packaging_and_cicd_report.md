# Industrial Packaging And CI/CD Report

## Scope
This report covers the industrial packaging and automated quality-gate work for the Buraq agent. The deliverables are the container build, multi-service orchestration, runtime secret injection strategy, the CI-ready evaluation gate, threshold configuration, and breaking-change evidence.

## Industrial Packaging And Deployment Strategy

### Container image design
- Base image: `python:3.11-slim`
- Why this image: it is small, stable, and compatible with the project dependencies without carrying a full desktop Linux userland.
- Runtime optimization: the Docker deployment uses `BURAQ_LIGHTWEIGHT_EMBEDDINGS=true`, which switches the API to a deterministic hash-based embedding model inside the container. This keeps the packaged runtime reproducible without forcing a large Torch download for the deployment proof.
- Layer ordering strategy:
  1. Install system libraries first.
  2. Copy dependency manifests before application code.
  3. Install Python packages.
  4. Copy source code last.
- Why this ordering matters: Docker can reuse the expensive dependency layer when only application code changes.
- Multi-stage decision: a single-stage image was kept intentionally because the runtime still needs the compiled Python stack used by FastAPI, Chroma, and NumPy at execution time. The main optimization came from a lean runtime dependency file and the lightweight container embedding mode rather than a second build stage.

### Secret-free image
- No secrets are copied during `docker build`.
- `.dockerignore` excludes `.env`, `credentials.json`, `token.pickle`, local SQLite files, local Chroma data, and Python caches.
- Runtime secret injection is handled through environment variables in `docker-compose.yaml`.
- Example runtime injection:

```powershell
$env:GROQ_API_KEY="your-key"
docker compose up -d
```

- The image can still start without `GROQ_API_KEY`; in that case the API falls back to the deterministic demo model for verification and local lab evidence.

### Multi-service orchestration
- Service 1: `agent-api`
- Service 2: `chroma`
- Service discovery: the API uses `CHROMA_HOST=chroma` and both services share the `buraq-net` Docker network.
- Start together: `docker compose up -d`
- Stop together: `docker compose down`
- Persistent data:
  - `chroma_data` keeps the vector index alive across restarts.
  - `checkpoint_data` keeps `checkpoint_db.sqlite` alive across API restarts.

### Zero-manual-step startup
- `main.py` now ensures the Chroma collection exists during application startup.
- If the collection is empty, the API automatically ingests the grounded source files from `Initial_Data`.
- This means the packaged system can start from source alone without a manual pre-ingestion step.

### End-to-end evidence
- `docker_build.log` is the primary deployment evidence file.
- `generate_docker_build_log.ps1` automates:
  - `docker compose build`
  - `docker compose up -d`
  - `/health` verification
  - `/chat` verification
  - container restart
  - persistence checks for checkpoint rows and Chroma collection count
  - `docker ps`

## Automated Quality Gates And CI/CD

### CI-ready evaluation script
- File: `run_eval.py`
- Headless behavior:
  - no interactive prompts
  - all credentials read from environment variables
  - exit code `0` when all thresholds pass
  - exit code `1` when any threshold fails
- Machine-readable outputs:
  - `evaluation_results.json`: case-level results
  - `ci_eval_results.json`: metric summary with score, threshold, and pass/fail
  - `evaluation_report.md`: human-readable summary

### Pipeline configuration
- File: `.github/workflows/main.yml`
- Trigger: every push to `main`
- Steps:
  1. checkout
  2. setup Python 3.11
  3. install dependencies
  4. build Docker image
  5. run `python run_eval.py`
  6. upload evaluation artifacts
- Secrets are sourced from the GitHub Actions secret store:
  - `OPENAI_API_KEY`
  - `GROQ_API_KEY`
  - `LANGSMITH_API_KEY`

### Versioned threshold configuration
- File: `eval_thresholds.json`
- Metrics:
  - `min_faithfulness = 0.9`
  - `min_relevancy = 0.6`
  - `min_tool_call_accuracy = 0.95`

### Threshold justification
- Faithfulness `0.9`:
  - High because grounded answers should stay close to retrieved evidence.
  - If raised by 10% to `0.99`, small wording differences would create noisy CI failures.
  - If lowered by 10% to `0.81`, noticeably weaker grounding could slip through.
- Relevancy `0.6`:
  - Moderate because some tool outputs are intentionally verbose compared with the short reference answers.
  - If raised by 10% to `0.66`, draft-style responses would fail more often despite still being useful.
  - If lowered by 10% to `0.54`, vague answers would pass too easily.
- Tool-call accuracy `0.95`:
  - High because tool routing is a core correctness property.
  - If raised by 10% it effectively becomes `1.0`, which is too brittle for minor evaluation noise.
  - If lowered by 10% to `0.855`, the pipeline could allow meaningful routing regressions.

### Breaking-change demonstration
- File: `ci_break_test_results.txt`
- Passing state:
  - `python run_eval.py`
  - Exit code `0`
- Failing state:
  - `BREAK_AGENT_FOR_CI=true python run_eval.py`
  - Exit code `1`
- This simulates a grounding or prompt regression and proves the quality gate blocks degraded behavior.

## Submission Map
- `Dockerfile`
- `docker-compose.yaml`
- `docker-requirements.txt`
- `generate_docker_build_log.ps1`
- `docker_build.log`
- `.github/workflows/main.yml`
- `run_eval.py`
- `eval_thresholds.json`
- `ci_eval_results.json`
- `ci_break_test_results.txt`
- `packaging_and_cicd_report.md`
