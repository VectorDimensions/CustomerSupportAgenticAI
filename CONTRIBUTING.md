# Contributing — Local Setup & Architecture

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                        AGENT (inference.py)                     │
│                                                                 │
│   reads env vars ──► builds prompt ──► calls LLM API           │
│         │                                    │                  │
│         │                             parses JSON response      │
│         │                                    │                  │
│         └──────────── SupportTicketAction ───┘                  │
│                              │                                  │
│              [START] / [STEP] / [END]  ──► stdout               │
└──────────────────────────────┼──────────────────────────────────┘
                               │  HTTP POST /step
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     SERVER  (server/app.py)                     │
│                                                                 │
│   POST /reset ──► SupportTicketEnvironment.reset()             │
│   POST /step  ──► SupportTicketEnvironment.step()              │
│   POST /grade ──► grade()   (standalone, harness-callable)     │
│   GET  /state ──► get_state()                                   │
│   GET  /health──► {"status": "ok"}                             │
└──────────────────────────────┼──────────────────────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
   ┌─────────────┐    ┌──────────────┐    ┌──────────────────┐
   │  data.py    │    │  rewards.py  │    │   graders.py     │
   │             │    │              │    │                  │
   │  Orders     │    │  per-step    │    │  grade_easy()    │
   │  Customers  │    │  reward      │    │  grade_medium()  │
   │  Products   │    │  rule chain  │    │  grade_hard()    │
   │  Policies   │    │  (priority   │    │  grade()         │
   │             │    │   ordered)   │    │  weighted sum    │
   └─────────────┘    └──────────────┘    └──────────────────┘
          │
          ▼
   ┌─────────────┐    ┌──────────────┐
   │ scenarios.py│    │environment.py│
   │             │    │              │
   │  easy       │───►│  reset()     │
   │  medium     │    │  step()      │
   │  hard       │    │  9-stage     │
   │             │    │  pipeline    │
   └─────────────┘    └──────────────┘


Episode lifecycle (one step):
─────────────────────────────
  Agent sends action
       │
       ▼
  1. Validate command ──── invalid? ──► -0.10 penalty, return error obs
       │
  2. Increment step count
       │
  3. Execute command against backend
       │
  4. Compute reward (rewards.py)
       │
  5. Update context (accumulated working memory)
       │
  6. Record in action history
       │
  7. Check termination (send_response / escalate / max_steps)
       │
  8. Grade episode if done (graders.py) ──► score in [0.0, 1.0]
       │
  9. Return SupportTicketObservation


Three tasks:
────────────
  easy   (max  5 steps) ── order status inquiry    ── ORD-1042
  medium (max  8 steps) ── refund for damaged item ── ORD-2087
  hard   (max 12 steps) ── wrong item + overcharge ── ORD-3021 + ORD-3022
```

---

## Local Setup (5 minutes)

### Prerequisites

- Python 3.11+ at `B:\Python311` (Windows) or system `python3` (Linux/Mac)
- Git

### 1. Clone & install

```bash
git clone <repo-url>
cd support-ticket-env

# Install the client package in editable mode
pip install -e .

# Install server dependencies
pip install fastapi uvicorn pydantic httpx openai
```

### 2. Start the server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```

Server is now live at `http://localhost:7860`.

### 3. Smoke-test the endpoints

```bash
# Health check
curl http://localhost:7860/health

# Reset an episode (easy task)
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy"}'

# Take a step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"command": "lookup_order", "parameters": {"order_id": "ORD-1042"}}'

# Send a response (ends the episode)
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"command": "send_response", "parameters": {"message": "Your order ORD-1042 is shipped, tracking TRK-9821034, delivery Jan 17."}}'
```

### 4. Run the test suites

```bash
# Core functionality (31 tests)
python -m pytest tests/test_submission.py -v

# Adversarial & DQ-gate tests (36 tests)
python -m pytest tests/test_adversarial.py -v

# All tests at once
python -m pytest tests/ -v
```

### 5. Run the baseline agent (needs HF token)

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_your_token_here"
export SUPPORT_TICKET_TASK="easy"   # or medium / hard
export ENV_BASE_URL="http://localhost:7860"

python inference.py
```

Expected output:
```
[START] task=easy env=support_ticket_env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=lookup_order({"order_id": "ORD-1042"}) reward=0.10 done=false error=null
[STEP] step=2 action=send_response(...) reward=0.15 done=true error=null
[END] success=true steps=2 score=1.0000 rewards=0.10,0.15
```

### 6. Run with Docker

```bash
docker build -t support-ticket-env server/
docker run -p 7860:7860 -e SUPPORT_TICKET_TASK=easy support-ticket-env
```

---

## Project Structure

```
support_ticket_env/     ← pip-installable client package
  models.py             ← Pydantic DTOs (Action, Observation)
  client.py             ← HTTP client (thin wrapper)
  __init__.py

server/
  data.py               ← in-memory backend (orders, customers, products, policies)
  scenarios.py          ← task definitions (easy / medium / hard)
  rewards.py            ← per-step reward logic
  graders.py            ← episode graders (weighted criteria)
  environment.py        ← episode lifecycle (reset / step / grade)
  app.py                ← FastAPI server factory
  Dockerfile
  requirements.txt

tests/
  test_submission.py    ← 31 tests: API contract, happy path, boundaries
  test_adversarial.py   ← 36 tests: DQ-gate, serialization, exploits, adversarial

inference.py            ← baseline agent (project root, required by OpenEnv spec)
openenv.yaml            ← environment manifest
pyproject.toml          ← package config
README.md               ← HF Space readme
```

---

## Available Commands (Action Space)

| Command | Required Parameters | What it does |
|---|---|---|
| `lookup_order` | `order_id` | Fetch order details |
| `lookup_customer` | `customer_id` | Fetch customer profile |
| `check_policy` | `policy_type` | Fetch company policy |
| `check_inventory` | `product_id` | Check stock level |
| `issue_refund` | `order_id`, `amount`, `reason` | Process a refund |
| `send_replacement` | `order_id`, `product_id` | Ship replacement item |
| `escalate` | `reason`, `priority` | Escalate to human (last resort) |
| `send_response` | `message` | Reply to customer — **ends the episode** |

---

## Reward Quick Reference

| Situation | Reward |
|---|---|
| Valid info-gathering (new data) | +0.10 |
| Correct resolution action | +0.25 |
| Good customer response | +0.15 |
| Invalid command | -0.10 |
| Wrong / missing parameters | -0.05 |
| Policy violation | -0.15 |
| Repeated identical action | -0.02 |
| Unnecessary escalation | -0.10 |

---

## Common Issues

**`ModuleNotFoundError: No module named 'support_ticket_env'`**
→ Run `pip install -e .` from the project root.

**`uvicorn: command not found`**
→ Run `pip install uvicorn` or use `python -m uvicorn server.app:app --port 7860`.

**Server returns 400 on `/step` before `/reset`**
→ Always call `/reset` first to start an episode.

**`SUPPORT_TICKET_TASK` not recognized**
→ Valid values are exactly `easy`, `medium`, `hard` (lowercase).
