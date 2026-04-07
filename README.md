---
title: support_ticket_env
emoji: 🎫
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
tags:
  - openenv
---

# support_ticket_env

An OpenEnv-compatible reinforcement learning environment that simulates an e-commerce customer support desk.

An AI agent resolves support tickets by issuing **structured commands** against a fully in-memory backend (orders, customers, products, policies). The environment evaluates agents on whether they took the **correct sequence of actions** to resolve each ticket — not on free-text quality.

Built for the **Meta PyTorch OpenEnv Hackathon**.

---

## Environment Description and Motivation

Customer support ticket resolution is one of the most common, high-volume real-world tasks performed globally. Companies handle millions of tickets daily, each requiring an agent to:

1. Understand the customer issue
2. Look up relevant information from backend systems
3. Apply company policies
4. Take corrective actions (refunds, replacements, escalations)
5. Compose a professional response

This environment provides a **structured, deterministic, reproducible testbed** for training agents on multi-step decision-making with policy constraints. Unlike chat benchmarks, agents are evaluated on *what they did*, not *what they said*.

---

## Action Space

The agent chooses from 8 structured commands:

| Command | Parameters | Description |
|---|---|---|
| `lookup_order` | `order_id: str` | Retrieve order details (status, tracking, amount) |
| `lookup_customer` | `customer_id: str` | Retrieve customer profile (tier, history) |
| `check_policy` | `policy_type: str` | Look up company policy (refund, return, shipping, etc.) |
| `check_inventory` | `product_id: str` | Check current stock level for a product |
| `issue_refund` | `order_id: str, amount: float, reason: str` | Process a refund (must comply with policy) |
| `send_replacement` | `order_id: str, product_id: str` | Ship a replacement item (must check inventory first) |
| `escalate` | `reason: str, priority: str` | Escalate to a human agent (only for fraud or refunds > $500) |
| `send_response` | `message: str` | Send a customer-facing message (terminal action) |

---

## Observation Space

After each step, the environment returns:

| Field | Type | Description |
|---|---|---|
| `ticket_id` | `str` | Unique ticket identifier (constant per episode) |
| `customer_message` | `str` | The original customer complaint (constant per episode) |
| `step_number` | `int` | 1-based step counter |
| `last_action_result` | `dict \| null` | Structured result of the last command |
| `last_action_error` | `str \| null` | Error message if the last command failed |
| `available_commands` | `list[str]` | All 8 valid command names |
| `context` | `dict` | Accumulated data retrieved so far in this episode |
| `done` | `bool` | True when the episode has ended |

---

## Task Descriptions

### Task 1: Order Status Inquiry (Easy)

- **Scenario**: Customer asks "Where is my order ORD-1042?"
- **Max steps**: 5
- **Correct resolution**: Look up the order, send a response with the status and delivery estimate
- **Grader criteria**: lookup_order called (0.30) + send_response called (0.20) + status mentioned (0.30) + delivery info mentioned (0.20)

### Task 2: Refund Request (Medium)

- **Scenario**: Customer wants a refund for a damaged item on order ORD-2087
- **Max steps**: 8
- **Correct resolution**: Look up order + customer + refund policy, issue full refund ($149.99), send response
- **Grader criteria**: lookup_order (0.15) + lookup_customer (0.10) + check_policy (0.15) + issue_refund (0.25) + correct amount (0.15) + send_response (0.10) + professional response (0.10)

### Task 3: Complex Multi-Issue (Hard)

- **Scenario**: Customer received wrong item in ORD-3021 AND was overcharged $15 on ORD-3022
- **Max steps**: 12
- **Correct resolution**: Look up both orders, check inventory for replacement, send replacement (or handle out-of-stock), issue $15 partial refund, send unified response
- **Grader criteria**: Both orders looked up (0.10) + customer lookup (0.05) + policies checked (0.10) + inventory checked (0.10) + send_replacement (0.15) + issue_refund $15 (0.15) + both issues in response (0.15) + no unnecessary escalation (0.05) + correct order of operations (0.05) + professional response (0.10)

---

## Reward Function

| Event | Reward |
|---|---|
| Valid info-gathering action (new data) | +0.10 |
| Correct resolution action (refund/replacement) | +0.25 |
| Good customer response | +0.15 |
| Invalid command | -0.10 |
| Incorrect parameters | -0.05 |
| Policy violation | -0.15 |
| Repeated identical action | -0.02 |
| Unnecessary escalation | -0.10 |

---

## Setup Instructions

### Prerequisites

- Python 3.11+
- pip

### Local Installation

```bash
# Clone the repository
git clone https://huggingface.co/spaces/<username>/support-ticket-env
cd support-ticket-env

# Install the package in editable mode
pip install -e .

# Install server dependencies
pip install -r server/requirements.txt
```

### Running the Server Locally

```bash
# Start the FastAPI server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Test the server is running
curl http://localhost:7860/health
```

### Running with Docker

```bash
# Build the image
docker build -t support-ticket-env server/

# Run the container
docker run -p 7860:7860 \
  -e SUPPORT_TICKET_TASK=easy \
  support-ticket-env
```

### Running the Inference Script

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your_hf_token_here"
export SUPPORT_TICKET_TASK="easy"
export ENV_BASE_URL="http://localhost:7860"

python inference.py
```

---

## Usage Example

```python
from support_ticket_env.client import SupportTicketEnv
from support_ticket_env.models import SupportTicketAction

# Connect to the environment server
with SupportTicketEnv(base_url="http://localhost:7860") as env:
    # Start a new episode
    obs = env.reset(task_id="easy")
    print(f"Ticket: {obs.ticket_id}")
    print(f"Customer: {obs.customer_message}")

    # Agent loop
    while not obs.done:
        # Look up the order
        action = SupportTicketAction(
            command="lookup_order",
            parameters={"order_id": "ORD-1042"}
        )
        obs, reward, done = env.step(action)
        print(f"Step {obs.step_number}: reward={reward:.2f}, done={done}")
        print(f"Result: {obs.last_action_result}")

        if not done:
            # Send a response to close the ticket
            action = SupportTicketAction(
                command="send_response",
                parameters={"message": "Your order ORD-1042 has been shipped and will arrive by Jan 17."}
            )
            obs, reward, done = env.step(action)

    # Final score is in the context dict
    print(f"Final score: {obs.context.get('score', 0.0):.4f}")
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `HF_TOKEN` | *(required)* | Hugging Face API key |
| `IMAGE_NAME` | *(optional)* | Docker image name |
| `SUPPORT_TICKET_TASK` | `easy` | Task to run: `easy`, `medium`, or `hard` |
| `ENV_BASE_URL` | `http://localhost:7860` | Environment server URL |

---

## Baseline Scores

Scores from the baseline Qwen2.5-72B-Instruct agent:

| Task | Difficulty | Score | Steps |
|---|---|---|---|
| Order Status Inquiry | Easy | 0.85 | 3 |
| Refund Request | Medium | 0.72 | 6 |
| Complex Multi-Issue | Hard | 0.58 | 10 |

---

## HF Space Deployment

```bash
# Install huggingface_hub
pip install huggingface_hub

# Login
huggingface-cli login

# Push to HF Space
cd support_ticket_env
openenv push --repo-id <username>/support-ticket-env

# Verify the Space is live
curl -X POST https://<space-url>/reset

# Run validation
./validate-submission.sh https://<space-url>
```
