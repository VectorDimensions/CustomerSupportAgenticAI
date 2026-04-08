"""
inference.py — Baseline agent for the support_ticket_env OpenEnv environment.

This script is the entry point for hackathon evaluation. It connects to the
environment server, runs one episode using an LLM as the agent, and emits
structured log lines that the evaluation harness parses.

Log format (MANDATORY — do not change field names or ordering):
    [START] task=<task_name> env=support_ticket_env model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Why structured logs?
    The evaluation harness greps for these exact patterns to extract scores and
    step counts. Any deviation in field names, ordering, or formatting will cause
    the harness to miss the data and score the submission as 0. We use a fixed
    format string for each line type to make it impossible to accidentally deviate.

Environment variables (all read at startup):
    API_BASE_URL        — LLM API endpoint (default: https://router.huggingface.co/v1)
    MODEL_NAME          — Model identifier (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN            — Hugging Face API key (required for authenticated access)
    IMAGE_NAME          — Docker image name (optional, used with from_docker_image)
    SUPPORT_TICKET_TASK — Task to run: easy / medium / hard (default: easy)
    ENV_BASE_URL        — Environment server URL (default: http://localhost:7860)
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from typing import Any

from openai import OpenAI

from support_ticket_env.client import SupportTicketEnv, SupportTicketEnvError
from support_ticket_env.models import SupportTicketAction, SupportTicketObservation, VALID_COMMANDS


# ---------------------------------------------------------------------------
# Configuration — read from environment variables at startup
# ---------------------------------------------------------------------------
# We read all config up front so any missing required variables surface
# immediately, not halfway through an episode.

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN: str = os.environ.get("HF_TOKEN", "")
IMAGE_NAME: str = os.environ.get("IMAGE_NAME", "")
TASK_NAME: str = os.environ.get("SUPPORT_TICKET_TASK", "easy")
ENV_BASE_URL: str = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

# Amendment 3: align score computation with the sample script's pattern.
# MAX_TOTAL_REWARD is the theoretical ceiling if every step earned the highest
# positive reward (+0.25 correct resolution). We use it to normalise the raw
# reward sum into [0, 1] as a fallback when the grader score isn't available.
# Per-task step budgets: easy=5, medium=8, hard=12.
_MAX_REWARD_PER_STEP: float = 0.25
_MAX_STEPS_BY_TASK: dict[str, int] = {"easy": 5, "medium": 8, "hard": 12}
MAX_TOTAL_REWARD: float = _MAX_REWARD_PER_STEP * _MAX_STEPS_BY_TASK.get(TASK_NAME, 12)

# The system prompt is the most important part of the inference script.
# It tells the LLM exactly what commands are available, what format to respond
# in, and what the goal is. A well-crafted system prompt dramatically improves
# baseline performance without any fine-tuning.
SYSTEM_PROMPT = f"""You are a customer support agent for an e-commerce company. Your job is to resolve customer support tickets by using the available tools/commands.

AVAILABLE COMMANDS:
{json.dumps({
    "lookup_order": {"params": {"order_id": "str"}, "description": "Look up order details by order ID"},
    "lookup_customer": {"params": {"customer_id": "str"}, "description": "Look up customer profile by customer ID"},
    "check_policy": {"params": {"policy_type": "str (refund_policy|replacement_policy|return_policy|shipping_policy|escalation_criteria)"}, "description": "Check company policy"},
    "check_inventory": {"params": {"product_id": "str"}, "description": "Check if a product is in stock"},
    "issue_refund": {"params": {"order_id": "str", "amount": "float", "reason": "str"}, "description": "Issue a refund for an order"},
    "send_replacement": {"params": {"order_id": "str", "product_id": "str"}, "description": "Send a replacement item"},
    "escalate": {"params": {"reason": "str", "priority": "str (low|normal|high)"}, "description": "Escalate to a human agent (only for fraud or refunds > $500)"},
    "send_response": {"params": {"message": "str"}, "description": "Send a response to the customer (MUST be your final action)"},
}, indent=2)}

RESPONSE FORMAT:
You MUST respond with a single JSON object. No other text, no markdown, no explanation.
{{
  "command": "<command_name>",
  "parameters": {{<key>: <value>}},
  "reasoning": "<brief explanation of why you chose this action>"
}}

WORKFLOW:
1. First, gather information: look up the order, customer, and relevant policies
2. Then take the correct resolution action (refund, replacement, etc.)
3. Finally, send a professional response to the customer with send_response

IMPORTANT RULES:
- Always look up the order before taking any action
- Check policy before issuing refunds or replacements
- send_response MUST be your last action — it ends the episode
- Do NOT escalate unless fraud is suspected or refund > $500
- Do NOT repeat the same action twice
- Be efficient — you have a limited number of steps
"""


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

def build_user_prompt(obs: SupportTicketObservation, step_rewards: list[float]) -> str:
    """Build the user-turn prompt from the current observation.

    We include the full observation context so the LLM has all the information
    it needs to decide the next action. The step rewards are included as a hint
    about whether previous actions were correct.

    Args:
        obs: The current observation from the environment.
        step_rewards: List of rewards received so far in this episode.

    Returns:
        A formatted string to send as the user turn to the LLM.
    """
    context_str = json.dumps(obs.context, indent=2) if obs.context else "{}"
    rewards_str = str(step_rewards) if step_rewards else "[]"

    return f"""TICKET: {obs.ticket_id}
CUSTOMER MESSAGE: {obs.customer_message}

CURRENT STEP: {obs.step_number}
REWARDS SO FAR: {rewards_str}

LAST ACTION RESULT: {json.dumps(obs.last_action_result, indent=2) if obs.last_action_result else "None"}
LAST ACTION ERROR: {obs.last_action_error or "None"}

ACCUMULATED CONTEXT (data you've gathered so far):
{context_str}

What is your next action? Respond with a single JSON object."""


def call_llm(client: OpenAI, conversation: list[dict]) -> str:
    """Call the LLM and return the raw response text.

    Args:
        client: The OpenAI client configured with the API base URL and token.
        conversation: The full conversation history (system + user turns).

    Returns:
        The raw text content of the LLM's response.
    """
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=conversation,
        temperature=0.1,   # low temperature for more deterministic, policy-following behaviour
        max_tokens=512,    # actions are short JSON objects — 512 tokens is more than enough
    )
    return response.choices[0].message.content or ""


def parse_action(raw_response: str) -> SupportTicketAction:
    """Parse the LLM's raw response into a SupportTicketAction.

    The LLM is instructed to respond with a JSON object. We try to parse it
    directly; if that fails, we fall back to a safe default action (lookup_order
    for the first order mentioned in the task) so the episode can continue.

    Args:
        raw_response: The raw text returned by the LLM.

    Returns:
        A SupportTicketAction parsed from the response, or a safe fallback action.
    """
    # Strip markdown code fences if the LLM wrapped the JSON in ```json ... ```
    # Some models do this despite being told not to — we handle it gracefully.
    cleaned = raw_response.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove the opening ``` line and the closing ``` line
        cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        data = json.loads(cleaned)
        return SupportTicketAction(
            command=data.get("command", "lookup_order"),
            parameters=data.get("parameters", {}),
            message=data.get("message"),
        )
    except (json.JSONDecodeError, KeyError, TypeError):
        # Safety net: if the LLM returns garbage, fall back to a harmless lookup.
        # This keeps the episode alive rather than crashing the inference script.
        # The -0.10 penalty for an invalid command is better than a crash.
        print(
            f"  [WARN] Failed to parse LLM response as JSON. "
            f"Raw response: {raw_response[:200]!r}. Using fallback action.",
            file=sys.stderr,
        )
        return SupportTicketAction(
            command="lookup_order",
            parameters={"order_id": "ORD-1042"},
        )


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------

def run_episode() -> None:
    """Run one complete episode and emit the required log lines.

    Connects to the environment server, resets the episode, runs the agent loop
    until done=True or an exception occurs, and always emits the [END] log line.

    Returns:
        None. All output goes to stdout (log lines) and stderr (debug/warnings).
    """
    # Validate required config before doing anything else.
    if not HF_TOKEN:
        print(
            "[WARN] HF_TOKEN is not set. Requests to the LLM API may fail with 401.",
            file=sys.stderr,
        )

    # Initialise the OpenAI client pointing at the HF inference router.
    # We use the OpenAI client (not requests) because it handles retries,
    # streaming, and token counting out of the box.
    llm_client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN or "dummy",  # some endpoints accept any non-empty string
    )

    # Emit the [START] log line immediately so the harness knows the episode began.
    print(f"[START] task={TASK_NAME} env=support_ticket_env model={MODEL_NAME}", flush=True)

    step_rewards: list[float] = []
    steps_taken: int = 0
    final_score: float = 0.0
    success: bool = False

    # We use a context manager so close() is always called, even on exception.
    with SupportTicketEnv(base_url=ENV_BASE_URL) as env:
        try:
            # Reset the environment and get the initial observation.
            obs = env.reset(task_id=TASK_NAME)

            # Build the conversation history. We keep the full history so the LLM
            # has context about what it has already tried (avoids repeated actions).
            conversation: list[dict[str, Any]] = [
                {"role": "system", "content": SYSTEM_PROMPT},
            ]

            # Agent loop: keep stepping until the episode is done.
            while not obs.done:
                # Build the user prompt from the current observation.
                user_prompt = build_user_prompt(obs, step_rewards)
                conversation.append({"role": "user", "content": user_prompt})

                # Call the LLM to get the next action.
                raw_response = call_llm(llm_client, conversation)

                # Add the LLM's response to the conversation history so it can
                # reference its own previous reasoning in subsequent turns.
                conversation.append({"role": "assistant", "content": raw_response})

                # Parse the LLM response into a typed action.
                action = parse_action(raw_response)
                action_str = f"{action.command}({json.dumps(action.parameters)})"

                # Submit the action to the environment.
                obs, reward, done = env.step(action)
                step_rewards.append(reward)
                steps_taken += 1

                # Emit the [STEP] log line. Format is MANDATORY — do not change.
                error_str = obs.last_action_error or "null"
                print(
                    f"[STEP] step={steps_taken} action={action_str} "
                    f"reward={reward:.2f} done={str(done).lower()} error={error_str}",
                    flush=True,
                )

            # Score computation — matches sample inference.py pattern exactly.
            # Primary: grader score from context (deterministic weighted criteria).
            # Fallback: sum(rewards) / MAX_TOTAL_REWARD clamped to [0, 1].
            grader_score = obs.context.get("score")
            if grader_score is not None:
                final_score = min(max(float(grader_score), 0.0), 1.0)
            else:
                raw_sum = sum(step_rewards)
                final_score = max(0.0, min(1.0, raw_sum / MAX_TOTAL_REWARD)) if MAX_TOTAL_REWARD > 0 else 0.0
            success = final_score >= 0.5

        except SupportTicketEnvError as exc:
            print(f"[ERROR] Environment server error: {exc}", file=sys.stderr)
        except Exception as exc:
            print(f"[ERROR] Unexpected error: {exc}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        finally:
            # Emit the [END] log line ALWAYS — even if an exception occurred.
            # The harness expects this line to be present for every episode.
            rewards_str = ",".join(f"{r:.2f}" for r in step_rewards)
            print(
                f"[END] success={str(success).lower()} steps={steps_taken} "
                f"score={final_score:.2f} rewards={rewards_str}",
                flush=True,
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_episode()
