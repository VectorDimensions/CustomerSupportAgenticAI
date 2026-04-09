"""
server/graders.py — Deterministic episode graders for the SupportTicket RL environment.

Grading Philosophy
------------------
The grader is the evaluation metric — it answers "did the agent actually solve the
customer's problem?" at the end of an episode. It is deliberately separate from the
reward function (rewards.py), which is the training signal that fires at every step.

Why separate grading from rewards?
    Step rewards are dense and noisy — they fire on every action and guide learning.
    The grader is sparse and precise — it fires once per episode and measures success.
    Conflating the two would make it impossible to tell whether an agent is "learning"
    (improving step rewards) vs "succeeding" (improving grader scores).

Grading approach: weighted sub-score sum
    Each grader breaks the task into independent criteria, assigns each a weight, and
    sums the weighted scores. This gives partial credit — an agent that looked up the
    order but forgot to send a response still scores better than one that did nothing.
    The final score is clamped to [0.0, 1.0] to satisfy the OpenEnv contract.
"""

from __future__ import annotations

from typing import Any


def grade_easy(action_history: list[dict[str, Any]], context: dict[str, Any]) -> float:
    """Grade a completed easy-scenario episode.

    Criteria and weights:
        1. lookup_order called for ORD-1042          -> 0.30
        2. send_response called                       -> 0.20
        3. Response mentions correct order status     -> 0.30
        4. Response includes delivery information     -> 0.20

    Args:
        action_history: List of all actions taken in the episode.
        context: The accumulated episode context dict.

    Returns:
        A float in [0.0, 1.0] representing the agent's score.
    """
    score = 0.0

    # Criterion 1 (0.30): prerequisite — agent must ground its response in real data
    if _did_lookup_order(action_history, "ORD-1042"):
        score += 0.30

    response_message = _get_response_message(action_history)

    # Criterion 2 (0.20): agent must close the loop with the customer
    if response_message is not None:
        score += 0.20

    # Criterion 3 (0.30): the customer asked about status — this is the core answer
    if response_message and _mentions_any(response_message, ["shipped", "tracking", "status", "on its way"]):
        score += 0.30

    # Criterion 4 (0.20): the customer wants to know *when* — delivery info completes the answer
    if response_message and _mentions_any(response_message, ["delivery", "arrive", "estimated", "days", "january", "jan"]):
        score += 0.20

    return max(0.0, min(1.0, score))


def grade_medium(action_history: list[dict[str, Any]], context: dict[str, Any]) -> float:
    """Grade a completed medium-scenario episode.

    Criteria and weights:
        1. lookup_order called for ORD-2087           -> 0.15
        2. lookup_customer called                     -> 0.10
        3. check_policy('refund_policy') called       -> 0.15
        4. issue_refund called                        -> 0.25
        5. Refund amount matches policy (149.99)      -> 0.15
        6. send_response called                       -> 0.10
        7. Response is professional and accurate      -> 0.10

    Args:
        action_history: List of all actions taken in the episode.
        context: The accumulated episode context dict.

    Returns:
        A float in [0.0, 1.0] representing the agent's score.
    """
    score = 0.0

    # Criterion 1 (0.15): must know the order details before verifying eligibility
    if _did_lookup_order(action_history, "ORD-2087"):
        score += 0.15

    # Criterion 2 (0.10): verify identity and determine account tier
    if _did_lookup_customer(action_history):
        score += 0.10

    # Criterion 3 (0.15): ground the decision in policy, not assumption
    if _did_check_policy(action_history, "refund_policy"):
        score += 0.15

    # Criterion 4 (0.25): the core resolution — the customer wants their money back
    refund_action = _get_refund_action(action_history)
    if refund_action is not None:
        score += 0.25

    # Criterion 5 (0.15): wrong amount is worse than no refund
    if refund_action is not None:
        amount = refund_action.get("parameters", {}).get("amount")
        try:
            if abs(float(amount) - 149.99) <= 0.01:
                score += 0.15
        except (TypeError, ValueError):
            pass

    response_message = _get_response_message(action_history)

    # Criterion 6 (0.10): the customer must be informed of the resolution
    if response_message is not None:
        score += 0.10

    # Criterion 7 (0.10): a professional response addresses the specific issue
    if response_message and _mentions_any(response_message, ["refund", "damaged", "sorry", "apolog", "processed", "credited"]):
        score += 0.10

    return max(0.0, min(1.0, score))


def grade_hard(action_history: list[dict[str, Any]], context: dict[str, Any]) -> float:
    """Grade a completed hard-scenario episode.

    Criteria and weights:
        1. Both orders looked up (ORD-3021 + ORD-3022)  -> 0.10
        2. Customer looked up                            -> 0.05
        3. Relevant policies checked                     -> 0.10
        4. Inventory checked for PROD-003                -> 0.10
        5. send_replacement called                       -> 0.15
        6. issue_refund called for exactly $15.00        -> 0.15
        7. Both issues addressed in response             -> 0.15
        8. No unnecessary escalation                     -> 0.05
        9. Correct order of operations                   -> 0.05
       10. Response professional and complete            -> 0.10

    Args:
        action_history: List of all actions taken in the episode.
        context: The accumulated episode context dict.

    Returns:
        A float in [0.0, 1.0] representing the agent's score.
    """
    score = 0.0

    # Criterion 1 (0.10): both orders must be looked up — 0.05 each for partial credit
    if _did_lookup_order(action_history, "ORD-3021"):
        score += 0.05
    if _did_lookup_order(action_history, "ORD-3022"):
        score += 0.05

    # Criterion 2 (0.05): verify identity before taking any action
    if _did_lookup_customer(action_history):
        score += 0.05

    # Criterion 3 (0.10): check either replacement or refund policy
    if _did_check_policy(action_history, "replacement_policy") or _did_check_policy(action_history, "refund_policy"):
        score += 0.10

    # Criterion 4 (0.10): must discover PROD-003 is out of stock before promising replacement
    if _did_check_inventory(action_history, "PROD-003"):
        score += 0.10

    # Criterion 5 (0.15): attempt the replacement even if stock is 0
    if _did_send_replacement(action_history):
        score += 0.15

    # Criterion 6 (0.15): partial refund of exactly $15 for the billing error
    refund_action = _get_refund_action(action_history)
    if refund_action is not None:
        amount = refund_action.get("parameters", {}).get("amount")
        try:
            if abs(float(amount) - 15.00) <= 0.01:
                score += 0.15
        except (TypeError, ValueError):
            pass

    # Criterion 7 (0.15): response must address both issues
    response_message = _get_response_message(action_history)
    if response_message:
        mentions_replacement = _mentions_any(response_message, ["replacement", "wrong item", "wrong color", "blue", "red", "mouse"])
        mentions_refund = _mentions_any(response_message, ["refund", "overcharged", "billing", "$15", "15.00", "credited"])
        if mentions_replacement and mentions_refund:
            score += 0.15
        elif mentions_replacement or mentions_refund:
            score += 0.07  # partial credit for addressing at least one issue

    # Criterion 8 (0.05): bonus for NOT escalating a routine ticket
    if not _did_escalate(action_history):
        score += 0.05

    # Criterion 9 (0.05): bonus for gathering info before acting
    if _correct_order_of_operations(action_history):
        score += 0.05

    # Criterion 10 (0.10): professional tone markers
    if response_message and _mentions_any(response_message, ["sorry", "apolog", "resolved", "processed", "please", "contact"]):
        score += 0.10

    return max(0.0, min(1.0, score))


def grade(task_id: str, action_history: list[dict[str, Any]], context: dict[str, Any]) -> float:
    """Route to the correct grader function based on task_id.

    Scores are clamped to (0.01, 0.99) — strictly between 0 and 1 as required
    by the OpenEnv hackathon validator. Exactly 0.0 or 1.0 fails validation.

    Args:
        task_id: The task name — "easy"/"task_1", "medium"/"task_2", or "hard"/"task_3".
        action_history: List of all actions taken in the episode.
        context: The accumulated episode context dict.

    Returns:
        A float strictly in (0.01, 0.99).

    Raises:
        ValueError: If task_id is not one of the known task names.
    """
    graders = {
        "easy": grade_easy,
        "medium": grade_medium,
        "hard": grade_hard,
        # aligned with openenv.yaml and env/registry.py task IDs
        "task_1": grade_easy,
        "task_2": grade_medium,
        "task_3": grade_hard,
    }
    grader_fn = graders.get(task_id)
    if grader_fn is None:
        raise ValueError(f"Unknown task_id: {task_id!r}. Valid options are: {list(graders.keys())}")
    raw_score = grader_fn(action_history, context)
    # Clamp strictly between 0 and 1 — validator requires score in (0, 1) exclusive.
    return max(0.01, min(0.99, raw_score))


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _did_lookup_order(action_history: list[dict], order_id: str) -> bool:
    """Return True if lookup_order was called with the specified order_id."""
    return any(
        a.get("command") == "lookup_order" and a.get("parameters", {}).get("order_id") == order_id
        for a in action_history
    )


def _did_lookup_customer(action_history: list[dict]) -> bool:
    """Return True if lookup_customer was called at least once."""
    return any(a.get("command") == "lookup_customer" for a in action_history)


def _did_check_policy(action_history: list[dict], policy_type: str) -> bool:
    """Return True if check_policy was called with the specified policy_type."""
    return any(
        a.get("command") == "check_policy" and a.get("parameters", {}).get("policy_type") == policy_type
        for a in action_history
    )


def _did_check_inventory(action_history: list[dict], product_id: str) -> bool:
    """Return True if check_inventory was called with the specified product_id."""
    return any(
        a.get("command") == "check_inventory" and a.get("parameters", {}).get("product_id") == product_id
        for a in action_history
    )


def _did_send_replacement(action_history: list[dict]) -> bool:
    """Return True if send_replacement was called at least once."""
    return any(a.get("command") == "send_replacement" for a in action_history)


def _did_escalate(action_history: list[dict]) -> bool:
    """Return True if escalate was called at least once."""
    return any(a.get("command") == "escalate" for a in action_history)


def _get_refund_action(action_history: list[dict]) -> dict | None:
    """Return the first issue_refund action dict, or None if none was taken."""
    for action in action_history:
        if action.get("command") == "issue_refund":
            return action
    return None


def _get_response_message(action_history: list[dict]) -> str | None:
    """Return the message text from the first send_response action, or None."""
    for action in action_history:
        if action.get("command") == "send_response":
            msg = action.get("parameters", {}).get("message") or action.get("message")
            if msg:
                return str(msg)
    return None


def _mentions_any(text: str, keywords: list[str]) -> bool:
    """Return True if the text contains at least one keyword (case-insensitive)."""
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def _correct_order_of_operations(action_history: list[dict]) -> bool:
    """Return True if at least one lookup happened before the first resolution action."""
    resolution_commands = {"issue_refund", "send_replacement"}
    lookup_commands = {"lookup_order", "lookup_customer", "check_policy", "check_inventory"}

    first_resolution_idx = None
    last_lookup_before_resolution_idx = None

    for i, action in enumerate(action_history):
        cmd = action.get("command", "")
        if cmd in lookup_commands:
            last_lookup_before_resolution_idx = i
        elif cmd in resolution_commands and first_resolution_idx is None:
            first_resolution_idx = i

    if first_resolution_idx is None or last_lookup_before_resolution_idx is None:
        return False
    return last_lookup_before_resolution_idx < first_resolution_idx


# ---------------------------------------------------------------------------
# GRADERS mapping — aligns with openenv.yaml task IDs and env/registry.py
# ---------------------------------------------------------------------------
from env.graders.grader_1 import grader_1
from env.graders.grader_2 import grader_2
from env.graders.grader_3 import grader_3

GRADERS = {
    "task_1": grader_1,
    "task_2": grader_2,
    "task_3": grader_3,
}
