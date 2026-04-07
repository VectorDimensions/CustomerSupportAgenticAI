"""
server/scenarios.py — Task scenario definitions for the SupportTicket RL environment.

This module is the single source of truth for what each episode is *about*. A Scenario
bundles together the ticket narrative (what the customer said), the constraints (how many
steps the agent gets), and the success criteria (what the grader checks at the end).

Why hard-code scenarios instead of loading them from a config file?
    Determinism. The hackathon judging requires that every agent faces exactly the same
    task under exactly the same conditions. Hard-coded scenarios eliminate any risk of
    a config file being accidentally modified between runs, which would make scores
    incomparable. If you need a new scenario, add it here and bump the version.

Three difficulty levels are provided:
    easy   — single order, status inquiry only, 5 steps
    medium — single order, refund workflow, 8 steps
    hard   — two orders, wrong item + billing error, 12 steps

The `get_scenario()` function is the public entry point; callers should never
instantiate Scenario directly (though nothing prevents it).
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Scenario dataclass
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    """Describes a single support-ticket task that the agent must resolve.

    A Scenario is immutable by convention — nothing in the environment mutates
    it after construction. It is loaded once at episode reset and then read-only.

    Args:
        name: Short identifier used as the SUPPORT_TICKET_TASK value
            (e.g. "easy", "medium", "hard").
        ticket_id: The ticket reference shown to the agent in every observation
            (e.g. "TICKET-001"). Purely cosmetic — it does not index into the backend.
        customer_message: The opening message the customer sent. This is the
            agent's primary signal for what needs to be done.
        max_steps: Hard cap on the number of steps before the episode is force-
            terminated. Chosen to give a competent agent enough room to succeed
            without allowing infinite loops.
        success_criteria: A dict of grading keys consumed by graders.py. Each key
            maps to the value the grader will check (e.g. an order ID string, a
            boolean, a float amount, or a list of keywords). Keeping this as a
            plain dict rather than a typed sub-dataclass makes it easy to add new
            criteria without changing the Scenario schema.
        difficulty: Human-readable difficulty label ("easy", "medium", "hard").
            Redundant with `name` for the three built-in scenarios, but useful if
            someone adds a custom scenario with a different name.
        required_orders: List of order IDs the agent must look up to fully resolve
            the ticket. Used by graders.py to check coverage.
        required_actions: List of command names that must appear in the action
            history for a full score. The grader checks these in addition to the
            success_criteria dict.

    Returns:
        A Scenario instance ready to be loaded by the environment.
    """

    name: str
    ticket_id: str
    customer_message: str
    max_steps: int
    success_criteria: dict
    difficulty: str
    required_orders: list[str] = field(default_factory=list)
    required_actions: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Easy scenario — order status inquiry
# ---------------------------------------------------------------------------

# The easy scenario is intentionally minimal: one order, one question, two actions.
# An agent that can look up an order and compose a reply should score 1.0 here.
# max_steps=5 gives a little slack for an agent that checks policy before responding,
# but a perfect agent only needs 2 steps (lookup + send_response).
EASY_SCENARIO = Scenario(
    name="easy",
    ticket_id="TICKET-001",
    difficulty="easy",
    max_steps=5,  # generous for a 2-action task — allows exploratory lookups
    customer_message=(
        "Hi, I placed order ORD-1042 three days ago and haven't received any updates. "
        "Can you check the status?"
    ),
    # The agent must look up this specific order — no other order is relevant here.
    required_orders=["ORD-1042"],
    # Minimum viable action set: look up the order, then tell the customer what you found.
    required_actions=["lookup_order", "send_response"],
    success_criteria={
        # The grader checks that the agent actually queried this order ID, not just
        # any order. This prevents an agent from hallucinating a response.
        "must_lookup_order": "ORD-1042",
        # The agent must close the ticket with a customer-facing message.
        "must_send_response": True,
        # The response should mention these keywords so the customer knows their
        # package is on the way. We check for any of these, not all of them,
        # because natural language has many ways to convey the same information.
        "response_must_mention": ["shipped", "tracking", "delivery"],
    },
)


# ---------------------------------------------------------------------------
# Medium scenario — refund for damaged item
# ---------------------------------------------------------------------------

# The medium scenario introduces a multi-step workflow: the agent must gather
# information (order + customer + policy) before taking action (issue_refund).
# max_steps=8 is enough for a perfect run (5 actions) plus a few exploratory steps.
# The refund amount (149.99) is the exact total_amount on ORD-2087 — a full refund
# because the item arrived damaged, which is covered by the refund_policy.
MEDIUM_SCENARIO = Scenario(
    name="medium",
    ticket_id="TICKET-002",
    difficulty="medium",
    max_steps=8,  # 5 required actions + 3 slack steps for policy exploration
    customer_message=(
        "I want a refund for order ORD-2087. The product arrived damaged."
    ),
    required_orders=["ORD-2087"],
    # The agent must follow the full refund workflow: gather context, check policy,
    # act, and communicate. Skipping any step should reduce the score.
    required_actions=[
        "lookup_order",
        "lookup_customer",
        "check_policy",
        "issue_refund",
        "send_response",
    ],
    success_criteria={
        "must_lookup_order": "ORD-2087",
        # Customer lookup is required because the grader needs to verify the agent
        # confirmed the customer's identity before issuing money back.
        "must_lookup_customer": True,
        # The agent must check the refund policy — not just assume it applies.
        # This teaches the agent to ground its decisions in policy, not intuition.
        "must_check_policy": "refund_policy",
        # The refund must actually be issued, not just mentioned in the response.
        "must_issue_refund": True,
        # The correct amount is the full order total (149.99) because the item
        # arrived damaged — a full-refund reason per the refund_policy rules.
        "correct_refund_amount": 149.99,
        "must_send_response": True,
    },
)


# ---------------------------------------------------------------------------
# Hard scenario — wrong item + billing discrepancy (two orders)
# ---------------------------------------------------------------------------

# The hard scenario is the most complex: two separate issues on two separate orders
# for the same customer. The agent must handle both in a single episode.
#
# Issue 1 (ORD-3021): Customer ordered a Red Wireless Mouse but received a Blue one.
#   → Agent should check inventory for PROD-003 (Red Mouse) before promising a
#     replacement. Since stock_count=0, the agent must handle the out-of-stock case.
#
# Issue 2 (ORD-3022): Customer was overcharged $15 on a 4K Monitor.
#   → Agent should issue a partial refund of exactly $15.00 (not the full order amount).
#
# max_steps=12 reflects the higher complexity: 7 required actions + 5 slack steps.
# The "no_unnecessary_escalation" criterion penalises agents that escalate routine
# issues — escalation is only warranted for fraud or refunds > $500 (per policy).
HARD_SCENARIO = Scenario(
    name="hard",
    ticket_id="TICKET-003",
    difficulty="hard",
    max_steps=12,  # 7 required actions + 5 slack steps for a complex dual-issue ticket
    customer_message=(
        "I received the wrong item in order ORD-3021 (got blue instead of red), "
        "AND I was overcharged on order ORD-3022 by $15."
    ),
    # Both orders must be looked up — the agent cannot resolve either issue without
    # knowing the order details (product ID, customer ID, amounts charged).
    required_orders=["ORD-3021", "ORD-3022"],
    required_actions=[
        "lookup_order",       # must be called for both orders
        "lookup_customer",    # needed to verify identity before any action
        "check_policy",       # must check replacement and/or refund policy
        "check_inventory",    # must verify stock before promising a replacement
        "send_replacement",   # attempt replacement (even if stock is 0, the attempt is graded)
        "issue_refund",       # partial refund of $15 for the billing error
        "send_response",      # unified response addressing both issues
    ],
    success_criteria={
        # Both order IDs must appear in the lookup history — a list check, not a single ID.
        "must_lookup_orders": ["ORD-3021", "ORD-3022"],
        "must_lookup_customer": True,
        # The agent must check inventory for the Red Mouse (PROD-003) before deciding
        # whether to send a replacement. This is the product from ORD-3021.
        "must_check_inventory": "PROD-003",
        # A refund must be issued for the billing error on ORD-3022.
        "must_issue_refund": True,
        # The overcharge was exactly $15.00 — the agent must compute this correctly
        # (314.99 charged − 299.99 correct price = 15.00). A full refund would be wrong.
        "correct_refund_amount": 15.00,
        "must_send_response": True,
        # Escalation is NOT required here — both issues are routine (wrong item + billing
        # error, both well under the $500 escalation threshold). An agent that escalates
        # anyway is penalised because it wastes a human agent's time unnecessarily.
        "no_unnecessary_escalation": True,
    },
)


# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------

# A flat dict makes scenario lookup O(1) and keeps get_scenario() simple.
# The keys match the valid values of the SUPPORT_TICKET_TASK environment variable.
SCENARIOS: dict[str, Scenario] = {
    "easy": EASY_SCENARIO,
    "medium": MEDIUM_SCENARIO,
    "hard": HARD_SCENARIO,
}


# ---------------------------------------------------------------------------
# Public accessor
# ---------------------------------------------------------------------------

def get_scenario(task: str) -> Scenario:
    """Return the Scenario for the given task name.

    This is the canonical way to retrieve a scenario. It validates the task name
    and raises a clear error for unknown values, which surfaces misconfiguration
    at startup rather than silently returning wrong results at request time.

    Args:
        task: The task name to look up. Must be one of "easy", "medium", or "hard".
            Typically sourced from the SUPPORT_TICKET_TASK environment variable.

    Returns:
        The Scenario instance corresponding to the given task name.

    Raises:
        ValueError: If `task` is not a recognised scenario name. The error message
            includes the valid options so the caller can fix the configuration
            without consulting the source code.
    """
    scenario = SCENARIOS.get(task)
    if scenario is None:
        valid = ", ".join(f'"{k}"' for k in SCENARIOS)
        raise ValueError(
            f"Unknown task name: {task!r}. "
            f"Valid options are: {valid}."
        )
    return scenario
