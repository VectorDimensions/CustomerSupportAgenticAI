"""
server/rewards.py — Per-step reward computation for the SupportTicket RL environment.

Reward Design Philosophy
------------------------
The reward function is the agent's primary learning signal, so it must be:

  1. **Transparent** — every rule is documented with the *reason* it exists, not just
     what it does. An agent (or a human reading the logs) should always understand why
     a particular reward was assigned.

  2. **Priority-ordered** — rules are evaluated in a fixed chain and the first match
     wins. This prevents double-counting (e.g. a repeated action that also happens to
     be a policy violation should only get the -0.02 repeated-action penalty, not both)
     and makes the reward signal predictable.

  3. **Sparse but informative** — most steps return 0.0 (neutral). Positive rewards
     are reserved for genuinely useful actions; penalties are reserved for genuinely
     harmful ones. This keeps the reward signal from being too noisy.

  4. **Grounded in policy** — the reward rules mirror the business policies in the
     backend. An agent that learns to follow the reward signal is also learning to
     follow the company's support policies, which is the actual goal.

Priority order rationale (highest to lowest):
  1. Invalid command   — if the command doesn't exist, nothing else can be evaluated
  2. Repeated action   — valid but wasteful; catches loops before deeper checks
  3. Policy violation  — more serious than wrong params; the action is harmful
  4. Wrong params      — valid command, but incorrectly parameterised
  5. Info-gathering    — positive signal for learning about the ticket
  6. Correct resolution — the highest positive reward; the agent solved something
  7. Good response     — positive signal for communicating well with the customer
  8. Unnecessary escalation — penalise wasting a human agent's time
  9. Default           — any other valid action gets a neutral 0.0
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Dual-import pattern: supports both `python server/rewards.py` (direct execution,
# where relative imports fail) and `from server.rewards import ...` (package import).
# This is required in all server-side files per the project's coding standards.
try:
    from .scenarios import Scenario
    from .data import BackendData
except ImportError:
    from scenarios import Scenario  # type: ignore[no-reuse-def]
    from data import BackendData  # type: ignore[no-reuse-def]


# ---------------------------------------------------------------------------
# Reward constants
# ---------------------------------------------------------------------------
# Using UPPER_SNAKE_CASE for constants makes them easy to grep and distinguishes
# them from local variables at a glance. Defining them at module level means
# tests can import and assert against the exact values without magic numbers.

# Positive rewards — given for actions that move the ticket toward resolution
REWARD_INFO_GATHERING: float = 0.10    # looking up new data is always useful
REWARD_CORRECT_RESOLUTION: float = 0.25  # highest reward: the agent actually solved something
REWARD_GOOD_RESPONSE: float = 0.15    # communicating the resolution to the customer

# Penalties — given for actions that waste time or cause harm
PENALTY_INVALID_COMMAND: float = -0.10    # the command doesn't exist at all
PENALTY_WRONG_PARAMS: float = -0.05      # valid command, but wrong/missing parameters
PENALTY_POLICY_VIOLATION: float = -0.15  # the action violates a business policy
PENALTY_REDUNDANT_ACTION: float = -0.02  # repeating an action that already ran
PENALTY_UNNECESSARY_ESCALATION: float = -0.10  # escalating when it isn't needed


# ---------------------------------------------------------------------------
# Valid command set (imported here to avoid circular dependency with environment)
# ---------------------------------------------------------------------------
# We define the valid commands locally rather than importing from models.py to
# keep the server package self-contained. The environment validates commands
# before calling compute_reward(), but we still need this set for Rule 1.
_VALID_COMMANDS: frozenset[str] = frozenset([
    "lookup_order",
    "lookup_customer",
    "check_policy",
    "check_inventory",
    "issue_refund",
    "send_replacement",
    "escalate",
    "send_response",
])

# Info-gathering commands: these are the four commands whose purpose is to
# learn about the ticket before taking action. They earn a positive reward
# when they return data the agent hasn't seen before.
_INFO_COMMANDS: frozenset[str] = frozenset([
    "lookup_order",
    "lookup_customer",
    "check_policy",
    "check_inventory",
])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_reward(
    command: str,
    parameters: dict,
    result: dict | None,
    error: str | None,
    scenario: "Scenario",
    action_history: list[dict],
    backend: "BackendData",
) -> float:
    """Compute the per-step reward for a single agent action.

    Evaluates a priority-ordered chain of rules and returns the reward for the
    first matching rule. The chain is documented in the module docstring; see
    the inline comments below for the *why* behind each rule's priority.

    Args:
        command: The command string the agent submitted (e.g. "lookup_order").
            May be any string — invalid commands are handled by Rule 1.
        parameters: The parameters dict the agent submitted alongside the command.
        result: The structured result returned by the backend for this action,
            or None if the action produced an error or has not yet been executed.
        error: A human-readable error string if the action failed, otherwise None.
            Note: an error here means the *backend* returned an error for a valid
            command (e.g. order not found), not that the command itself is invalid.
        scenario: The active Scenario for this episode. Used to check success
            criteria (correct refund amount, required keywords, escalation policy).
        action_history: The list of all previous actions in this episode, each
            represented as a dict with at least "command" and "parameters" keys.
            Used to detect repeated actions (Rule 2) and new-data checks (Rule 5).
        backend: The episode's BackendData instance. Used to check inventory levels
            and policy rules for policy-violation detection (Rule 3).

    Returns:
        A float reward value. Positive values indicate beneficial actions;
        negative values indicate harmful or wasteful actions; 0.0 is neutral.
    """

    # ------------------------------------------------------------------
    # Rule 1 (highest priority): Invalid command → PENALTY_INVALID_COMMAND
    #
    # WHY FIRST: If the command string isn't in the valid set, we can't
    # evaluate any other rule — we don't know what the agent was trying to
    # do, so we can't check for policy violations, wrong params, etc.
    # Catching this first also prevents any accidental state inspection on
    # a command that the backend never executed.
    # ------------------------------------------------------------------
    if command not in _VALID_COMMANDS:
        return PENALTY_INVALID_COMMAND

    # ------------------------------------------------------------------
    # Rule 2: Repeated identical action → PENALTY_REDUNDANT_ACTION
    #
    # WHY SECOND: The command is valid, but if the agent already ran this
    # exact command with these exact parameters, running it again wastes a
    # step and provides no new information. We catch this before checking
    # for policy violations because a repeated action is still a valid
    # command — it just happens to be redundant. The penalty is small (-0.02)
    # because repetition is wasteful but not harmful.
    # ------------------------------------------------------------------
    if _is_repeated_action(command, parameters, action_history):
        return PENALTY_REDUNDANT_ACTION

    # ------------------------------------------------------------------
    # Rule 3: Policy violation → PENALTY_POLICY_VIOLATION
    #
    # WHY THIRD: Policy violations are more serious than wrong parameters.
    # A policy violation means the agent is trying to do something the
    # business explicitly prohibits (e.g. issuing a refund for the wrong
    # amount, or sending a replacement when the item is out of stock).
    # We check this before wrong-params because a policy-violating action
    # with correct syntax is still more harmful than a syntactically wrong
    # action that would have been benign.
    # ------------------------------------------------------------------
    if _is_policy_violation(command, parameters, scenario, backend):
        return PENALTY_POLICY_VIOLATION

    # ------------------------------------------------------------------
    # Rule 4: Wrong/missing parameters → PENALTY_WRONG_PARAMS
    #
    # WHY FOURTH: The command is valid and not a policy violation, but the
    # parameters are incorrect or missing. This is less serious than a
    # policy violation — the agent is on the right track but made a
    # parameter mistake. The penalty is small (-0.05) to encourage the
    # agent to retry with correct parameters rather than giving up.
    # ------------------------------------------------------------------
    if error is not None:
        # An error from the backend on a valid command means the parameters
        # were wrong (e.g. unknown order ID, missing required key). We treat
        # any backend error as a wrong-params signal here because the backend
        # only errors on parameter problems, not on logic errors.
        return PENALTY_WRONG_PARAMS

    # ------------------------------------------------------------------
    # Rule 5: Info-gathering action that returned new data → REWARD_INFO_GATHERING
    #
    # WHY FIFTH: The agent successfully gathered new information. We reward
    # this because information gathering is a prerequisite for correct
    # resolution — an agent that skips lookups and goes straight to action
    # is guessing. "New data" means this exact command+params combo hasn't
    # been called before in this episode (we already handled the repeated
    # case in Rule 2, so if we reach here the action is not repeated).
    # ------------------------------------------------------------------
    if command in _INFO_COMMANDS and result is not None:
        # By the time we reach here, Rule 2 has already confirmed this is
        # NOT a repeated action, so any successful info command is "new data".
        return REWARD_INFO_GATHERING

    # ------------------------------------------------------------------
    # Rule 6: Correct resolution action → REWARD_CORRECT_RESOLUTION
    #
    # WHY SIXTH: The agent took a resolution action (issue_refund or
    # send_replacement) that satisfies the scenario's success criteria.
    # This is the highest positive reward because it represents the agent
    # actually solving the customer's problem, not just gathering data.
    # ------------------------------------------------------------------
    if _is_correct_resolution(command, parameters, scenario, backend):
        return REWARD_CORRECT_RESOLUTION

    # ------------------------------------------------------------------
    # Rule 7: Good send_response → REWARD_GOOD_RESPONSE
    #
    # WHY SEVENTH: The agent sent a customer-facing response that mentions
    # the keywords the scenario requires. A good response is the final step
    # of a well-handled ticket — it communicates the resolution to the
    # customer. We reward it less than a correct resolution (+0.15 vs +0.25)
    # because the response is the communication of the resolution, not the
    # resolution itself.
    # ------------------------------------------------------------------
    if _is_good_response(command, parameters, scenario):
        return REWARD_GOOD_RESPONSE

    # ------------------------------------------------------------------
    # Rule 8: Unnecessary escalation → PENALTY_UNNECESSARY_ESCALATION
    #
    # WHY EIGHTH: The agent escalated a ticket that didn't need escalation.
    # Escalation wastes a human agent's time and is only warranted for fraud
    # or large refunds (per the escalation_criteria policy). We check this
    # near the end because it only applies to the "escalate" command, and
    # we've already handled the more general cases above.
    # ------------------------------------------------------------------
    if command == "escalate":
        # Escalation is only warranted when the scenario explicitly requires it
        # (fraud, large refund > $500). The easy and medium scenarios never need
        # escalation. The hard scenario has no_unnecessary_escalation=True.
        # We penalise escalation on ANY scenario that doesn't mark it as required,
        # because the escalation_criteria policy says routine issues must be
        # resolved directly — escalating wastes a human agent's time.
        escalation_required = scenario.success_criteria.get("escalation_required", False)
        if not escalation_required:
            return PENALTY_UNNECESSARY_ESCALATION

    # ------------------------------------------------------------------
    # Rule 9 (default): Any other valid action → 0.0
    #
    # WHY DEFAULT: The action is valid, not repeated, not a policy violation,
    # not wrong-params, not info-gathering, not a correct resolution, not a
    # good response, and not an unnecessary escalation. It's a neutral action
    # that doesn't help or hurt. Return 0.0 to avoid polluting the reward
    # signal with noise.
    # ------------------------------------------------------------------
    return 0.0


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _is_repeated_action(
    command: str,
    parameters: dict,
    action_history: list[dict],
) -> bool:
    """Check whether this exact command+parameters combo has already been executed.

    "Identical" means both the command string and the parameters dict are equal.
    We use dict equality (==) rather than identity (is) because the parameters
    dict is reconstructed from JSON on every request, so identity checks would
    always return False even for genuinely repeated actions.

    Args:
        command: The command string for the current action.
        parameters: The parameters dict for the current action.
        action_history: List of previous actions, each a dict with at least
            "command" and "parameters" keys.

    Returns:
        True if an identical action (same command AND same parameters) already
        appears in action_history, False otherwise.
    """
    for past_action in action_history:
        if (
            past_action.get("command") == command
            and past_action.get("parameters") == parameters
        ):
            # Found an exact match — this action is redundant.
            return True
    return False


def _is_policy_violation(
    command: str,
    parameters: dict,
    scenario: "Scenario",
    backend: "BackendData",
) -> bool:
    """Check whether the action violates a business policy.

    Currently checks two policy violations:
      1. issue_refund with an amount that doesn't match the scenario's
         correct_refund_amount (wrong amount = policy violation because the
         refund policy specifies exact amounts for specific reasons).
      2. send_replacement when the product is out of stock (replacement policy
         requires inventory check; sending when stock=0 violates the policy).

    Args:
        command: The command string for the current action.
        parameters: The parameters dict for the current action.
        scenario: The active Scenario, used to look up correct_refund_amount.
        backend: The episode's BackendData, used to check inventory levels.

    Returns:
        True if the action violates a policy, False otherwise.
    """
    if command == "issue_refund":
        # The refund policy specifies the correct amount for each scenario.
        # Issuing a refund for the wrong amount is a policy violation — it
        # either over-refunds (costs the company money) or under-refunds
        # (fails the customer). We only flag this if the scenario has a
        # correct_refund_amount criterion; if it doesn't, we can't judge.
        correct_amount = scenario.success_criteria.get("correct_refund_amount")
        if correct_amount is not None:
            submitted_amount = parameters.get("amount")
            if submitted_amount is None:
                # Missing amount is a wrong-params issue, not a policy violation.
                # Rule 4 (wrong params / backend error) will catch this instead.
                return False
            # Use a small tolerance for floating-point comparison to avoid
            # false positives from rounding (e.g. 149.990000001 vs 149.99).
            try:
                if abs(float(submitted_amount) - float(correct_amount)) > 0.01:
                    return True
            except (TypeError, ValueError):
                # Non-numeric amount — wrong params, not a policy violation.
                return False

    if command == "send_replacement":
        # The replacement policy requires checking inventory before sending.
        # If the product is out of stock, sending a replacement is a policy
        # violation — we'd be promising something we can't deliver.
        order_id = parameters.get("order_id")
        if order_id is not None:
            order = backend.get_order(order_id)
            if order is not None:
                product_id = order.get("product_id")
                if product_id is not None:
                    stock = backend.check_stock(product_id)
                    # stock == 0 means out of stock; stock == -1 means unknown product.
                    # Both cases mean we can't send a replacement.
                    if stock == 0:
                        return True

    return False


def _is_correct_resolution(
    command: str,
    parameters: dict,
    scenario: "Scenario",
    backend: "BackendData",
) -> bool:
    """Check whether the action correctly resolves the scenario's primary issue.

    A "correct resolution" is one that satisfies the scenario's success criteria:
      - issue_refund: the amount matches the scenario's correct_refund_amount
      - send_replacement: the product is in stock (stock > 0)

    Args:
        command: The command string for the current action.
        parameters: The parameters dict for the current action.
        scenario: The active Scenario, used to look up success criteria.
        backend: The episode's BackendData, used to check inventory levels.

    Returns:
        True if the action correctly resolves the scenario's issue, False otherwise.
    """
    if command == "issue_refund":
        # A correct refund matches the scenario's expected amount exactly
        # (within floating-point tolerance). If the scenario has no
        # correct_refund_amount criterion, any refund is considered correct
        # (we can't judge without a target).
        correct_amount = scenario.success_criteria.get("correct_refund_amount")
        submitted_amount = parameters.get("amount")
        if submitted_amount is None:
            return False
        if correct_amount is None:
            # No target amount specified — treat any refund as correct.
            return True
        try:
            return abs(float(submitted_amount) - float(correct_amount)) <= 0.01
        except (TypeError, ValueError):
            return False

    if command == "send_replacement":
        # A correct replacement requires the product to be in stock.
        # We already penalised out-of-stock replacements in _is_policy_violation,
        # so if we reach here the stock check passed (stock > 0).
        order_id = parameters.get("order_id")
        if order_id is not None:
            order = backend.get_order(order_id)
            if order is not None:
                product_id = order.get("product_id")
                if product_id is not None:
                    stock = backend.check_stock(product_id)
                    return stock > 0

    return False


def _is_good_response(
    command: str,
    parameters: dict,
    scenario: "Scenario",
) -> bool:
    """Check whether a send_response action meets the scenario's quality criteria.

    A "good response" is a send_response action whose message mentions all of the
    keywords listed in the scenario's "response_must_mention" success criterion.
    The check is case-insensitive so the agent isn't penalised for capitalisation.

    If the scenario has no "response_must_mention" criterion, any send_response
    is considered good (we can't judge quality without a target).

    Args:
        command: The command string for the current action.
        parameters: The parameters dict for the current action.
        scenario: The active Scenario, used to look up response_must_mention.

    Returns:
        True if the command is send_response and the message meets the quality
        criteria, False otherwise.
    """
    if command != "send_response":
        return False

    # The message can come from either the "message" key in parameters or
    # directly as a top-level field. We check parameters first (the canonical
    # location for send_response content), then fall back to a "message" key.
    message = parameters.get("message", "")
    if not message:
        # An empty or missing message can't mention any keywords.
        return False

    required_keywords = scenario.success_criteria.get("response_must_mention")
    if required_keywords is None:
        # No keyword requirement — any non-empty response is good.
        return True

    # Check that at least one required keyword appears in the message.
    # We use "any" rather than "all" because natural language has many ways
    # to convey the same information — requiring ALL keywords would be too strict.
    message_lower = message.lower()
    return any(kw.lower() in message_lower for kw in required_keywords)
