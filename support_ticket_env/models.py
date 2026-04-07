"""
support_ticket_env/models.py — Pydantic v2 data-transfer objects for the support-ticket-env.

Role in the OpenEnv architecture
---------------------------------
This module sits at the boundary between the agent (client side) and the environment
(server side). Every Action the agent sends and every Observation the environment returns
is serialised/deserialised through these two models.

Keeping the models in the *client* package (support_ticket_env/) rather than the server
means the agent code can import them without pulling in FastAPI or any server-side
dependencies. The server imports them too — Python's import system resolves the same
source file regardless of which side initiates the import.

Why Pydantic v2?
  - model_dump_json() / model_validate_json() give us fast, zero-boilerplate JSON
    round-trips that are required by the OpenEnv HTTP contract.
  - Field(description=...) makes the JSON Schema self-documenting, which is useful
    when the OpenEnv SDK introspects the schema to generate API docs.
  - Pydantic v2's Rust-backed core is significantly faster than v1 for the tight
    step-loop that RL training runs.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# VALID_COMMANDS
# ---------------------------------------------------------------------------
# This constant is the single source of truth for the action space.
# Defining it at module level (rather than inside the model) means both the
# environment and the client can import it without creating a circular dependency,
# and property tests can sample from it directly.
#
# The eight commands cover the full lifecycle of a support ticket:
#   - Information gathering: lookup_order, lookup_customer, check_policy, check_inventory
#   - Resolution actions:    issue_refund, send_replacement
#   - Terminal actions:      escalate, send_response
VALID_COMMANDS: list[str] = [
    "lookup_order",
    "lookup_customer",
    "check_policy",
    "check_inventory",
    "issue_refund",
    "send_replacement",
    "escalate",
    "send_response",
]


# ---------------------------------------------------------------------------
# SupportTicketAction
# ---------------------------------------------------------------------------

class SupportTicketAction(BaseModel):
    """An action submitted by the agent to the support-ticket environment.

    An Action is the agent's turn in the Action–Observation loop. It names a
    command from the fixed action space and supplies any parameters that command
    requires. The optional ``message`` field carries free-text content used by
    the ``send_response`` and ``escalate`` commands.

    Args:
        command: One of the eight valid command strings defined in VALID_COMMANDS.
            The validator lowercases the value before storage so that agents
            sending "Lookup_Order" still work. Invalid commands are *not* rejected
            here — the environment handles them and returns a descriptive error
            observation. This keeps the model lenient at parse time and centralises
            validation logic in one place (the environment).
        parameters: A dictionary of command-specific key/value pairs. For example,
            ``lookup_order`` expects ``{"order_id": "ORD-1042"}``. Defaults to an
            empty dict so that commands with no required parameters (e.g. a future
            ``list_orders``) don't force callers to pass ``{}``.
        message: Optional free-text string. Used by ``send_response`` to carry the
            customer-facing reply and by ``escalate`` to explain the reason for
            escalation. None for all other commands.

    Returns:
        A validated SupportTicketAction instance ready for serialisation.
    """

    command: str = Field(
        ...,
        description=(
            "The command to execute. Must be one of the eight valid command strings: "
            + ", ".join(VALID_COMMANDS)
            + ". The value is lowercased automatically before storage."
        ),
    )

    parameters: dict[str, Any] = Field(
        default={},
        description=(
            "Command-specific parameters as a string-keyed dictionary. "
            "For example, lookup_order requires {'order_id': 'ORD-1042'}. "
            "Defaults to an empty dict for commands that take no parameters."
        ),
    )

    message: str | None = Field(
        default=None,
        description=(
            "Optional free-text message. Used by send_response to carry the "
            "customer-facing reply, and by escalate to explain the escalation reason. "
            "Should be None for all other commands."
        ),
    )

    # Why a soft validator instead of a hard Literal type?
    # Using Literal["lookup_order", ...] would cause Pydantic to raise a
    # ValidationError for unknown commands before the environment ever sees them.
    # That would break the OpenEnv contract, which requires the environment to
    # return an error *observation* (not an HTTP 422) for invalid commands.
    # The validator only lowercases — the environment is responsible for the
    # "is this command valid?" check.
    @field_validator("command", mode="before")
    @classmethod
    def lowercase_command(cls, v: Any) -> str:
        """Lowercase the command string so comparisons are case-insensitive."""
        if isinstance(v, str):
            return v.lower()
        # Non-string values pass through unchanged; Pydantic's type coercion
        # will handle (or reject) them in the normal validation pipeline.
        return v  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# SupportTicketObservation
# ---------------------------------------------------------------------------

class SupportTicketObservation(BaseModel):
    """An observation returned by the environment after each step.

    An Observation is the environment's response to an Action. It gives the agent
    everything it needs to decide the next action: the ticket context, the result
    (or error) of the last action, the current step counter, and a ``done`` flag
    that signals episode termination.

    Args:
        ticket_id: The unique identifier of the support ticket being resolved
            (e.g. "TICKET-001"). Constant across all steps of an episode.
        customer_message: The original customer complaint or inquiry text.
            Constant across all steps of an episode.
        step_number: 1-based index of the current step within the episode.
            Increments by 1 on every successful (non-error) action.
        last_action_result: The structured result returned by the last command,
            or None if no action has been taken yet (initial observation) or if
            the last action produced an error.
        last_action_error: A human-readable error string if the last action was
            invalid or failed, otherwise None. The agent should inspect this field
            to detect and recover from mistakes.
        available_commands: The complete list of valid command strings. Always
            equal to VALID_COMMANDS — included in every observation so the agent
            never needs to hard-code the action space.
        context: A dictionary that accumulates episode-level state visible to the
            agent. The environment writes intermediate results here (e.g. looked-up
            order data) so the agent can reference them without re-querying. On the
            final step, the grader score is also embedded here under the key "score".
        done: True when the episode has terminated (terminal action executed or
            maximum step limit reached), False otherwise. The agent loop should
            stop calling step() once done is True.

    Returns:
        A validated SupportTicketObservation instance ready for serialisation.
    """

    ticket_id: str = Field(
        ...,
        description=(
            "Unique identifier for the support ticket being resolved. "
            "Constant across all steps of an episode (e.g. 'TICKET-001')."
        ),
    )

    customer_message: str = Field(
        ...,
        description=(
            "The original customer complaint or inquiry text. "
            "Constant across all steps of an episode."
        ),
    )

    step_number: int = Field(
        ...,
        description=(
            "1-based index of the current step within the episode. "
            "Increments by 1 on every successful action."
        ),
    )

    last_action_result: dict | None = Field(
        default=None,
        description=(
            "Structured result returned by the last command, or None if no action "
            "has been taken yet or if the last action produced an error."
        ),
    )

    last_action_error: str | None = Field(
        default=None,
        description=(
            "Human-readable error string if the last action was invalid or failed, "
            "otherwise None. Inspect this field to detect and recover from mistakes."
        ),
    )

    available_commands: list[str] = Field(
        ...,
        description=(
            "The complete list of valid command strings available to the agent. "
            "Always equal to VALID_COMMANDS — included in every observation so the "
            "agent never needs to hard-code the action space."
        ),
    )

    context: dict = Field(
        default={},
        description=(
            "Episode-level state dictionary visible to the agent. The environment "
            "accumulates intermediate results here (e.g. looked-up order data). "
            "On the final step, the grader score is embedded under the key 'score'."
        ),
    )

    done: bool = Field(
        ...,
        description=(
            "True when the episode has terminated (terminal action executed or "
            "maximum step limit reached), False otherwise."
        ),
    )


# ---------------------------------------------------------------------------
# pretty_print helper
# ---------------------------------------------------------------------------

def pretty_print(obj: SupportTicketAction | SupportTicketObservation) -> str:
    """Return a human-readable, indented JSON string for a model instance.

    Uses Pydantic's model_dump_json() rather than json.dumps(obj.model_dump())
    because model_dump_json() handles non-JSON-serialisable types (e.g. datetime,
    UUID) that might appear in the ``parameters`` or ``context`` dicts, whereas
    json.dumps() would raise a TypeError for those types.

    Args:
        obj: A SupportTicketAction or SupportTicketObservation instance.

    Returns:
        A valid JSON string with 2-space indentation.
    """
    # model_dump_json(indent=2) is the idiomatic Pydantic v2 way to get pretty JSON.
    # We don't use json.dumps(json.loads(...)) as a two-step round-trip because
    # that would lose any custom serialisers registered on the model.
    return obj.model_dump_json(indent=2)
