"""
server/environment.py — Core episode logic for the SupportTicket RL environment.

This module is the heart of the environment. It owns the episode lifecycle:
  reset() → step() → step() → ... → step() (done=True)
"""

from __future__ import annotations

import uuid
from typing import Any

try:
    from .data import BackendData
    from .scenarios import Scenario, get_scenario
    from .rewards import compute_reward
    from .graders import grade
except ImportError:
    from data import BackendData          # type: ignore[no-reuse-def]
    from scenarios import Scenario, get_scenario  # type: ignore[no-reuse-def]
    from rewards import compute_reward    # type: ignore[no-reuse-def]
    from graders import grade             # type: ignore[no-reuse-def]

try:
    from support_ticket_env.models import (
        SupportTicketAction,
        SupportTicketObservation,
        VALID_COMMANDS,
    )
except ImportError:
    from models import SupportTicketAction, SupportTicketObservation, VALID_COMMANDS  # type: ignore[no-reuse-def]


class SupportTicketEnvironment:
    """Manages the full lifecycle of a support-ticket resolution episode."""

    def __init__(self, task_id: str = "easy") -> None:
        self._default_task_id = task_id
        get_scenario(task_id)  # validate at startup
        self._canonical_backend = BackendData()
        self._episode_id: str | None = None
        self._scenario: Scenario | None = None
        self._backend: BackendData | None = None
        self._step_count: int = 0
        self._action_history: list[dict[str, Any]] = []
        self._context: dict[str, Any] = {}
        self._done: bool = False

    def reset(self, task_id: str | None = None) -> SupportTicketObservation:
        """Start a new episode."""
        effective_task = task_id or self._default_task_id
        self._scenario = get_scenario(effective_task)
        self._episode_id = str(uuid.uuid4())
        self._backend = self._canonical_backend.reset()
        self._step_count = 0
        self._action_history = []
        self._context = {}
        self._done = False
        return self._build_observation(last_action_result=None, last_action_error=None)

    def step(self, action: SupportTicketAction) -> tuple[SupportTicketObservation, float, bool]:
        """Execute one action and return (observation, reward, done)."""
        if self._scenario is None or self._backend is None:
            raise RuntimeError("step() called before reset()")

        if self._done:
            obs = self._build_observation(
                last_action_result=None,
                last_action_error="Episode is already done. Call reset() to start a new episode.",
            )
            return obs, 0.0, True

        # Stage 1: validate command
        if action.command not in VALID_COMMANDS:
            reward = compute_reward(
                command=action.command, parameters=action.parameters,
                result=None, error=f"Unknown command: {action.command!r}",
                scenario=self._scenario, action_history=self._action_history,
                backend=self._backend,
            )
            obs = self._build_observation(
                last_action_result=None,
                last_action_error=f"Unknown command: {action.command!r}. Valid commands: {VALID_COMMANDS}",
            )
            return obs, reward, False

        # Stage 2: increment step
        self._step_count += 1

        # Stage 3: execute
        result, error = self._execute_command(action.command, action.parameters)

        # Stage 4: reward
        reward = compute_reward(
            command=action.command, parameters=action.parameters,
            result=result, error=error,
            scenario=self._scenario, action_history=self._action_history,
            backend=self._backend,
        )

        # Stage 5: update context
        if result is not None:
            self._update_context(action.command, action.parameters, result)

        # Stage 6: record history
        self._action_history.append({
            "command": action.command, "parameters": action.parameters,
            "message": action.message, "result": result, "error": error,
            "step": self._step_count,
        })

        # Stage 7: check termination
        terminal_commands = {"send_response", "escalate"}
        self._done = action.command in terminal_commands or self._step_count >= self._scenario.max_steps

        # Stage 8: grade if done
        if self._done:
            final_score = grade(
                task_id=self._scenario.name,
                action_history=self._action_history,
                context=self._context,
            )
            self._context["score"] = final_score
            self._context["episode_id"] = self._episode_id

        # Stage 9: build observation
        obs = self._build_observation(last_action_result=result, last_action_error=error)
        return obs, reward, self._done

    def get_state(self) -> dict[str, Any]:
        """Return current episode state."""
        return {
            "episode_id": self._episode_id or "",
            "step_count": self._step_count,
        }

    def _execute_command(self, command: str, parameters: dict[str, Any]) -> tuple[dict | None, str | None]:
        if command == "lookup_order":
            order_id = parameters.get("order_id")
            if not order_id:
                return None, "Missing required parameter: order_id"
            record = self._backend.get_order(order_id)
            if record is None:
                return None, f"Order not found: {order_id!r}"
            return {"order": record}, None

        if command == "lookup_customer":
            customer_id = parameters.get("customer_id")
            if not customer_id:
                return None, "Missing required parameter: customer_id"
            record = self._backend.get_customer(customer_id)
            if record is None:
                return None, f"Customer not found: {customer_id!r}"
            return {"customer": record}, None

        if command == "check_policy":
            policy_type = parameters.get("policy_type")
            if not policy_type:
                return None, "Missing required parameter: policy_type"
            record = self._backend.get_policy(policy_type)
            if record is None:
                return None, f"Policy not found: {policy_type!r}"
            return {"policy": record}, None

        if command == "check_inventory":
            product_id = parameters.get("product_id")
            if not product_id:
                return None, "Missing required parameter: product_id"
            stock = self._backend.check_stock(product_id)
            if stock == -1:
                return None, f"Product not found: {product_id!r}"
            return {"product_id": product_id, "stock_count": stock, "in_stock": stock > 0}, None

        if command == "issue_refund":
            order_id = parameters.get("order_id")
            amount = parameters.get("amount")
            reason = parameters.get("reason", "")
            if not order_id:
                return None, "Missing required parameter: order_id"
            if amount is None:
                return None, "Missing required parameter: amount"
            try:
                amount_float = float(amount)
            except (TypeError, ValueError):
                return None, f"Invalid amount: {amount!r}"
            success = self._backend.apply_refund(order_id, amount_float)
            if not success:
                return None, f"Order not found: {order_id!r}"
            return {"status": "refund_processed", "order_id": order_id, "amount": amount_float, "reason": reason}, None

        if command == "send_replacement":
            order_id = parameters.get("order_id")
            product_id = parameters.get("product_id")
            if not order_id:
                return None, "Missing required parameter: order_id"
            if not product_id:
                return None, "Missing required parameter: product_id"
            stock = self._backend.check_stock(product_id)
            if stock == 0:
                return {"status": "replacement_unavailable", "order_id": order_id, "product_id": product_id, "reason": "Out of stock"}, None
            success = self._backend.apply_replacement(order_id, product_id)
            if not success:
                return None, f"Order or product not found"
            return {"status": "replacement_sent", "order_id": order_id, "product_id": product_id}, None

        if command == "escalate":
            reason = parameters.get("reason", "No reason provided")
            priority = parameters.get("priority", "normal")
            return {"status": "escalated", "reason": reason, "priority": priority,
                    "ticket_id": self._scenario.ticket_id if self._scenario else "UNKNOWN"}, None

        if command == "send_response":
            message = parameters.get("message", "")
            if not message:
                return None, "Missing required parameter: message (send_response requires a message)"
            return {"status": "response_sent", "message": message}, None

        return None, f"Unhandled command: {command!r}"

    def _update_context(self, command: str, parameters: dict[str, Any], result: dict[str, Any]) -> None:
        if command == "lookup_order":
            self._context[f"order_{parameters.get('order_id', 'unknown')}"] = result.get("order")
        elif command == "lookup_customer":
            self._context[f"customer_{parameters.get('customer_id', 'unknown')}"] = result.get("customer")
        elif command == "check_policy":
            self._context[f"policy_{parameters.get('policy_type', 'unknown')}"] = result.get("policy")
        elif command == "check_inventory":
            self._context[f"inventory_{parameters.get('product_id', 'unknown')}"] = {"stock_count": result.get("stock_count"), "in_stock": result.get("in_stock")}
        elif command == "issue_refund":
            self._context["refund_issued"] = {"order_id": result.get("order_id"), "amount": result.get("amount"), "reason": result.get("reason")}
        elif command == "send_replacement":
            self._context["replacement_status"] = {"order_id": result.get("order_id"), "product_id": result.get("product_id"), "status": result.get("status")}

    def _build_observation(self, last_action_result: dict | None, last_action_error: str | None) -> SupportTicketObservation:
        if self._scenario is None:
            raise RuntimeError("Cannot build observation before reset()")
        return SupportTicketObservation(
            ticket_id=self._scenario.ticket_id,
            customer_message=self._scenario.customer_message,
            step_number=self._step_count,
            last_action_result=last_action_result,
            last_action_error=last_action_error,
            available_commands=list(VALID_COMMANDS),
            context=dict(self._context),
            done=self._done,
        )
