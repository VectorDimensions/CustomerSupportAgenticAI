"""
server/environment.py — Core episode logic for the SupportTicket RL environment.

This module is the heart of the environment. It owns the episode lifecycle:
  reset() → step() → step() → ... → step() (done=True)

Architecture note
-----------------
SupportTicketEnvironment is intentionally a plain Python class rather than a
FastAPI router or a Pydantic model. Keeping the core logic framework-agnostic
means it can be unit-tested without spinning up an HTTP server, and it can be
wrapped by different server frameworks in the future without rewriting the logic.

The FastAPI wiring lives in app.py; this class only knows about episodes.

Step sequence (why this order?)
--------------------------------
1. Validate command        — fail fast: if the command is unknown, nothing else matters
2. Check for repeat        — detect loops before touching the backend
3. Execute command          — query or mutate the backend
4. Compute reward           — reward depends on execution result, so it comes after
5. Update context           — store results for the agent to reference later
6. Record action history    — must happen after execution so the result is available
7. Check termination        — check done *after* recording so the final action is in history
8. Grade if done            — grader needs the complete history, so it runs last
9. Build observation        — assemble everything into the response object

This order ensures that each step has access to all the information it needs
without requiring lookahead or backtracking.
"""

from __future__ import annotations

import uuid
from typing import Any

# Dual-import pattern: works both when running directly (bare imports) and
# when imported as part of the server package (relative imports).
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

# The client-side models live outside the server package, so we use a try/except
# that covers both the installed-package path and the in-repo development path.
try:
    from support_ticket_env.models import (
        SupportTicketAction,
        SupportTicketObservation,
        VALID_COMMANDS,
    )
except ImportError:
    from models import SupportTicketAction, SupportTicketObservation, VALID_COMMANDS  # type: ignore[no-reuse-def]


# ---------------------------------------------------------------------------
# SupportTicketEnvironment
# ---------------------------------------------------------------------------

class SupportTicketEnvironment:
    """Manages the full lifecycle of a support-ticket resolution episode.

    One instance of this class is created per server process and shared across
    all requests. Episode state is stored on the instance, so the server is
    effectively single-session (one active episode at a time). This is fine for
    the hackathon use case where a single agent runs one episode at a time.

    If multi-session support is needed in the future, the episode state could be
    moved into a session dict keyed by episode_id.

    Args:
        task_id: The task to load on reset. Defaults to "easy". Can be overridden
            per-reset via the reset() method's task_id parameter.

    Returns:
        A SupportTicketEnvironment instance ready to accept reset() calls.
    """

    def __init__(self, task_id: str = "easy") -> None:
        """Initialise the environment with a default task and a fresh backend.

        We validate the task_id at construction time (fail fast) so that a
        misconfigured server fails immediately on startup rather than on the
        first request — much easier to debug.

        Args:
            task_id: Default task name. Must be "easy", "medium", or "hard".
        """
        # Validate the task at startup so misconfiguration surfaces immediately.
        # get_scenario() raises ValueError for unknown task names.
        self._default_task_id = task_id
        get_scenario(task_id)  # validation only — we don't store the result yet

        # The canonical backend is constructed once and deep-copied on each reset.
        # Constructing it here (not in reset()) means the hard-coded data is only
        # parsed once per server process, not once per episode.
        self._canonical_backend = BackendData()

        # Episode state — initialised to None/empty until reset() is called.
        self._episode_id: str | None = None
        self._scenario: Scenario | None = None
        self._backend: BackendData | None = None
        self._step_count: int = 0
        self._action_history: list[dict[str, Any]] = []
        self._context: dict[str, Any] = {}
        self._done: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, task_id: str | None = None) -> SupportTicketObservation:
        """Start a new episode, resetting all state to a clean baseline.

        Why deep-copy the backend on every reset?
            The backend is mutable — apply_refund() and apply_replacement() modify
            order and product records. If we reused the same backend instance across
            episodes, a refund applied in episode N would still be visible in episode
            N+1. Deep copy guarantees episode isolation.

        Args:
            task_id: The task to load for this episode. If None, uses the default
                task_id passed to __init__. Must be "easy", "medium", or "hard".

        Returns:
            The initial SupportTicketObservation with the customer message,
            step_number=0, empty context, and done=False.
        """
        effective_task = task_id or self._default_task_id

        # Load the scenario first — this validates the task name and raises
        # ValueError early if it's wrong, before we touch any episode state.
        self._scenario = get_scenario(effective_task)

        # Fresh episode state — everything resets to zero/empty.
        self._episode_id = str(uuid.uuid4())
        self._backend = self._canonical_backend.reset()  # deep copy for isolation
        self._step_count = 0
        self._action_history = []
        self._context = {}
        self._done = False

        # The initial observation has no action result yet — the agent hasn't
        # done anything. step_number=0 signals "before the first step".
        return self._build_observation(
            last_action_result=None,
            last_action_error=None,
        )

    def step(self, action: SupportTicketAction) -> tuple[SupportTicketObservation, float, bool]:
        """Execute one action and return (observation, reward, done).

        This method implements the full step sequence documented in the module
        docstring. Each stage is commented with the reasoning for its position
        in the sequence.

        Args:
            action: The SupportTicketAction submitted by the agent.

        Returns:
            A 3-tuple of:
              - SupportTicketObservation: the environment's response
              - float: the per-step reward
              - bool: whether the episode has terminated

        Raises:
            RuntimeError: If step() is called before reset() (no active episode).
        """
        if self._scenario is None or self._backend is None:
            raise RuntimeError(
                "step() called before reset(). Call reset() to start an episode first."
            )

        if self._done:
            # The episode is already over. Return the current observation with
            # zero reward rather than raising — this is more robust to agent
            # code that calls step() one extra time after done=True.
            obs = self._build_observation(
                last_action_result=None,
                last_action_error="Episode is already done. Call reset() to start a new episode.",
            )
            return obs, 0.0, True

        # ------------------------------------------------------------------
        # Stage 1: Validate the command
        #
        # WHY FIRST: If the command is unknown, we can't dispatch it, compute
        # a meaningful reward, or update context. Return an error observation
        # immediately without advancing any episode state. The step_count does
        # NOT increment for invalid commands — the agent gets another chance.
        # ------------------------------------------------------------------
        if action.command not in VALID_COMMANDS:
            reward = compute_reward(
                command=action.command,
                parameters=action.parameters,
                result=None,
                error=f"Unknown command: {action.command!r}",
                scenario=self._scenario,
                action_history=self._action_history,
                backend=self._backend,
            )
            obs = self._build_observation(
                last_action_result=None,
                last_action_error=f"Unknown command: {action.command!r}. Valid commands: {VALID_COMMANDS}",
            )
            return obs, reward, False

        # ------------------------------------------------------------------
        # Stage 2: Increment step count
        #
        # WHY HERE: We increment after command validation so invalid commands
        # don't consume a step. This gives the agent a chance to recover from
        # a typo without burning a step budget.
        # ------------------------------------------------------------------
        self._step_count += 1

        # ------------------------------------------------------------------
        # Stage 3: Execute the command against the backend
        #
        # _execute_command() returns (result_dict, error_str). Exactly one of
        # these will be non-None: result on success, error on failure.
        # ------------------------------------------------------------------
        result, error = self._execute_command(action.command, action.parameters)

        # ------------------------------------------------------------------
        # Stage 4: Compute the per-step reward
        #
        # WHY AFTER EXECUTION: The reward depends on the execution result
        # (e.g. whether the refund amount was correct, whether the info was
        # new). We can't compute it before we know what happened.
        # ------------------------------------------------------------------
        reward = compute_reward(
            command=action.command,
            parameters=action.parameters,
            result=result,
            error=error,
            scenario=self._scenario,
            action_history=self._action_history,
            backend=self._backend,
        )

        # ------------------------------------------------------------------
        # Stage 5: Update the accumulated context
        #
        # WHY AFTER REWARD: Context update is a side effect that doesn't affect
        # the reward for this step. Updating it here means the agent can see
        # the result in the *next* observation's context dict.
        # ------------------------------------------------------------------
        if result is not None:
            self._update_context(action.command, action.parameters, result)

        # ------------------------------------------------------------------
        # Stage 6: Record the action in history
        #
        # WHY AFTER EXECUTION: We store the result alongside the action so the
        # grader can inspect what each action returned without re-executing it.
        # ------------------------------------------------------------------
        self._action_history.append({
            "command": action.command,
            "parameters": action.parameters,
            "message": action.message,
            "result": result,
            "error": error,
            "step": self._step_count,
        })

        # ------------------------------------------------------------------
        # Stage 7: Check termination conditions
        #
        # The episode ends when:
        #   a) The agent calls a terminal action (send_response or escalate), OR
        #   b) The step count reaches the scenario's max_steps limit.
        #
        # WHY AFTER RECORDING: The final action must be in the history before
        # the grader runs, so we check done after recording.
        # ------------------------------------------------------------------
        terminal_commands = {"send_response", "escalate"}
        is_terminal_action = action.command in terminal_commands
        is_max_steps = self._step_count >= self._scenario.max_steps

        self._done = is_terminal_action or is_max_steps

        # ------------------------------------------------------------------
        # Stage 8: Grade the episode if it's done
        #
        # WHY LAST: The grader needs the complete action history, which is only
        # available after stage 6. We embed the score in the context so the
        # agent (and the inference script) can read it from the final observation.
        # ------------------------------------------------------------------
        if self._done:
            final_score = grade(
                task_id=self._scenario.name,
                action_history=self._action_history,
                context=self._context,
            )
            self._context["score"] = final_score
            self._context["episode_id"] = self._episode_id

        # ------------------------------------------------------------------
        # Stage 9: Build and return the observation
        # ------------------------------------------------------------------
        obs = self._build_observation(
            last_action_result=result,
            last_action_error=error,
        )
        return obs, reward, self._done

    def get_state(self) -> dict[str, Any]:
        """Return the current episode state as a plain dict.

        Used by the /state endpoint to let clients inspect the episode without
        taking an action. Useful for debugging and monitoring.

        Args:
            None

        Returns:
            A dict with episode_id and step_count.
        """
        return {
            "episode_id": self._episode_id or "",
            "step_count": self._step_count,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _execute_command(
        self,
        command: str,
        parameters: dict[str, Any],
    ) -> tuple[dict | None, str | None]:
        """Dispatch a validated command to the appropriate backend method.

        Returns a (result, error) tuple where exactly one is non-None.
        On success: (result_dict, None)
        On failure: (None, error_string)

        Args:
            command: A validated command string (guaranteed to be in VALID_COMMANDS).
            parameters: The parameters dict from the action.

        Returns:
            A 2-tuple of (result_dict | None, error_str | None).
        """
        # Each branch extracts the required parameter(s), calls the backend,
        # and returns a structured result dict. Missing parameters return an
        # error string so the reward function can apply the wrong-params penalty.

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
                return None, f"Invalid amount: {amount!r} — must be a number"
            success = self._backend.apply_refund(order_id, amount_float)
            if not success:
                return None, f"Order not found: {order_id!r}"
            return {
                "status": "refund_processed",
                "order_id": order_id,
                "amount": amount_float,
                "reason": reason,
            }, None

        if command == "send_replacement":
            order_id = parameters.get("order_id")
            product_id = parameters.get("product_id")
            if not order_id:
                return None, "Missing required parameter: order_id"
            if not product_id:
                return None, "Missing required parameter: product_id"
            stock = self._backend.check_stock(product_id)
            if stock == 0:
                # Out of stock — record the attempt but return an informative result
                # rather than an error. The agent should handle this gracefully.
                return {
                    "status": "replacement_unavailable",
                    "order_id": order_id,
                    "product_id": product_id,
                    "reason": "Product is out of stock. Consider offering a refund instead.",
                }, None
            success = self._backend.apply_replacement(order_id, product_id)
            if not success:
                return None, f"Order or product not found: order={order_id!r}, product={product_id!r}"
            return {
                "status": "replacement_sent",
                "order_id": order_id,
                "product_id": product_id,
            }, None

        if command == "escalate":
            reason = parameters.get("reason", "No reason provided")
            priority = parameters.get("priority", "normal")
            return {
                "status": "escalated",
                "reason": reason,
                "priority": priority,
                "ticket_id": self._scenario.ticket_id if self._scenario else "UNKNOWN",
            }, None

        if command == "send_response":
            # The message can come from parameters["message"] or the action's
            # top-level message field. We check parameters first (canonical location).
            message = parameters.get("message", "")
            if not message:
                return None, "Missing required parameter: message (send_response requires a message)"
            return {
                "status": "response_sent",
                "message": message,
            }, None

        # This branch should never be reached because we validated the command
        # in step() before calling _execute_command(). It's here as a safety net.
        return None, f"Unhandled command: {command!r}"

    def _update_context(
        self,
        command: str,
        parameters: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        """Store the action result in the episode context for the agent to reference.

        The context dict is the agent's "working memory" for the episode. We store
        results under descriptive keys so the agent can look up previously retrieved
        data without re-querying the backend.

        Args:
            command: The command that produced the result.
            parameters: The parameters used for the command.
            result: The result dict returned by _execute_command().
        """
        if command == "lookup_order":
            order_id = parameters.get("order_id", "unknown")
            # Store under a key that includes the order ID so multiple orders
            # can be looked up in the same episode (hard scenario).
            self._context[f"order_{order_id}"] = result.get("order")

        elif command == "lookup_customer":
            customer_id = parameters.get("customer_id", "unknown")
            self._context[f"customer_{customer_id}"] = result.get("customer")

        elif command == "check_policy":
            policy_type = parameters.get("policy_type", "unknown")
            self._context[f"policy_{policy_type}"] = result.get("policy")

        elif command == "check_inventory":
            product_id = parameters.get("product_id", "unknown")
            self._context[f"inventory_{product_id}"] = {
                "stock_count": result.get("stock_count"),
                "in_stock": result.get("in_stock"),
            }

        elif command == "issue_refund":
            # Record the refund in context so the grader can verify it without
            # scanning the full action history.
            self._context["refund_issued"] = {
                "order_id": result.get("order_id"),
                "amount": result.get("amount"),
                "reason": result.get("reason"),
            }

        elif command == "send_replacement":
            self._context["replacement_status"] = {
                "order_id": result.get("order_id"),
                "product_id": result.get("product_id"),
                "status": result.get("status"),
            }

    def _build_observation(
        self,
        last_action_result: dict | None,
        last_action_error: str | None,
    ) -> SupportTicketObservation:
        """Assemble a SupportTicketObservation from the current episode state.

        This is called at the end of every step (and after reset) to produce the
        observation the agent receives. Centralising observation construction here
        ensures all observations have the same shape regardless of which code path
        produced them.

        Args:
            last_action_result: The result dict from the last command, or None.
            last_action_error: The error string from the last command, or None.

        Returns:
            A fully populated SupportTicketObservation.
        """
        # If reset() hasn't been called yet, we can't build a meaningful observation.
        # This shouldn't happen in normal usage, but we handle it gracefully.
        if self._scenario is None:
            raise RuntimeError("Cannot build observation before reset() is called.")

        return SupportTicketObservation(
            ticket_id=self._scenario.ticket_id,
            customer_message=self._scenario.customer_message,
            step_number=self._step_count,
            last_action_result=last_action_result,
            last_action_error=last_action_error,
            available_commands=list(VALID_COMMANDS),
            context=dict(self._context),  # shallow copy so mutations don't affect history
            done=self._done,
        )
