"""
server/app.py — FastAPI application factory for the SupportTicket RL environment.

Why a factory function instead of a module-level `app` object?
    A module-level `app = FastAPI()` is a singleton — it's created once when the
    module is first imported and shared across all tests and server instances.
    That makes it impossible to create a fresh app with different configuration
    (e.g. a different task_id) for each test case.

    The `create_app()` factory pattern solves this: each call returns a new FastAPI
    instance with its own environment object. Tests can call `create_app("easy")`,
    `create_app("medium")`, etc. without interfering with each other.

OpenEnv HTTP contract (v0.2.x)
-------------------------------
  POST /reset  — start a new episode, return the initial observation
  POST /step   — execute one action, return (observation, reward, done)
  POST /grade  — score a completed episode externally (amendment 2)
  GET  /state  — inspect current episode state
  GET  /health — liveness check

task_id resolution for POST /reset (amendment 1 — fully documented):
    1. task_id field in the JSON request body  (per-request override)
    2. SUPPORT_TICKET_TASK environment variable (server-wide default)
    3. Hard-coded fallback "easy"
    This lets a single container serve all three tasks without restarting,
    which is exactly what the evaluation harness needs.
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Dual-import pattern: works both when running directly (bare imports) and
# when imported as part of the server package (relative imports).
try:
    from .environment import SupportTicketEnvironment
except ImportError:
    from environment import SupportTicketEnvironment  # type: ignore[no-reuse-def]

# Amendment 2: import grade() so it can be exposed on a standalone /grade endpoint.
# This lets the external evaluation harness call the grader directly without
# needing to replay the episode through the environment.
try:
    from .graders import grade as grade_episode
except ImportError:
    from graders import grade as grade_episode  # type: ignore[no-reuse-def]

try:
    from support_ticket_env.models import SupportTicketAction, SupportTicketObservation
except ImportError:
    from models import SupportTicketAction, SupportTicketObservation  # type: ignore[no-reuse-def]


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    """Request body for POST /reset.

    task_id resolution order (first non-None wins):
      1. This field in the JSON request body  — per-request override
      2. SUPPORT_TICKET_TASK environment variable — server-wide default
      3. Hard-coded fallback "easy"

    Args:
        task_id: Optional task override ("easy", "medium", or "hard").
    """
    task_id: str | None = None


class StepRequest(BaseModel):
    """Request body for POST /step.

    Args:
        command: The command to execute (e.g. "lookup_order").
        parameters: Command-specific parameters dict.
        message: Optional free-text message (used by send_response / escalate).
    """
    command: str
    parameters: dict[str, Any] = {}
    message: str | None = None


class StepResponse(BaseModel):
    """Response body for POST /step.

    Args:
        observation: The environment's observation after the action.
        reward: The per-step reward scalar.
        done: Whether the episode has terminated.
    """
    observation: dict[str, Any]
    reward: float
    done: bool


class GradeRequest(BaseModel):
    """Request body for POST /grade.

    Exposes grade() as a standalone HTTP endpoint so the external evaluation
    harness can score a completed episode without replaying it through the
    environment. This satisfies amendment 2: graders independently callable.

    Args:
        task_id: The task name ("easy", "medium", or "hard").
        action_history: The full list of actions taken in the episode.
        context: The accumulated context dict from the episode.
    """
    task_id: str
    action_history: list[dict[str, Any]] = []
    context: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app(task_id: str | None = None) -> FastAPI:
    """Create and configure a FastAPI application for the support-ticket environment.

    Args:
        task_id: The default task ("easy", "medium", or "hard"). If None, reads
            from SUPPORT_TICKET_TASK env var, defaulting to "easy".

    Returns:
        A configured FastAPI instance with all endpoints registered.

    Raises:
        ValueError: If the resolved task_id is not a valid task name.
    """
    effective_task = task_id or os.environ.get("SUPPORT_TICKET_TASK", "easy")

    # Validate at startup — fail fast rather than on the first request.
    env = SupportTicketEnvironment(task_id=effective_task)

    app = FastAPI(
        title="support_ticket_env",
        description=(
            "OpenEnv-compatible RL environment simulating an e-commerce customer "
            "support desk. An AI agent resolves support tickets by issuing structured "
            "commands against a fully in-memory backend."
        ),
        version="0.1.0",
    )

    # ------------------------------------------------------------------
    # POST /reset
    # ------------------------------------------------------------------

    @app.post("/reset")
    async def reset(request: ResetRequest = ResetRequest()) -> dict[str, Any]:
        """Start a new episode and return the initial observation.

        task_id in the request body overrides the server's default task,
        allowing a single server to run easy → medium → hard in sequence.

        Args:
            request: ResetRequest with optional task_id field.

        Returns:
            The initial SupportTicketObservation as a JSON dict.
        """
        try:
            obs = env.reset(task_id=request.task_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return obs.model_dump()

    # ------------------------------------------------------------------
    # POST /step
    # ------------------------------------------------------------------

    @app.post("/step")
    async def step(request: StepRequest) -> StepResponse:
        """Execute one action and return (observation, reward, done).

        Args:
            request: StepRequest containing command, parameters, and optional message.

        Returns:
            StepResponse with observation dict, reward float, and done bool.
        """
        action = SupportTicketAction(
            command=request.command,
            parameters=request.parameters,
            message=request.message,
        )
        try:
            obs, reward, done = env.step(action)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return StepResponse(observation=obs.model_dump(), reward=reward, done=done)

    # ------------------------------------------------------------------
    # POST /grade  (amendment 2: grader independently callable)
    # ------------------------------------------------------------------

    @app.post("/grade")
    async def grade(request: GradeRequest) -> dict[str, Any]:
        """Score a completed episode without replaying it through the environment.

        The external evaluation harness can POST the action_history and context
        from any completed episode and receive the deterministic grader score.
        This makes grading independently verifiable — no need to re-run the agent.

        Args:
            request: GradeRequest with task_id, action_history, and context.

        Returns:
            A dict with "score" (float in [0.0, 1.0]) and "task_id".
        """
        try:
            score = grade_episode(
                task_id=request.task_id,
                action_history=request.action_history,
                context=request.context,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"task_id": request.task_id, "score": score}

    # ------------------------------------------------------------------
    # GET /state
    # ------------------------------------------------------------------

    @app.get("/state")
    async def state() -> dict[str, Any]:
        """Return the current episode state (episode_id and step_count)."""
        return env.get_state()

    # ------------------------------------------------------------------
    # GET /tasks  (required by validator to enumerate tasks)
    # ------------------------------------------------------------------

    @app.get("/tasks")
    async def tasks() -> dict[str, Any]:
        """Return the list of available tasks with metadata.

        The evaluation harness calls this endpoint to discover tasks,
        then runs each grader and verifies scores are in (0.0, 1.0).
        """
        return {
            "tasks": [
                {
                    "id": "easy",
                    "name": "Order Status Inquiry",
                    "difficulty": "easy",
                    "max_steps": 5,
                    "description": "Customer asks about the status of order ORD-1042",
                },
                {
                    "id": "medium",
                    "name": "Refund Request",
                    "difficulty": "medium",
                    "max_steps": 8,
                    "description": "Customer wants a refund for damaged item on order ORD-2087",
                },
                {
                    "id": "hard",
                    "name": "Complex Multi-Issue Resolution",
                    "difficulty": "hard",
                    "max_steps": 12,
                    "description": "Wrong item in ORD-3021 and billing overcharge on ORD-3022",
                },
            ]
        }

    # ------------------------------------------------------------------
    # GET /health  (OpenEnv spec requires status="healthy")
    # ------------------------------------------------------------------

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Liveness check — OpenEnv spec requires status='healthy'."""
        return {"status": "healthy", "task": effective_task}

    # ------------------------------------------------------------------
    # GET /metadata  (required by openenv validate)
    # ------------------------------------------------------------------

    @app.get("/metadata")
    async def metadata() -> dict[str, Any]:
        """Return environment metadata — name and description required by OpenEnv spec."""
        return {
            "name": "support_ticket_env",
            "description": (
                "OpenEnv-compatible RL environment simulating an e-commerce customer "
                "support desk. An AI agent resolves support tickets by issuing structured "
                "commands against a fully in-memory backend."
            ),
            "version": "0.1.0",
            "tasks": ["easy", "medium", "hard"],
        }

    # ------------------------------------------------------------------
    # GET /schema  (required by openenv validate)
    # ------------------------------------------------------------------

    @app.get("/schema")
    async def schema() -> dict[str, Any]:
        """Return action, observation, and state JSON schemas — required by OpenEnv spec."""
        return {
            "action": SupportTicketAction.model_json_schema(),
            "observation": SupportTicketObservation.model_json_schema(),
            "state": {
                "type": "object",
                "properties": {
                    "episode_id": {"type": "string"},
                    "step_count": {"type": "integer"},
                },
            },
        }

    # ------------------------------------------------------------------
    # POST /mcp  (required by openenv validate — JSON-RPC 2.0 endpoint)
    # ------------------------------------------------------------------

    @app.post("/mcp")
    async def mcp(request: dict[str, Any] = {}) -> dict[str, Any]:
        """MCP JSON-RPC 2.0 endpoint — required by OpenEnv spec for tool discovery."""
        return {
            "jsonrpc": "2.0",
            "id": request.get("id", 1),
            "result": {
                "tools": [
                    {"name": cmd, "description": f"Execute {cmd} command"}
                    for cmd in [
                        "lookup_order", "lookup_customer", "check_policy",
                        "check_inventory", "issue_refund", "send_replacement",
                        "escalate", "send_response",
                    ]
                ]
            },
        }

    return app


# ---------------------------------------------------------------------------
# Module-level app instance
# ---------------------------------------------------------------------------
# Entry point for uvicorn: `uvicorn server.app:app`
# Tests should call create_app() directly to get a fresh isolated instance.
app = create_app()


def main() -> None:
    """Entry point for `openenv serve` and `uv run server` commands.

    Starts the uvicorn server on host 0.0.0.0 port 7860 (HF Spaces default).
    All configuration is read from environment variables at startup.
    """
    import uvicorn
    import os

    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "0.0.0.0")
    workers = int(os.environ.get("WORKERS", 1))

    uvicorn.run(
        "server.app:app",
        host=host,
        port=port,
        workers=workers,
    )


if __name__ == "__main__":
    main()
