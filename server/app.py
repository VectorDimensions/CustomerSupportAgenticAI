"""
server/app.py — FastAPI application factory for the SupportTicket RL environment.
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:
    from .environment import SupportTicketEnvironment
    from .graders import grade as grade_episode
except ImportError:
    from environment import SupportTicketEnvironment  # type: ignore[no-reuse-def]
    from graders import grade as grade_episode  # type: ignore[no-reuse-def]

try:
    from support_ticket_env.models import SupportTicketAction, SupportTicketObservation
except ImportError:
    from models import SupportTicketAction, SupportTicketObservation  # type: ignore[no-reuse-def]


class ResetRequest(BaseModel):
    task_id: str | None = None


class StepRequest(BaseModel):
    command: str
    parameters: dict[str, Any] = {}
    message: str | None = None


class StepResponse(BaseModel):
    observation: dict[str, Any]
    reward: float
    done: bool


class GradeRequest(BaseModel):
    task_id: str
    action_history: list[dict[str, Any]] = []
    context: dict[str, Any] = {}


def create_app(task_id: str | None = None) -> FastAPI:
    effective_task = task_id or os.environ.get("SUPPORT_TICKET_TASK", "easy")
    env = SupportTicketEnvironment(task_id=effective_task)

    app = FastAPI(
        title="support_ticket_env",
        description="OpenEnv-compatible RL environment for e-commerce customer support.",
        version="0.1.0",
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "healthy", "task": effective_task}

    @app.get("/tasks")
    async def tasks() -> dict[str, Any]:
        """Enumerate all tasks with graders — required by Scaler validator."""
        return {
            "tasks": [
                {
                    "id": "easy",
                    "name": "Order Status Inquiry",
                    "difficulty": "easy",
                    "max_steps": 5,
                    "description": "Customer asks about the status of order ORD-1042",
                    "has_grader": True,
                    "grader": {"type": "function", "module": "env.graders.grader_1", "function": "grader_1"},
                },
                {
                    "id": "medium",
                    "name": "Refund Request",
                    "difficulty": "medium",
                    "max_steps": 8,
                    "description": "Customer wants a refund for damaged item on order ORD-2087",
                    "has_grader": True,
                    "grader": {"type": "function", "module": "env.graders.grader_2", "function": "grader_2"},
                },
                {
                    "id": "hard",
                    "name": "Complex Multi-Issue Resolution",
                    "difficulty": "hard",
                    "max_steps": 12,
                    "description": "Wrong item in ORD-3021 and billing overcharge on ORD-3022",
                    "has_grader": True,
                    "grader": {"type": "function", "module": "env.graders.grader_3", "function": "grader_3"},
                },
            ]
        }

    @app.get("/metadata")
    async def metadata() -> dict[str, Any]:
        return {
            "name": "support_ticket_env",
            "description": "OpenEnv-compatible RL environment simulating an e-commerce customer support desk.",
            "version": "0.1.0",
            "tasks": ["easy", "medium", "hard"],
        }

    @app.get("/schema")
    async def schema() -> dict[str, Any]:
        return {
            "action": SupportTicketAction.model_json_schema(),
            "observation": SupportTicketObservation.model_json_schema(),
            "state": {"type": "object", "properties": {"episode_id": {"type": "string"}, "step_count": {"type": "integer"}}},
        }

    @app.post("/mcp")
    async def mcp(request: dict[str, Any] = {}) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": request.get("id", 1),
            "result": {"tools": [{"name": cmd, "description": f"Execute {cmd} command"} for cmd in ["lookup_order", "lookup_customer", "check_policy", "check_inventory", "issue_refund", "send_replacement", "escalate", "send_response"]]},
        }

    @app.post("/reset")
    async def reset(request: ResetRequest = ResetRequest()) -> dict[str, Any]:
        try:
            obs = env.reset(task_id=request.task_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return obs.model_dump()

    @app.post("/step")
    async def step(request: StepRequest) -> StepResponse:
        action = SupportTicketAction(command=request.command, parameters=request.parameters, message=request.message)
        try:
            obs, reward, done = env.step(action)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return StepResponse(observation=obs.model_dump(), reward=reward, done=done)

    @app.get("/state")
    async def state() -> dict[str, Any]:
        return env.get_state()

    @app.post("/grade")
    async def grade(request: GradeRequest) -> dict[str, Any]:
        try:
            score = grade_episode(task_id=request.task_id, action_history=request.action_history, context=request.context)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"task_id": request.task_id, "score": score}

    return app


app = create_app()


def main() -> None:
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("server.app:app", host=host, port=port, workers=1)


if __name__ == "__main__":
    main()
