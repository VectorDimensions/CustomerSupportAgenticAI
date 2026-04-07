"""
support_ticket_env — installable client package for the SupportTicket RL environment.

This package is the *client-side* half of the support-ticket-env project. It exists
as a separate installable unit (via `pip install -e .`) so that agent code can import
typed models and the environment client without pulling in the FastAPI server stack.

Why split client and server at all? Because in a real OpenEnv deployment the server
runs inside a Docker container while the agent runs on the host (or in a separate
process). Keeping the client package lightweight means the agent's dependency footprint
stays small — no uvicorn, no server-side logic, just Pydantic models and an HTTP client.

Exports:
    SupportTicketAction      — Pydantic model for actions sent to the server
    SupportTicketObservation — Pydantic model for observations returned by the server
    SupportTicketEnv         — OpenEnv EnvClient subclass; the main entry point for agents
"""

# We re-export the public API here so callers can do:
#   from support_ticket_env import SupportTicketEnv
# instead of digging into sub-modules they shouldn't need to know about.

from support_ticket_env.models import SupportTicketAction, SupportTicketObservation  # noqa: F401
from support_ticket_env.client import SupportTicketEnv  # noqa: F401

__all__ = [
    "SupportTicketAction",
    "SupportTicketObservation",
    "SupportTicketEnv",
]
