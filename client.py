"""Root-level client.py — re-exports from support_ticket_env.client for OpenEnv CLI compatibility."""
from support_ticket_env.client import SupportTicketEnv, SupportTicketEnvError  # noqa: F401

__all__ = ["SupportTicketEnv", "SupportTicketEnvError"]
