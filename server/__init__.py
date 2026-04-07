"""
server — FastAPI application package for the SupportTicket RL environment.

This package hosts the environment logic: in-memory backend data, scenario definitions,
per-step reward computation, episode grading, and the HTTP API that exposes all of it
to OpenEnv-compatible clients.

Why is this a package (with __init__.py) rather than a flat directory of scripts?
Because Python's import system needs a package marker to resolve relative imports like
`from .data import BackendData`. Without this file, `server/` is just a directory and
intra-server imports would require PYTHONPATH manipulation — fragile in Docker and CI.

This __init__.py is intentionally empty of logic. The real entry point is `app.py`
via the `create_app()` factory.
"""
