"""
tests — test suite for the support-ticket-env project.

This package marker exists so pytest can discover tests across sub-modules and so
that test files can import from both `support_ticket_env` (the client package) and
`server` (the server package) without PYTHONPATH gymnastics.

Why a package rather than a flat directory? pytest supports both, but using a package
makes relative imports between test helpers consistent and avoids name collisions if
two test files happen to define the same helper function name.

All test files live directly in this directory (no sub-packages) to keep discovery
simple. Property-based tests use hypothesis; unit tests use pytest.
"""
