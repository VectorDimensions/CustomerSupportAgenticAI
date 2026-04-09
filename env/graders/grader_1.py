"""Grader 1: Order Status Inquiry (Easy)"""
from __future__ import annotations
from typing import Any


def grader_1(action_history_or_result: Any = None, context: dict | None = None) -> float:
    """Grade the easy task. Accepts both (action_history, context) and (result_dict,) signatures."""
    try:
        from server.graders import grade_easy
        if isinstance(action_history_or_result, list):
            raw = grade_easy(action_history_or_result, context or {})
        elif isinstance(action_history_or_result, dict):
            history = action_history_or_result.get("action_history", [])
            ctx = action_history_or_result.get("context", context or {})
            raw = grade_easy(history, ctx)
        else:
            raw = 0.05
        return max(0.01, min(0.99, raw))
    except Exception:
        return 0.05
