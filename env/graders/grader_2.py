"""Grader 2: Refund Request (Medium)"""
from __future__ import annotations
from typing import Any


def grader_2(action_history_or_result: Any = None, context: dict | None = None) -> float:
    """Grade the medium task. Accepts both (action_history, context) and (result_dict,) signatures."""
    try:
        from server.graders import grade_medium
        if isinstance(action_history_or_result, list):
            raw = grade_medium(action_history_or_result, context or {})
        elif isinstance(action_history_or_result, dict):
            history = action_history_or_result.get("action_history", [])
            ctx = action_history_or_result.get("context", context or {})
            raw = grade_medium(history, ctx)
        else:
            raw = 0.05
        return max(0.01, min(0.99, raw))
    except Exception:
        return 0.05
