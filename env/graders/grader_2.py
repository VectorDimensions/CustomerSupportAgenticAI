"""Grader 2: Refund Request (Medium)"""
from typing import Any

def grader_2(result: Any) -> float:
    """Grade the medium task — refund request.

    Checks whether the agent looked up the order, checked policy,
    issued the correct refund amount, and sent a response.

    Args:
        result: The action history or result dict from the episode.

    Returns:
        A float score strictly in (0.0, 1.0).
    """
    try:
        from server.graders import grade_medium
        if isinstance(result, list):
            raw = grade_medium(result, {})
        elif isinstance(result, dict):
            history = result.get("action_history", [])
            context = result.get("context", {})
            raw = grade_medium(history, context)
        else:
            raw = 0.05
        return max(0.01, min(0.99, raw))
    except Exception:
        return 0.05
