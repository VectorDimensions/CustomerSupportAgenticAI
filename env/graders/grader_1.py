"""Grader 1: Order Status Inquiry (Easy)"""
from typing import Any

def grader_1(result: Any) -> float:
    """Grade the easy task — order status inquiry.

    Checks whether the agent looked up the order and sent a response
    mentioning the correct status and delivery information.

    Args:
        result: The action history or result dict from the episode.

    Returns:
        A float score strictly in (0.0, 1.0).
    """
    try:
        from server.graders import grade_easy
        if isinstance(result, list):
            raw = grade_easy(result, {})
        elif isinstance(result, dict):
            history = result.get("action_history", [])
            context = result.get("context", {})
            raw = grade_easy(history, context)
        else:
            raw = 0.05
        # Clamp strictly between 0 and 1
        return max(0.01, min(0.99, raw))
    except Exception:
        return 0.05
