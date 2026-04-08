"""Grader 3: Complex Multi-Issue Resolution (Hard)"""
from typing import Any

def grader_3(result: Any) -> float:
    """Grade the hard task — wrong item + billing overcharge.

    Checks whether the agent resolved both issues: attempted replacement
    for wrong item and issued exact $15 partial refund for billing error.

    Args:
        result: The action history or result dict from the episode.

    Returns:
        A float score strictly in (0.0, 1.0).
    """
    try:
        from server.graders import grade_hard
        if isinstance(result, list):
            raw = grade_hard(result, {})
        elif isinstance(result, dict):
            history = result.get("action_history", [])
            context = result.get("context", {})
            raw = grade_hard(history, context)
        else:
            raw = 0.05
        return max(0.01, min(0.99, raw))
    except Exception:
        return 0.05
