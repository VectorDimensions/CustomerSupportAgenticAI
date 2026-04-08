"""
graders.py — Root-level grader entry point for the Scaler/OpenEnv validator.

The validator imports this file and calls grade(task, action) for each task
defined in openenv.yaml to verify graders exist and return scores in (0, 1).

This file delegates to server/graders.py which contains the full grading logic.
"""

from __future__ import annotations

from typing import Any

# Import the actual graders from server package
try:
    from server.graders import (
        grade_easy,
        grade_medium,
        grade_hard,
        grade as _grade_internal,
    )
except ImportError:
    from server.graders import grade_easy, grade_medium, grade_hard, grade as _grade_internal  # type: ignore


def grade(task: dict | str, action: dict | None = None, action_history: list | None = None, context: dict | None = None) -> dict | float:
    """Grade an action or action history against a task.

    Supports two calling conventions:
    1. grade(task_dict, action_dict) — OpenEnv validator style
    2. grade(task_id_str, action_history, context) — internal style

    Args:
        task: Either a task dict (with 'id' key) or a task_id string.
        action: Optional action dict for single-action grading.
        action_history: Optional list of actions for episode grading.
        context: Optional context dict.

    Returns:
        Either a float score in (0.01, 0.99) or a dict with score field.
    """
    # Determine task_id
    if isinstance(task, dict):
        task_id = str(task.get("id", task.get("name", "easy")))
        # Map numeric IDs to names
        id_map = {"0": "easy", "1": "medium", "2": "hard",
                  "easy": "easy", "medium": "medium", "hard": "hard"}
        task_id = id_map.get(task_id, "easy")
    else:
        task_id = str(task)
        id_map = {"0": "easy", "1": "medium", "2": "hard",
                  "easy": "easy", "medium": "medium", "hard": "hard"}
        task_id = id_map.get(task_id, "easy")

    history = action_history or []
    ctx = context or {}

    # If called with a single action dict, wrap it
    if action is not None and not history:
        history = [action] if action else []

    score = _grade_internal(task_id, history, ctx)
    # Return as dict (validator style) or float
    if isinstance(task, dict):
        return {"score": score, "task_id": task_id}
    return score


# Convenience exports
__all__ = ["grade", "grade_easy", "grade_medium", "grade_hard"]
