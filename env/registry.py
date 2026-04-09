"""env/registry.py — Central registry of all tasks and their graders.

The validator looks for this TASKS list to discover tasks with graders.
Each entry maps a task definition function to its grader function.
"""

from env.tasks.task_1 import task_1
from env.tasks.task_2 import task_2
from env.tasks.task_3 import task_3

from env.graders.grader_1 import grader_1
from env.graders.grader_2 import grader_2
from env.graders.grader_3 import grader_3

# TASKS is the canonical registry — validator discovers tasks+graders from here
TASKS = [
    {"id": "easy", "task": task_1, "grader": grader_1},
    {"id": "medium", "task": task_2, "grader": grader_2},
    {"id": "hard", "task": task_3, "grader": grader_3},
]

__all__ = ["TASKS"]
