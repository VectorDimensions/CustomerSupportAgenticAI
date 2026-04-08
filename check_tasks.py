import importlib
import os
import sys

def fail(msg):
    print(f"[FAIL] {msg}")
    sys.exit(1)

def ok(msg):
    print(f"[OK] {msg}")

def load_registry():
    try:
        registry = importlib.import_module("env.registry")
    except Exception as e:
        fail(f"Could not import env.registry: {e}")
    return registry

def main():
    registry = load_registry()

    if not hasattr(registry, "TASKS"):
        fail("env.registry does not define TASKS")

    tasks = registry.TASKS

    if not isinstance(tasks, list):
        fail("TASKS is not a list")

    ok(f"Found {len(tasks)} task entries")

    valid = 0
    for i, item in enumerate(tasks, start=1):
        if not isinstance(item, dict):
            print(f"[WARN] Task entry {i} is not a dict")
            continue
        task = item.get("task")
        grader = item.get("grader")
        if task is None:
            print(f"[WARN] Task entry {i} missing task")
            continue
        if grader is None:
            print(f"[WARN] Task entry {i} missing grader")
            continue
        if not callable(task):
            print(f"[WARN] Task entry {i} task is not callable")
            continue
        if not callable(grader):
            print(f"[WARN] Task entry {i} grader is not callable")
            continue
        valid += 1
        print(f"[OK] Entry {i} has task + grader")

    if valid < 3:
        fail(f"Not enough tasks with graders: found {valid}, need at least 3")

    ok(f"Validation passed with {valid} tasks with graders")

if __name__ == "__main__":
    main()
