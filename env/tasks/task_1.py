"""Task 1: Order Status Inquiry (Easy)"""

def task_1():
    return {
        "id": "easy",
        "name": "Order Status Inquiry",
        "difficulty": "easy",
        "max_steps": 5,
        "prompt": (
            "Hi, I placed order ORD-1042 three days ago and haven't received any updates. "
            "Can you check the status?"
        ),
        "description": "Customer asks about the status of order ORD-1042. "
                       "Agent must look up the order and respond with status and delivery info.",
    }
