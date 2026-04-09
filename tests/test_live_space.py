"""
tests/test_live_space.py â€” Run key tests against the live HF Space.

Tests the deployed environment at:
https://shivani-zadke-support-ticket-env.hf.space

Run with:
    B:\\Python311\\python.exe tests/test_live_space.py
"""

from __future__ import annotations

import sys
import unittest
import httpx

BASE_URL = "https://shivani-zadke-support-ticket-env.hf.space"
client = httpx.Client(timeout=30.0)


def post(path: str, body: dict) -> dict:
    r = client.post(f"{BASE_URL}{path}", json=body)
    r.raise_for_status()
    return r.json()


def get(path: str) -> dict:
    r = client.get(f"{BASE_URL}{path}")
    r.raise_for_status()
    return r.json()


class TestLiveHealth(unittest.TestCase):

    def test_health(self):
        data = get("/health")
        self.assertEqual(data["status"], "healthy")
        print(f"  health: {data}")

    def test_metadata(self):
        data = get("/metadata")
        self.assertIn("name", data)
        self.assertIn("description", data)
        print(f"  metadata name: {data['name']}")

    def test_schema(self):
        data = get("/schema")
        self.assertIn("action", data)
        self.assertIn("observation", data)
        self.assertIn("state", data)
        print("  schema: action/observation/state present")


class TestLiveAPIContract(unittest.TestCase):

    def test_reset_no_params(self):
        data = post("/reset", {})
        self.assertIn("ticket_id", data)
        self.assertFalse(data["done"])
        self.assertEqual(data["step_number"], 0)
        print(f"  reset: ticket={data['ticket_id']} step={data['step_number']}")

    def test_reset_each_task(self):
        for task, expected_order in [("easy", "ORD-1042"), ("medium", "ORD-2087"), ("hard", "ORD-3021")]:
            data = post("/reset", {"task_id": task})
            self.assertIn(expected_order, data["customer_message"])
            print(f"  reset {task}: contains {expected_order} âœ“")

    def test_double_reset_clean_state(self):
        post("/reset", {"task_id": "easy"})
        # Take a step
        post("/step", {"command": "lookup_order", "parameters": {"order_id": "ORD-1042"}})
        # Reset again
        data = post("/reset", {"task_id": "easy"})
        self.assertEqual(data["step_number"], 0)
        self.assertEqual(data["context"], {})
        print("  double reset: clean state âœ“")

    def test_state_after_reset(self):
        post("/reset", {"task_id": "easy"})
        data = get("/state")
        self.assertIsInstance(data["episode_id"], str)
        self.assertEqual(data["step_count"], 0)
        print(f"  state: episode_id={data['episode_id'][:8]}... step_count={data['step_count']}")

    def test_state_after_steps(self):
        post("/reset", {"task_id": "easy"})
        post("/step", {"command": "lookup_order", "parameters": {"order_id": "ORD-1042"}})
        post("/step", {"command": "lookup_customer", "parameters": {"customer_id": "CUST-001"}})
        post("/step", {"command": "check_policy", "parameters": {"policy_type": "refund_policy"}})
        data = get("/state")
        self.assertEqual(data["step_count"], 3)
        print(f"  state after 3 steps: step_count={data['step_count']} âœ“")


class TestLiveHappyPath(unittest.TestCase):

    def test_easy_perfect_run(self):
        post("/reset", {"task_id": "easy"})
        s1 = post("/step", {"command": "lookup_order", "parameters": {"order_id": "ORD-1042"}})
        self.assertAlmostEqual(s1["reward"], 0.10)
        s2 = post("/step", {"command": "send_response", "parameters": {
            "message": "Your order ORD-1042 is shipped, tracking TRK-9821034, delivery Jan 17."
        }})
        self.assertTrue(s2["done"])
        score = s2["observation"]["context"]["score"]
        self.assertGreater(score, 0.7)
        print(f"  easy perfect run: score={score} âœ“")

    def test_medium_perfect_run(self):
        post("/reset", {"task_id": "medium"})
        post("/step", {"command": "lookup_order", "parameters": {"order_id": "ORD-2087"}})
        post("/step", {"command": "lookup_customer", "parameters": {"customer_id": "CUST-002"}})
        post("/step", {"command": "check_policy", "parameters": {"policy_type": "refund_policy"}})
        post("/step", {"command": "issue_refund", "parameters": {"order_id": "ORD-2087", "amount": 149.99, "reason": "damaged"}})
        s = post("/step", {"command": "send_response", "parameters": {
            "message": "A full refund of $149.99 has been processed for your damaged item."
        }})
        score = s["observation"]["context"]["score"]
        self.assertGreater(score, 0.7)
        print(f"  medium perfect run: score={score} âœ“")

    def test_hard_perfect_run(self):
        post("/reset", {"task_id": "hard"})
        post("/step", {"command": "lookup_order", "parameters": {"order_id": "ORD-3021"}})
        post("/step", {"command": "lookup_order", "parameters": {"order_id": "ORD-3022"}})
        post("/step", {"command": "lookup_customer", "parameters": {"customer_id": "CUST-003"}})
        post("/step", {"command": "check_policy", "parameters": {"policy_type": "replacement_policy"}})
        post("/step", {"command": "check_inventory", "parameters": {"product_id": "PROD-003"}})
        post("/step", {"command": "send_replacement", "parameters": {"order_id": "ORD-3021", "product_id": "PROD-003"}})
        post("/step", {"command": "issue_refund", "parameters": {"order_id": "ORD-3022", "amount": 15.00, "reason": "billing_error"}})
        s = post("/step", {"command": "send_response", "parameters": {
            "message": "Replacement attempted for wrong mouse in ORD-3021. $15 refund credited for billing error on ORD-3022."
        }})
        score = s["observation"]["context"]["score"]
        self.assertGreater(score, 0.7)
        print(f"  hard perfect run: score={score} âœ“")


class TestLiveBadBehavior(unittest.TestCase):

    def test_invalid_command(self):
        post("/reset", {"task_id": "easy"})
        s = post("/step", {"command": "do_magic", "parameters": {}})
        self.assertAlmostEqual(s["reward"], -0.10)
        self.assertFalse(s["done"])
        self.assertIsNotNone(s["observation"]["last_action_error"])
        print(f"  invalid command: reward={s['reward']} error set âœ“")

    def test_redundant_action(self):
        post("/reset", {"task_id": "easy"})
        post("/step", {"command": "lookup_order", "parameters": {"order_id": "ORD-1042"}})
        s = post("/step", {"command": "lookup_order", "parameters": {"order_id": "ORD-1042"}})
        self.assertAlmostEqual(s["reward"], -0.02)
        print(f"  redundant action: reward={s['reward']} âœ“")

    def test_wrong_refund_amount(self):
        post("/reset", {"task_id": "medium"})
        post("/step", {"command": "lookup_order", "parameters": {"order_id": "ORD-2087"}})
        s = post("/step", {"command": "issue_refund", "parameters": {"order_id": "ORD-2087", "amount": 500.00, "reason": "damaged"}})
        self.assertAlmostEqual(s["reward"], -0.15)
        print(f"  wrong refund amount: reward={s['reward']} âœ“")

    def test_sql_injection_safe(self):
        post("/reset", {"task_id": "easy"})
        s = post("/step", {"command": "lookup_order", "parameters": {"order_id": "'; DROP TABLE orders;--"}})
        self.assertIsNotNone(s["observation"]["last_action_error"])
        self.assertFalse(s["done"])
        print("  SQL injection: handled safely âœ“")


class TestLiveGraderDeterminism(unittest.TestCase):

    def _easy_score(self) -> float:
        post("/reset", {"task_id": "easy"})
        post("/step", {"command": "lookup_order", "parameters": {"order_id": "ORD-1042"}})
        s = post("/step", {"command": "send_response", "parameters": {
            "message": "ORD-1042 shipped tracking TRK-9821034 delivery Jan 17"
        }})
        return s["observation"]["context"]["score"]

    def test_same_actions_same_score(self):
        score_a = self._easy_score()
        score_b = self._easy_score()
        self.assertAlmostEqual(score_a, score_b, places=6)
        print(f"  determinism: {score_a} == {score_b} âœ“")

    def test_perfect_vs_minimal(self):
        perfect = self._easy_score()
        post("/reset", {"task_id": "easy"})
        s = post("/step", {"command": "send_response", "parameters": {"message": "I don't know."}})
        minimal = s["observation"]["context"]["score"]
        self.assertGreater(perfect - minimal, 0.3)
        print(f"  perfect={perfect} vs minimal={minimal} gap={perfect-minimal:.2f} âœ“")


class TestLiveGradeEndpoint(unittest.TestCase):

    def test_grade_standalone(self):
        data = post("/grade", {
            "task_id": "easy",
            "action_history": [
                {"command": "lookup_order", "parameters": {"order_id": "ORD-1042"}},
                {"command": "send_response", "parameters": {"message": "shipped tracking delivery jan 17"}}
            ],
            "context": {}
        })
        # Score must be strictly in (0, 1) â€” perfect run returns 0.99 (clamped)
        self.assertGreater(data["score"], 0.0)
        self.assertLess(data["score"], 1.0)
        self.assertGreater(data["score"], 0.5)
        print(f"  grade standalone: score={data['score']} strictly-in-range=True")

    def test_grade_all_three_tasks_differ(self):
        naive = [
            {"command": "lookup_order", "parameters": {"order_id": "ORD-1042"}},
            {"command": "send_response", "parameters": {"message": "done"}}
        ]
        scores = {}
        for task in ("easy", "medium", "hard"):
            data = post("/grade", {"task_id": task, "action_history": naive, "context": {}})
            scores[task] = data["score"]
        self.assertNotEqual(scores["easy"], scores["medium"])
        self.assertNotEqual(scores["medium"], scores["hard"])
        print(f"  scores differ: easy={scores['easy']} medium={scores['medium']} hard={scores['hard']} âœ“")


if __name__ == "__main__":
    print(f"\nTesting live HF Space: {BASE_URL}\n")
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [
        TestLiveHealth,
        TestLiveAPIContract,
        TestLiveHappyPath,
        TestLiveBadBehavior,
        TestLiveGraderDeterminism,
        TestLiveGradeEndpoint,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)

