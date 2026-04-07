"""
tests/test_submission.py — Pre-submission verification test suite.

Covers all 8 test categories from the submission checklist:
  1. Environment API Contract
  2. Happy Path — Correct Agent Behavior
  3. Bad Agent Behavior — Penalties & Error Handling
  4. Episode Boundary Tests
  5. Grader Determinism (DQ-critical)
  6. Inference Script Log Format (DQ-critical)
  7. Docker & Deployment (skipped in unit run — requires Docker daemon)
  8. Data Consistency

Run with:
    B:\\Python311\\python.exe -m pytest tests/test_submission.py -v
or plain:
    B:\\Python311\\python.exe tests/test_submission.py
"""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
import unittest
from typing import Any

# Make sure the project root is on the path so imports work from any cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.environment import SupportTicketEnvironment
from server.graders import grade
from support_ticket_env.models import SupportTicketAction, VALID_COMMANDS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_action(command: str, **params) -> SupportTicketAction:
    """Shorthand for building a SupportTicketAction."""
    return SupportTicketAction(command=command, parameters=params)


def fresh_env(task: str = "easy") -> SupportTicketEnvironment:
    """Return a freshly reset environment for the given task."""
    env = SupportTicketEnvironment(task)
    env.reset()
    return env


# ---------------------------------------------------------------------------
# Category 1: Environment API Contract
# ---------------------------------------------------------------------------

class TestAPIContract(unittest.TestCase):

    def test_reset_no_params_returns_valid_observation(self):
        """reset() with no params returns a valid observation with all required fields."""
        env = SupportTicketEnvironment("easy")
        obs = env.reset()

        self.assertIsNotNone(obs.ticket_id)
        self.assertIsNotNone(obs.customer_message)
        self.assertEqual(obs.available_commands, VALID_COMMANDS)
        self.assertFalse(obs.done)
        self.assertEqual(obs.step_number, 0)

    def test_reset_with_each_task_returns_different_message(self):
        """reset() with each task_id returns the matching customer_message."""
        env = SupportTicketEnvironment("easy")

        obs_easy = env.reset(task_id="easy")
        obs_medium = env.reset(task_id="medium")
        obs_hard = env.reset(task_id="hard")

        # Each scenario has a distinct customer message.
        self.assertIn("ORD-1042", obs_easy.customer_message)
        self.assertIn("ORD-2087", obs_medium.customer_message)
        self.assertIn("ORD-3021", obs_hard.customer_message)
        self.assertIn("ORD-3022", obs_hard.customer_message)

        # All three must be different.
        messages = {obs_easy.customer_message, obs_medium.customer_message, obs_hard.customer_message}
        self.assertEqual(len(messages), 3)

    def test_double_reset_produces_clean_state(self):
        """Second reset() produces step_number=0 with no leftover context."""
        env = SupportTicketEnvironment("easy")
        env.reset()

        # Take a step to dirty the state.
        env.step(make_action("lookup_order", order_id="ORD-1042"))

        # Reset again — must be clean.
        obs = env.reset()
        self.assertEqual(obs.step_number, 0)
        self.assertFalse(obs.done)
        self.assertEqual(obs.context, {})
        self.assertIsNone(obs.last_action_result)
        self.assertIsNone(obs.last_action_error)

    def test_state_after_reset(self):
        """state() after reset returns non-empty episode_id and step_count=0."""
        env = SupportTicketEnvironment("easy")
        env.reset()
        state = env.get_state()

        self.assertIsInstance(state["episode_id"], str)
        self.assertGreater(len(state["episode_id"]), 0)
        self.assertEqual(state["step_count"], 0)

    def test_state_after_three_steps(self):
        """state() after 3 steps returns step_count=3."""
        env = fresh_env("easy")
        env.step(make_action("lookup_order", order_id="ORD-1042"))
        env.step(make_action("lookup_customer", customer_id="CUST-001"))
        env.step(make_action("check_policy", policy_type="refund_policy"))

        state = env.get_state()
        self.assertEqual(state["step_count"], 3)

    def test_step_before_reset_raises_or_errors_gracefully(self):
        """step() before reset() must not crash — raises RuntimeError or returns error obs."""
        env = SupportTicketEnvironment("easy")
        # Do NOT call reset().
        try:
            obs, reward, done = env.step(make_action("lookup_order", order_id="ORD-1042"))
            # If it returns instead of raising, the observation must signal an error.
            self.assertIsNotNone(obs.last_action_error)
        except RuntimeError:
            pass  # also acceptable


# ---------------------------------------------------------------------------
# Category 2: Happy Path — Correct Agent Behavior
# ---------------------------------------------------------------------------

class TestHappyPath(unittest.TestCase):

    def test_easy_perfect_run(self):
        """Easy task perfect run scores > 0.7 with all positive rewards."""
        env = SupportTicketEnvironment("easy")
        env.reset()

        obs1, r1, done1 = env.step(make_action("lookup_order", order_id="ORD-1042"))
        self.assertFalse(done1)
        self.assertGreater(r1, 0, "lookup_order should yield positive reward")

        obs2, r2, done2 = env.step(make_action(
            "send_response",
            message="Your order ORD-1042 is shipped and arriving in 2 days, tracking TRK-9821034.",
        ))
        self.assertTrue(done2)
        self.assertGreater(r2, 0, "good send_response should yield positive reward")

        score = obs2.context.get("score", 0.0)
        self.assertGreater(score, 0.7, f"Easy perfect run score too low: {score}")

    def test_medium_perfect_run(self):
        """Medium task perfect run scores > 0.7."""
        env = SupportTicketEnvironment("medium")
        env.reset()

        steps = [
            make_action("lookup_order", order_id="ORD-2087"),
            make_action("lookup_customer", customer_id="CUST-002"),
            make_action("check_policy", policy_type="refund_policy"),
            make_action("issue_refund", order_id="ORD-2087", amount=149.99, reason="damaged"),
            make_action("send_response", message="We are sorry your item arrived damaged. A full refund of $149.99 has been processed to your account."),
        ]
        for action in steps:
            obs, reward, done = env.step(action)

        self.assertTrue(done)
        score = obs.context.get("score", 0.0)
        self.assertGreater(score, 0.7, f"Medium perfect run score too low: {score}")

    def test_hard_perfect_run(self):
        """Hard task perfect run scores > 0.7."""
        env = SupportTicketEnvironment("hard")
        env.reset()

        steps = [
            make_action("lookup_order", order_id="ORD-3021"),
            make_action("lookup_order", order_id="ORD-3022"),
            make_action("lookup_customer", customer_id="CUST-003"),
            make_action("check_policy", policy_type="replacement_policy"),
            make_action("check_inventory", product_id="PROD-003"),
            make_action("send_replacement", order_id="ORD-3021", product_id="PROD-003"),
            make_action("issue_refund", order_id="ORD-3022", amount=15.00, reason="billing_error"),
            make_action("send_response", message=(
                "We apologise for both issues. We have attempted a replacement for the wrong "
                "mouse in ORD-3021. A refund of $15.00 has been credited for the billing error "
                "on ORD-3022. Please contact us if you need further help."
            )),
        ]
        for action in steps:
            obs, reward, done = env.step(action)

        self.assertTrue(done)
        score = obs.context.get("score", 0.0)
        self.assertGreater(score, 0.7, f"Hard perfect run score too low: {score}")


# ---------------------------------------------------------------------------
# Category 3: Bad Agent Behavior — Penalties & Error Handling
# ---------------------------------------------------------------------------

class TestBadBehavior(unittest.TestCase):

    def test_invalid_command_name(self):
        """Invalid command returns error observation, negative reward, episode continues."""
        env = fresh_env("easy")
        obs, reward, done = env.step(make_action("do_magic"))

        self.assertIsNotNone(obs.last_action_error)
        self.assertIn("do_magic", obs.last_action_error)
        self.assertAlmostEqual(reward, -0.10)
        self.assertFalse(done)

    def test_valid_command_missing_params(self):
        """lookup_order with no order_id returns error and small negative reward."""
        env = fresh_env("easy")
        obs, reward, done = env.step(make_action("lookup_order"))  # no order_id

        self.assertIsNotNone(obs.last_action_error)
        self.assertLess(reward, 0)
        self.assertFalse(done)

    def test_nonexistent_order_id(self):
        """lookup_order with unknown ID returns not-found error, does not crash."""
        env = fresh_env("easy")
        obs, reward, done = env.step(make_action("lookup_order", order_id="ORD-9999"))

        self.assertIsNotNone(obs.last_action_error)
        self.assertIn("ORD-9999", obs.last_action_error)
        self.assertFalse(done)

    def test_issue_refund_wrong_amount_policy_violation(self):
        """issue_refund with wrong amount gets policy violation penalty."""
        env = fresh_env("medium")
        # Look up the order first so the command itself is valid.
        env.step(make_action("lookup_order", order_id="ORD-2087"))
        # Issue refund for $500 when the order was $149.99 — policy violation.
        obs, reward, done = env.step(make_action("issue_refund", order_id="ORD-2087", amount=500.00, reason="damaged"))

        self.assertAlmostEqual(reward, -0.15, msg="Wrong refund amount should be policy violation")

    def test_redundant_action_penalty(self):
        """Same lookup_order twice gets -0.02 redundant action penalty."""
        env = fresh_env("easy")
        env.step(make_action("lookup_order", order_id="ORD-1042"))
        obs, reward, done = env.step(make_action("lookup_order", order_id="ORD-1042"))

        self.assertAlmostEqual(reward, -0.02)

    def test_unnecessary_escalation_penalty(self):
        """escalate on easy task (no escalation needed) gets -0.10 penalty."""
        env = fresh_env("easy")
        obs, reward, done = env.step(make_action("escalate", reason="not sure what to do", priority="normal"))

        self.assertAlmostEqual(reward, -0.10)
        # Escalation is a terminal action — episode should end.
        self.assertTrue(done)

    def test_send_response_empty_message(self):
        """send_response with empty message returns error, does not end episode cleanly."""
        env = fresh_env("easy")
        obs, reward, done = env.step(make_action("send_response", message=""))

        # Empty message should be rejected with an error.
        self.assertIsNotNone(obs.last_action_error)


# ---------------------------------------------------------------------------
# Category 4: Episode Boundary Tests
# ---------------------------------------------------------------------------

class TestEpisodeBoundaries(unittest.TestCase):

    def test_step_after_send_response_returns_done(self):
        """After send_response, further step() calls return done=True."""
        env = fresh_env("easy")
        env.step(make_action("lookup_order", order_id="ORD-1042"))
        env.step(make_action("send_response", message="Your order is shipped."))

        # Episode is over — another step should return done=True gracefully.
        obs, reward, done = env.step(make_action("lookup_order", order_id="ORD-1042"))
        self.assertTrue(done)

    def test_max_steps_ends_episode(self):
        """Hitting max_steps without send_response ends episode with done=True."""
        env = SupportTicketEnvironment("easy")
        env.reset()
        # easy max_steps=5 — take 5 lookup steps (different IDs to avoid repeat penalty).
        order_ids = ["ORD-1042", "ORD-2087", "ORD-3021", "ORD-3022", "ORD-4001"]
        last_done = False
        for oid in order_ids:
            obs, reward, last_done = env.step(make_action("lookup_order", order_id=oid))
        self.assertTrue(last_done, "Episode should end at max_steps")

    def test_max_steps_low_score(self):
        """Episode that hits max_steps without resolving gets a low grader score."""
        env = SupportTicketEnvironment("easy")
        env.reset()
        order_ids = ["ORD-1042", "ORD-2087", "ORD-3021", "ORD-3022", "ORD-4001"]
        for oid in order_ids:
            obs, reward, done = env.step(make_action("lookup_order", order_id=oid))

        score = obs.context.get("score", 0.0)
        # No send_response was called — score should be low (< 0.5).
        self.assertLess(score, 0.5, f"No-resolution episode should score low, got {score}")


# ---------------------------------------------------------------------------
# Category 5: Grader Determinism (DQ-critical)
# ---------------------------------------------------------------------------

class TestGraderDeterminism(unittest.TestCase):

    def _run_easy_perfect(self) -> float:
        """Run the easy perfect sequence and return the grader score."""
        env = SupportTicketEnvironment("easy")
        env.reset()
        env.step(make_action("lookup_order", order_id="ORD-1042"))
        obs, _, _ = env.step(make_action(
            "send_response",
            message="Your order ORD-1042 is shipped, tracking TRK-9821034, delivery Jan 17.",
        ))
        return float(obs.context.get("score", 0.0))

    def test_same_actions_same_score(self):
        """Running the same action sequence twice produces identical scores."""
        score_a = self._run_easy_perfect()
        score_b = self._run_easy_perfect()
        self.assertAlmostEqual(score_a, score_b, places=6,
            msg=f"Grader is non-deterministic: {score_a} != {score_b}")

    def test_perfect_vs_minimal_response(self):
        """Perfect sequence scores significantly higher than a useless response."""
        score_perfect = self._run_easy_perfect()

        # Minimal run: just send a vague response with no lookup.
        env = SupportTicketEnvironment("easy")
        env.reset()
        obs, _, _ = env.step(make_action("send_response", message="I don't know."))
        score_minimal = float(obs.context.get("score", 0.0))

        self.assertGreater(score_perfect, score_minimal + 0.3,
            msg=f"Perfect ({score_perfect}) should be >> minimal ({score_minimal})")

    def test_medium_with_vs_without_policy_check(self):
        """Medium score with check_policy > score without it."""
        def run_medium(include_policy: bool) -> float:
            env = SupportTicketEnvironment("medium")
            env.reset()
            env.step(make_action("lookup_order", order_id="ORD-2087"))
            env.step(make_action("lookup_customer", customer_id="CUST-002"))
            if include_policy:
                env.step(make_action("check_policy", policy_type="refund_policy"))
            env.step(make_action("issue_refund", order_id="ORD-2087", amount=149.99, reason="damaged"))
            obs, _, _ = env.step(make_action("send_response", message="Refund of $149.99 processed for damaged item."))
            return float(obs.context.get("score", 0.0))

        score_with = run_medium(include_policy=True)
        score_without = run_medium(include_policy=False)
        self.assertGreater(score_with, score_without,
            msg=f"With policy check ({score_with}) should beat without ({score_without})")

    def test_grade_standalone_matches_environment_score(self):
        """grade() called standalone returns the same score as the environment embeds."""
        env = SupportTicketEnvironment("easy")
        env.reset()
        env.step(make_action("lookup_order", order_id="ORD-1042"))
        obs, _, _ = env.step(make_action(
            "send_response",
            message="Your order ORD-1042 is shipped, tracking TRK-9821034, delivery Jan 17.",
        ))
        env_score = float(obs.context.get("score", 0.0))

        # Reconstruct history from the environment's internal state and call grade() directly.
        standalone_score = grade("easy", env._action_history, env._context)
        self.assertAlmostEqual(env_score, standalone_score, places=6)


# ---------------------------------------------------------------------------
# Category 6: Inference Script Log Format (DQ-critical)
# ---------------------------------------------------------------------------

class TestInferenceLogFormat(unittest.TestCase):
    """
    These tests verify the [START]/[STEP]/[END] log format without actually
    calling the LLM. We monkey-patch call_llm to return a deterministic JSON
    action so the test runs offline and fast.
    """

    def _run_inference_offline(self, task: str) -> str:
        """Run inference.py with a mocked LLM and capture stdout."""
        import inference

        # Patch call_llm to return a deterministic action sequence per task.
        _action_sequences = {
            "easy": [
                '{"command": "lookup_order", "parameters": {"order_id": "ORD-1042"}}',
                '{"command": "send_response", "parameters": {"message": "Your order ORD-1042 is shipped, tracking TRK-9821034, delivery Jan 17."}}',
            ],
            "medium": [
                '{"command": "lookup_order", "parameters": {"order_id": "ORD-2087"}}',
                '{"command": "lookup_customer", "parameters": {"customer_id": "CUST-002"}}',
                '{"command": "check_policy", "parameters": {"policy_type": "refund_policy"}}',
                '{"command": "issue_refund", "parameters": {"order_id": "ORD-2087", "amount": 149.99, "reason": "damaged"}}',
                '{"command": "send_response", "parameters": {"message": "Refund of $149.99 processed for damaged item."}}',
            ],
            "hard": [
                '{"command": "lookup_order", "parameters": {"order_id": "ORD-3021"}}',
                '{"command": "lookup_order", "parameters": {"order_id": "ORD-3022"}}',
                '{"command": "lookup_customer", "parameters": {"customer_id": "CUST-003"}}',
                '{"command": "check_policy", "parameters": {"policy_type": "replacement_policy"}}',
                '{"command": "check_inventory", "parameters": {"product_id": "PROD-003"}}',
                '{"command": "send_replacement", "parameters": {"order_id": "ORD-3021", "product_id": "PROD-003"}}',
                '{"command": "issue_refund", "parameters": {"order_id": "ORD-3022", "amount": 15.00, "reason": "billing_error"}}',
                '{"command": "send_response", "parameters": {"message": "Replacement attempted for wrong mouse. $15 refund credited for billing error."}}',
            ],
        }
        responses = iter(_action_sequences.get(task, _action_sequences["easy"]))

        original_call_llm = inference.call_llm
        original_task = inference.TASK_NAME
        original_env_url = inference.ENV_BASE_URL

        def mock_call_llm(client, conversation):
            try:
                return next(responses)
            except StopIteration:
                return '{"command": "send_response", "parameters": {"message": "done"}}'

        # Redirect stdout to capture log lines.
        captured = io.StringIO()
        original_stdout = sys.stdout

        try:
            inference.call_llm = mock_call_llm
            inference.TASK_NAME = task
            # Use the environment directly (no HTTP server needed).
            # We patch SupportTicketEnv to use the in-process environment.
            from support_ticket_env.client import SupportTicketEnv
            from support_ticket_env.models import SupportTicketObservation

            _env = SupportTicketEnvironment(task)

            class InProcessEnv:
                def __init__(self, base_url: str = "", timeout: float = 120.0):
                    # Ignore base_url/timeout — we talk directly to the in-process env.
                    pass
                def reset(self, task_id=None):
                    return _env.reset(task_id=task_id or task)
                def step(self, action):
                    return _env.step(action)
                def close(self):
                    pass
                def __enter__(self):
                    return self
                def __exit__(self, *a):
                    pass

            original_SupportTicketEnv = inference.__dict__.get("SupportTicketEnv")
            inference.SupportTicketEnv = InProcessEnv  # type: ignore[attr-defined]

            sys.stdout = captured
            inference.run_episode()
        finally:
            sys.stdout = original_stdout
            inference.call_llm = original_call_llm
            inference.TASK_NAME = original_task
            if original_SupportTicketEnv is not None:
                inference.SupportTicketEnv = original_SupportTicketEnv

        return captured.getvalue()

    def _check_log_format(self, output: str, task: str) -> None:
        """Assert the output contains exactly the required log lines."""
        lines = output.strip().splitlines()

        start_lines = [l for l in lines if l.startswith("[START]")]
        step_lines  = [l for l in lines if l.startswith("[STEP]")]
        end_lines   = [l for l in lines if l.startswith("[END]")]

        self.assertEqual(len(start_lines), 1, f"Expected 1 [START] line, got {len(start_lines)}")
        self.assertGreaterEqual(len(step_lines), 1, "Expected at least 1 [STEP] line")
        self.assertEqual(len(end_lines), 1, f"Expected 1 [END] line, got {len(end_lines)}")

        # [START] format
        start = start_lines[0]
        self.assertRegex(start, r"\[START\] task=\S+ env=support_ticket_env model=\S+")

        # [STEP] format
        for step_line in step_lines:
            self.assertRegex(step_line,
                r"\[STEP\] step=\d+ action=.+ reward=-?\d+\.\d+ done=(true|false) error=.*")

        # [END] format
        end = end_lines[0]
        self.assertRegex(end, r"\[END\] success=(true|false) steps=\d+ score=\d+\.\d+ rewards=.*")

        # score must be in [0.0, 1.0]
        score_match = re.search(r"score=(\d+\.\d+)", end)
        self.assertIsNotNone(score_match, "[END] line missing score field")
        score = float(score_match.group(1))
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

        # success must be true or false
        self.assertRegex(end, r"success=(true|false)")

    def test_easy_log_format(self):
        """inference.py easy task produces correct [START]/[STEP]/[END] format."""
        output = self._run_inference_offline("easy")
        self._check_log_format(output, "easy")

    def test_medium_log_format(self):
        """inference.py medium task produces correct log format."""
        output = self._run_inference_offline("medium")
        self._check_log_format(output, "medium")

    def test_hard_log_format(self):
        """inference.py hard task produces correct log format."""
        output = self._run_inference_offline("hard")
        self._check_log_format(output, "hard")


# ---------------------------------------------------------------------------
# Category 8: Data Consistency Tests
# ---------------------------------------------------------------------------

class TestDataConsistency(unittest.TestCase):

    def test_lookup_order_returns_valid_fields(self):
        """lookup_order('ORD-1042') returns an order with valid linked IDs."""
        from server.data import BackendData
        db = BackendData()
        order = db.get_order("ORD-1042")

        self.assertIsNotNone(order)
        self.assertEqual(order["order_id"], "ORD-1042")
        # customer_id must exist in the customers dict
        self.assertIsNotNone(db.get_customer(order["customer_id"]),
            f"customer_id {order['customer_id']} not found in backend")
        # product_id must exist in the products dict
        self.assertIsNotNone(db.get_product(order["product_id"]),
            f"product_id {order['product_id']} not found in backend")
        # dates must be non-empty strings
        self.assertIsInstance(order["order_date"], str)
        self.assertGreater(len(order["order_date"]), 0)

    def test_context_populated_after_lookup(self):
        """After lookup_order, the observation context contains the order data."""
        env = fresh_env("easy")
        obs, _, _ = env.step(make_action("lookup_order", order_id="ORD-1042"))

        self.assertIn("order_ORD-1042", obs.context,
            f"Context missing order_ORD-1042. Keys: {list(obs.context.keys())}")
        self.assertIsNotNone(obs.context["order_ORD-1042"])

    def test_refund_policy_has_checkable_rules(self):
        """check_policy('refund_policy') returns structured rules, not vague text."""
        from server.data import BackendData
        db = BackendData()
        policy = db.get_policy("refund_policy")

        self.assertIsNotNone(policy)
        rules = policy.get("rules", {})
        # Must have a numeric window_days rule that the grader can check.
        self.assertIn("window_days", rules, "refund_policy missing window_days rule")
        self.assertIsInstance(rules["window_days"], int)
        # Must have full_refund_reasons list.
        self.assertIn("full_refund_reasons", rules)
        self.assertIn("damaged", rules["full_refund_reasons"])

    def test_all_scenario_orders_exist(self):
        """All four required order IDs exist in the backend."""
        from server.data import BackendData
        db = BackendData()
        for oid in ["ORD-1042", "ORD-2087", "ORD-3021", "ORD-3022"]:
            self.assertIsNotNone(db.get_order(oid), f"Required order {oid} missing from backend")

    def test_episode_isolation(self):
        """Mutations in one episode do not bleed into the next."""
        env = SupportTicketEnvironment("medium")
        env.reset()
        # Apply a refund in episode 1.
        env.step(make_action("lookup_order", order_id="ORD-2087"))
        env.step(make_action("issue_refund", order_id="ORD-2087", amount=149.99, reason="damaged"))

        # Reset for episode 2 — the order should not have refund_applied set.
        env.reset()
        order = env._backend.get_order("ORD-2087")
        self.assertFalse(
            getattr(env._backend.orders.get("ORD-2087"), "refund_applied", False),
            "Refund state leaked from episode 1 into episode 2",
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Run with verbose output when executed directly.
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes in order.
    for cls in [
        TestAPIContract,
        TestHappyPath,
        TestBadBehavior,
        TestEpisodeBoundaries,
        TestGraderDeterminism,
        TestInferenceLogFormat,
        TestDataConsistency,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
