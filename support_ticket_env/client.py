"""
support_ticket_env/client.py — HTTP client for the SupportTicket RL environment.

Why is the client intentionally thin?
    All the interesting logic — command validation, reward computation, grading —
    lives on the server. The client's only job is to serialise actions into HTTP
    requests and deserialise HTTP responses into typed Python objects.

    Keeping the client thin has two benefits:
      1. The agent code stays simple: it just calls reset() and step() and gets
         back typed Pydantic objects. No JSON wrangling, no HTTP boilerplate.
      2. The server is the single source of truth. If the reward function changes,
         the client doesn't need to be updated — it just passes actions through.

    This is the standard "dumb pipe" pattern for client/server architectures where
    the server owns the business logic.

Usage example:
    env = SupportTicketEnv(base_url="http://localhost:7860")
    obs = env.reset()
    while not obs.done:
        action = SupportTicketAction(command="lookup_order", parameters={"order_id": "ORD-1042"})
        obs, reward, done = env.step(action)
        print(f"reward={reward}, done={done}")
    env.close()
"""

from __future__ import annotations

import httpx

from support_ticket_env.models import (
    SupportTicketAction,
    SupportTicketObservation,
)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class SupportTicketEnvError(Exception):
    """Raised when the environment server returns an unexpected HTTP error.

    Args:
        status_code: The HTTP status code returned by the server.
        body: The response body text for debugging.

    Returns:
        A SupportTicketEnvError with a descriptive message.
    """

    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"Environment server error {status_code}: {body}")
        self.status_code = status_code
        self.body = body


# ---------------------------------------------------------------------------
# SupportTicketEnv client
# ---------------------------------------------------------------------------

class SupportTicketEnv:
    """HTTP client for the SupportTicket RL environment server.

    Wraps the server's /reset and /step endpoints with a clean Python API.
    Uses httpx for synchronous HTTP — no async required for the agent loop.

    Args:
        base_url: The base URL of the environment server, e.g.
            "http://localhost:7860" or "https://<hf-space-url>".
        timeout: Request timeout in seconds. Defaults to 120 to accommodate
            slow LLM inference on the server side.

    Returns:
        A SupportTicketEnv instance ready to call reset() and step().
    """

    def __init__(self, base_url: str = "http://localhost:7860", timeout: float = 120.0) -> None:
        """Initialise the client with a server base URL.

        We create the httpx client here (not in reset/step) so the TCP connection
        can be reused across multiple requests — much faster than opening a new
        connection for every step in a 12-step episode.

        Args:
            base_url: The environment server's base URL (no trailing slash).
            timeout: Per-request timeout in seconds.
        """
        # Strip trailing slash so we can always write f"{self._base_url}/reset"
        # without worrying about double slashes.
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    def reset(self, task_id: str | None = None) -> SupportTicketObservation:
        """Start a new episode and return the initial observation.

        Args:
            task_id: Optional task override ("easy", "medium", or "hard").
                If None, the server uses its configured default task.

        Returns:
            The initial SupportTicketObservation with the customer message
            and an empty context.

        Raises:
            SupportTicketEnvError: If the server returns a non-2xx status code.
        """
        payload: dict = {}
        if task_id is not None:
            payload["task_id"] = task_id

        response = self._client.post(f"{self._base_url}/reset", json=payload)
        self._raise_for_status(response)

        # model_validate() is the Pydantic v2 way to construct a model from a dict.
        # It validates all fields and raises ValidationError if the server returns
        # an unexpected shape — which would indicate a server/client version mismatch.
        return SupportTicketObservation.model_validate(response.json())

    def step(
        self, action: SupportTicketAction
    ) -> tuple[SupportTicketObservation, float, bool]:
        """Execute one action and return (observation, reward, done).

        Args:
            action: The SupportTicketAction to submit to the environment.

        Returns:
            A 3-tuple of:
              - SupportTicketObservation: the environment's response
              - float: the per-step reward
              - bool: whether the episode has terminated

        Raises:
            SupportTicketEnvError: If the server returns a non-2xx status code.
        """
        # model_dump() serialises the Pydantic model to a plain dict.
        # exclude_none=True omits the optional "message" field when it's None,
        # keeping the request payload clean.
        payload = action.model_dump(exclude_none=True)

        response = self._client.post(f"{self._base_url}/step", json=payload)
        self._raise_for_status(response)

        data = response.json()
        obs = SupportTicketObservation.model_validate(data["observation"])
        reward = float(data["reward"])
        done = bool(data["done"])
        return obs, reward, done

    def get_state(self) -> dict:
        """Return the current episode state from the server.

        Args:
            None

        Returns:
            A dict with episode_id (str) and step_count (int).

        Raises:
            SupportTicketEnvError: If the server returns a non-2xx status code.
        """
        response = self._client.get(f"{self._base_url}/state")
        self._raise_for_status(response)
        return response.json()

    def close(self) -> None:
        """Close the underlying HTTP client and release the connection pool.

        Always call this when you're done with the environment — especially in
        a finally block — to avoid leaving open TCP connections behind.

        Args:
            None

        Returns:
            None
        """
        self._client.close()

    def __enter__(self) -> "SupportTicketEnv":
        """Support use as a context manager: `with SupportTicketEnv(...) as env:`."""
        return self

    def __exit__(self, *args: object) -> None:
        """Automatically close the client when exiting the context manager."""
        self.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _raise_for_status(self, response: httpx.Response) -> None:
        """Raise SupportTicketEnvError for non-2xx responses.

        We wrap httpx's raise_for_status() in our own exception so callers
        don't need to import httpx just to catch HTTP errors.

        Args:
            response: The httpx.Response to check.

        Raises:
            SupportTicketEnvError: If the response status code is 4xx or 5xx.
        """
        if response.is_error:
            raise SupportTicketEnvError(
                status_code=response.status_code,
                body=response.text,
            )
