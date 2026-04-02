from openenv.core.env_client import EnvClient

from .models import InvestigateAction, InvestigateObservation, InvestigateState

# Default timeout for server requests (seconds)
DEFAULT_TIMEOUT = 15


class FakeNewsEnv(EnvClient):
    """Client for the Fake News Investigator environment."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = DEFAULT_TIMEOUT):
        super().__init__(base_url=base_url)
        self.timeout = timeout

    def _step_payload(self, action: InvestigateAction) -> dict:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: dict) -> InvestigateObservation:
        # The server wraps observation in {"observation": {...}, "reward": ..., "done": ...}
        if "observation" in payload:
            obs_data = payload["observation"]
            obs_data["reward"] = payload.get("reward", obs_data.get("reward"))
            obs_data["done"] = payload.get("done", obs_data.get("done", False))
            return InvestigateObservation(**obs_data)
        return InvestigateObservation(**payload)

    def _parse_state(self, payload: dict) -> InvestigateState:
        # State might also be wrapped
        if "state" in payload and isinstance(payload["state"], dict):
            return InvestigateState(**payload["state"])
        return InvestigateState(**payload)
