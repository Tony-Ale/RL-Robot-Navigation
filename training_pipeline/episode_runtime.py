"""Shared episode-boundary operations used by training, testing, and inspection."""

from typing import Optional


PLANNER_RESET_FAILURE_TEXT = "Planner produced no waypoints"


def is_planner_reset_failure(error: Exception) -> bool:
    """Return whether an environment reset failed because A* found no path."""
    return PLANNER_RESET_FAILURE_TEXT in str(error)


def reset_env(env, seed: Optional[int]):
    """Reset either Gym or Gymnasium environments and return ``(observation, info)``."""
    result = env.reset() if seed is None else env.reset(seed=seed)
    if isinstance(result, tuple) and len(result) == 2:
        return result
    return result, {}


def reset_policy(policy) -> None:
    """Clear episode-local policy memory when the policy exposes a reset hook."""
    reset = getattr(policy, "reset", None)
    if callable(reset):
        reset()
