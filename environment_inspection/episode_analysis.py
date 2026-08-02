"""Common seeded episode collection for aggregate inspection analyses."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from training_pipeline.episode_runtime import is_planner_reset_failure, reset_env, reset_policy


@dataclass
class AnalysisEpisode:
    seed: int
    final_info: Dict[str, Any]
    terminated: bool
    truncated: bool
    steps: int
    trace: Any


@dataclass
class AnalysisEpisodes:
    episodes: List[AnalysisEpisode]
    skipped_resets: int


def collect_analysis_episodes(
    env,
    policy,
    *,
    base_seed: int,
    episode_count: int,
    max_steps: int,
    start_trace: Optional[Callable[[Any], Any]] = None,
    update_trace: Optional[Callable[[Any, Any], None]] = None,
) -> AnalysisEpisodes:
    """Collect valid seeded episodes while consistently handling failed A* resets."""
    completed: List[AnalysisEpisode] = []
    skipped_resets = 0
    seed_offset = 0
    maximum_attempts = max(episode_count * 10, episode_count)

    while len(completed) < episode_count and seed_offset < maximum_attempts:
        episode_seed = base_seed + seed_offset
        seed_offset += 1
        try:
            observation, _ = reset_env(env, episode_seed)
        except RuntimeError as exc:
            if is_planner_reset_failure(exc):
                skipped_resets += 1
                continue
            raise

        # Policy memory belongs to the episode that starts after this successful reset.
        reset_policy(policy)
        trace = start_trace(env) if start_trace is not None else None
        final_info: Dict[str, Any] = {}
        terminated = truncated = False
        steps = 0

        for steps in range(1, max_steps + 1):
            action = policy.predict(observation, env) if hasattr(policy, "predict") else env.action_space.sample()
            observation, _, terminated, truncated, final_info = env.step(action)
            if update_trace is not None:
                update_trace(trace, env)
            if terminated or truncated:
                break

        completed.append(
            AnalysisEpisode(
                seed=episode_seed,
                final_info=final_info,
                terminated=bool(terminated),
                truncated=bool(truncated),
                steps=steps,
                trace=trace,
            )
        )

    return AnalysisEpisodes(episodes=completed, skipped_resets=skipped_resets)
