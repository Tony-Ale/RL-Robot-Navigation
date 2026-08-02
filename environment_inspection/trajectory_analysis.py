"""Replay selected episodes and save truthful robot trajectory snapshots."""

import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from environment_inspection.attention_visualization import render_environment_frame
from environment_inspection.output_utils import outcome_from_info, write_csv
from environment_inspection.trajectory_visualization import save_trajectory_snapshot
from training_pipeline.episode_runtime import is_planner_reset_failure, reset_env, reset_policy


TRAJECTORY_FIELDS = ["seed", "step", "simulation_time", "robot_x", "robot_y"]
SUMMARY_FIELDS = [
    "seed",
    "policy_type",
    "checkpoint",
    "steps",
    "success",
    "collision",
    "timeout",
    "outcome",
    "episode_reward",
    "recorded_path_length",
    "metric_path_length",
    "a_star_path_length",
    "a_star_spl",
]


def run_trajectory_analysis(env, policy, config: Mapping[str, Any]) -> Path:
    settings = config["trajectory_analysis"]
    seeds = [int(seed) for seed in settings.get("seeds", [])]
    if not seeds:
        raise ValueError("trajectory_analysis.seeds must contain at least one explicit seed.")
    if len(seeds) != len(set(seeds)):
        raise ValueError("trajectory_analysis.seeds must not contain duplicates.")

    output_dir = Path(settings.get("output_dir", "environment_inspection/trajectory_outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index, seed in enumerate(seeds, start=1):
        print(f"[{index}/{len(seeds)}] Replaying seed {seed} for trajectory analysis.")
        trace = collect_trajectory(
            env,
            policy,
            seed=seed,
            max_steps=int(settings.get("max_steps", 401)),
            require_astar=bool(settings.get("require_astar", True)),
            policy_type=str(config["policy"]["type"]),
            checkpoint=config["policy"].get("checkpoint"),
        )
        episode_dir = output_dir / f"seed_{seed}_{trace['summary']['outcome']}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        write_csv(episode_dir / "trajectory.csv", TRAJECTORY_FIELDS, trace["rows"])
        with open(episode_dir / "summary.json", "w") as file:
            json.dump(trace["summary"], file, indent=2)
        save_trajectory_snapshot(
            trace["initial_frame"],
            trace["robot_path"],
            trace["goal"],
            trace["summary"],
            episode_dir / "trajectory_snapshot.png",
            dpi=int(settings.get("dpi", 180)),
        )
        summaries.append(trace["summary"])

    write_csv(output_dir / "trajectory_summary.csv", SUMMARY_FIELDS, summaries)
    print(f"Trajectory analysis complete: {output_dir}")
    return output_dir


def collect_trajectory(
    env,
    policy,
    seed: int,
    max_steps: int,
    require_astar: bool,
    policy_type: str,
    checkpoint=None,
) -> Dict[str, Any]:
    if max_steps <= 0:
        raise ValueError("trajectory_analysis.max_steps must be positive.")
    try:
        observation, _ = reset_env(env, seed)
    except RuntimeError as exc:
        if is_planner_reset_failure(exc):
            raise RuntimeError(f"Selected trajectory seed {seed} does not produce a valid A* path.") from exc
        raise
    reset_policy(policy)

    base_env = env.unwrapped
    astar_path, astar_length = _reset_astar_reference(env)
    if require_astar and not astar_path:
        raise RuntimeError(
            "Trajectory analysis requires a reset-time A* path, but no active "
            "SocNavAStarWrapper plan was found."
        )

    robot_path = [(float(base_env.robot.x), float(base_env.robot.y))]
    goal = (float(base_env.robot.goal_x), float(base_env.robot.goal_y))
    initial_frame = render_environment_frame(base_env, include_callbacks=True)
    rows = [_trajectory_row(seed, 0, 0.0, robot_path[0])]
    total_reward = 0.0
    final_info: Dict[str, Any] = {}
    ended = False

    for step in range(1, max_steps + 1):
        action = (
            env.action_space.sample()
            if policy is None
            else policy.predict(observation, env)
        )
        observation, reward, terminated, truncated, final_info = env.step(action)
        total_reward += float(reward)
        point = (float(base_env.robot.x), float(base_env.robot.y))
        robot_path.append(point)
        rows.append(
            _trajectory_row(
                seed,
                step,
                step * float(base_env.TIMESTEP),
                point,
            )
        )
        if terminated or truncated:
            ended = True
            break

    outcome = outcome_from_info(final_info, fallback="other" if ended else "max_steps_reached")
    reported_astar_length = _finite_or_none(final_info.get("A_STAR_PATH_LENGTH"))
    summary = {
        "seed": int(seed),
        "policy_type": policy_type,
        "checkpoint": str(Path(checkpoint).resolve()) if checkpoint else None,
        "steps": len(robot_path) - 1,
        "success": bool(final_info.get("SUCCESS", False)),
        "collision": bool(final_info.get("COLLISION", False)),
        "timeout": bool(final_info.get("TIMEOUT", False)),
        "outcome": outcome,
        "episode_reward": total_reward,
        "recorded_path_length": _path_length(robot_path),
        "metric_path_length": _finite_or_none(final_info.get("PATH_LENGTH")),
        "a_star_path_length": reported_astar_length if reported_astar_length is not None else astar_length,
        "a_star_spl": _finite_or_none(final_info.get("A_STAR_SPL")),
        "map_width": float(base_env.MAP_X),
        "map_height": float(base_env.MAP_Y),
    }
    return {
        "rows": rows,
        "summary": summary,
        "robot_path": robot_path,
        "astar_path": astar_path,
        "goal": goal,
        "initial_frame": initial_frame,
    }


def _reset_astar_reference(env):
    current = env
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if "latest_plan" in vars(current):
            plan = vars(current).get("latest_plan")
            if plan is None:
                return [], None
            path = [(float(x), float(y)) for x, y in plan.path_world]
            length = getattr(current, "episode_astar_path_length", None)
            return path, _finite_or_none(length)
        current = getattr(current, "env", None)
    return [], None


def _trajectory_row(seed: int, step: int, simulation_time: float, point: Sequence[float]):
    return {
        "seed": int(seed),
        "step": int(step),
        "simulation_time": float(simulation_time),
        "robot_x": float(point[0]),
        "robot_y": float(point[1]),
    }


def _path_length(points: Sequence[Sequence[float]]) -> float:
    return float(
        sum(
            math.hypot(second[0] - first[0], second[1] - first[1])
            for first, second in zip(points, points[1:])
        )
    )


def _finite_or_none(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None
