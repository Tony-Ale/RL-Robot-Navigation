import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from training_pipeline.utils import configure_matplotlib_cache, load_yaml, save_json

configure_matplotlib_cache()

from training_pipeline.env_factory import make_eval_env
from training_pipeline.episode_runtime import is_planner_reset_failure, reset_env, reset_policy
from training_pipeline.metrics import CSVMetricWriter, NAVIGATION_METRIC_KEYS
from testing_pipeline.checkpoints import checkpoint_step
from testing_pipeline.metrics import (
    build_episode_row,
    paired_comparison_rows,
    summarize_comparison,
    summarize_rows,
)
from testing_pipeline.policies import ORCARobotPolicy
from testing_pipeline.policy_loading import load_learned_agent_policy


AGENT_FIELDNAMES = [
    "episode",
    "seed",
    "controller",
    "episode_reward",
    "episode_length",
    "checkpoint_path",
] + NAVIGATION_METRIC_KEYS

COMPARISON_FIELDNAMES = [
    "seed",
    "agent_SUCCESS",
    "orca_SUCCESS",
    "agent_COLLISION",
    "orca_COLLISION",
    "agent_TIMEOUT",
    "orca_TIMEOUT",
    "agent_PATH_LENGTH",
    "orca_PATH_LENGTH",
    "agent_A_STAR_PATH_LENGTH",
    "orca_A_STAR_PATH_LENGTH",
    "agent_A_STAR_SPL",
    "orca_A_STAR_SPL",
    "agent_TIME_TO_REACH_GOAL",
    "orca_TIME_TO_REACH_GOAL",
    "agent_SPL",
    "orca_SPL",
    "agent_STL",
    "orca_STL",
    "delta_PATH_LENGTH",
    "delta_TIME_TO_REACH_GOAL",
    "delta_A_STAR_SPL",
    "delta_SPL",
    "delta_STL",
    "winner",
]


def run_testing(
    config: Dict[str, Any],
    run_dir: Path,
    checkpoint_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run held-out testing for the learned agent and optional ORCA baseline."""
    testing_cfg = config.get("testing", {})
    output_dir = Path(run_dir) / "testing"
    output_dir.mkdir(parents=True, exist_ok=True)

    validate_testing_config(testing_cfg)
    if testing_cfg.get("overwrite_existing", True):
        clear_testing_outputs(output_dir, testing_cfg)

    checkpoint = resolve_checkpoint_path(testing_cfg, run_dir, checkpoint_path)
    seeds = test_seeds(config, testing_cfg)

    agent_env = make_eval_env(config)
    try:
        learned_policy = load_learned_agent_policy(checkpoint, agent_env, testing_cfg, config)
        agent_rows = run_policy_episodes(
            env=agent_env,
            policy=learned_policy,
            seeds=seeds,
            checkpoint_path=checkpoint,
            csv_path=output_dir / testing_cfg.get("agent_csv", "test_agent_metrics.csv"),
        )
    finally:
        agent_env.close()
    valid_seeds = [row["seed"] for row in agent_rows]

    summary = {
        "checkpoint_path": str(checkpoint),
        "test_seeds": valid_seeds,
        "learned_agent": summarize_rows(agent_rows),
    }

    if testing_cfg.get("compare_with_baseline", True):
        baseline = testing_cfg.get("baseline", "orca")
        if baseline != "orca":
            raise ValueError('testing.baseline must be "orca"; optional later baselines are not implemented.')
        orca_env = make_eval_env(config)
        try:
            orca_rows = run_policy_episodes(
                env=orca_env,
                policy=ORCARobotPolicy(),
                seeds=valid_seeds,
                checkpoint_path=None,
                csv_path=output_dir / testing_cfg.get("baseline_csv", "test_orca_metrics.csv"),
                fill_valid_episodes=False,
            )
        finally:
            orca_env.close()
        comparison_rows = paired_comparison_rows(agent_rows, orca_rows)
        comparison_writer = CSVMetricWriter(
            output_dir / testing_cfg.get("comparison_csv", "agent_vs_orca_comparison.csv"),
            COMPARISON_FIELDNAMES,
        )
        for row in comparison_rows:
            comparison_writer.write(row)
        summary["orca"] = summarize_rows(orca_rows)
        summary["comparison"] = summarize_comparison(comparison_rows)

    save_json(output_dir / testing_cfg.get("summary_json", "test_summary.json"), summary)
    return summary


def validate_testing_config(testing_cfg: Dict[str, Any]) -> None:
    if testing_cfg.get("compare_with_baseline", True) and not testing_cfg.get("fixed_test_seeds", True):
        raise ValueError("ORCA baseline comparison requires testing.fixed_test_seeds: true.")


def clear_testing_outputs(output_dir: Path, testing_cfg: Dict[str, Any]) -> None:
    filenames = [
        testing_cfg.get("agent_csv", "test_agent_metrics.csv"),
        testing_cfg.get("baseline_csv", "test_orca_metrics.csv"),
        testing_cfg.get("comparison_csv", "agent_vs_orca_comparison.csv"),
        testing_cfg.get("summary_json", "test_summary.json"),
    ]
    for filename in filenames:
        path = output_dir / filename
        if path.exists():
            path.unlink()


def run_policy_episodes(
    env,
    policy,
    seeds: Iterable[Optional[int]],
    checkpoint_path: Optional[Path],
    csv_path: Optional[Path],
    fill_valid_episodes: bool = True,
) -> List[Dict[str, Any]]:
    writer = CSVMetricWriter(csv_path, AGENT_FIELDNAMES) if csv_path is not None else None
    rows = []
    seed_list = list(seeds)
    target_episodes = len(seed_list)
    seed_index = 0
    next_seed = _next_seed_after(seed_list)

    while len(rows) < target_episodes:
        if seed_index < len(seed_list):
            seed = seed_list[seed_index]
            seed_index += 1
        elif fill_valid_episodes and next_seed is not None:
            seed = next_seed
            next_seed += 1
        else:
            break

        try:
            obs, _ = reset_env(env, seed)
        except RuntimeError as exc:
            if seed is not None and is_planner_reset_failure(exc):
                print(f"Skipping seed {seed}: no waypoints generated.")
                continue
            raise

        reset_policy(policy)

        done = False
        episode_reward = 0.0
        episode_length = 0
        final_info: Dict[str, Any] = {}
        while not done:
            action = policy.predict(obs, env=env)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            episode_reward += float(reward)
            episode_length += 1
            final_info = info

        row = build_episode_row(
            episode=len(rows) + 1,
            seed=seed,
            controller=policy.controller_name,
            episode_reward=episode_reward,
            episode_length=episode_length,
            final_info=final_info,
            checkpoint_path=checkpoint_path,
        )
        if writer is not None:
            writer.write(row)
        rows.append(row)
    return rows


def _next_seed_after(seeds: List[Optional[int]]) -> Optional[int]:
    if not seeds or seeds[-1] is None:
        return None
    return int(seeds[-1]) + 1


def test_seeds(config: Dict[str, Any], testing_cfg: Dict[str, Any]) -> List[Optional[int]]:
    n_episodes = int(testing_cfg.get("n_test_episodes", 50))
    if not testing_cfg.get("fixed_test_seeds", True):
        return [None for _ in range(n_episodes)]
    base = int(config["experiment"]["seed"]) + int(testing_cfg.get("test_seed_offset", 20000))
    return [base + index for index in range(n_episodes)]


def resolve_checkpoint_path(
    testing_cfg: Dict[str, Any],
    run_dir: Path,
    checkpoint_path: Optional[Path] = None,
) -> Path:
    if checkpoint_path is not None:
        return Path(checkpoint_path)
    if testing_cfg.get("checkpoint_path"):
        return Path(testing_cfg["checkpoint_path"])
    checkpoints = sorted(
        (Path(run_dir) / "checkpoints").glob("ppo_final_step_*.zip"),
        key=checkpoint_step,
    )
    if checkpoints:
        return checkpoints[-1]
    raise ValueError("No testing checkpoint was provided and no final checkpoint was found in the run directory.")


def main():
    parser = argparse.ArgumentParser(description="Test a trained navigation agent against ORCA.")
    parser.add_argument("--config", default="training_pipeline/config.yaml", help="Path to the training/testing YAML config.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing checkpoints and where test outputs are saved.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path. Defaults to the latest final checkpoint.")
    args = parser.parse_args()

    config = load_yaml(args.config)
    summary = run_testing(
        config=config,
        run_dir=Path(args.run_dir),
        checkpoint_path=None if args.checkpoint is None else Path(args.checkpoint),
    )
    print(f"Testing complete. Summary: {summary}")


if __name__ == "__main__":
    main()
