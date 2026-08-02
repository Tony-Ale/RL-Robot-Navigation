import argparse
import csv
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from training_pipeline.utils import configure_matplotlib_cache, load_yaml

configure_matplotlib_cache()

from training_pipeline.env_factory import make_eval_env
from training_pipeline.metrics import CSVMetricWriter, NAVIGATION_METRIC_KEYS, mean_or_none, numeric_values
from testing_pipeline.checkpoints import checkpoint_step
from testing_pipeline.metrics import paired_comparison_rows
from testing_pipeline.policies import ORCARobotPolicy
from testing_pipeline.policy_loading import load_learned_agent_policy
from testing_pipeline.runner import run_policy_episodes


RATE_KEYS = [
    "SUCCESS",
    "COLLISION",
    "COLLISION_HUMAN",
    "COLLISION_OBJECT",
    "COLLISION_WALL",
    "OUT_OF_MAP",
    "TIMEOUT",
    "FAILURE_TO_PROGRESS",
]

MEAN_KEYS = ["episode_reward", "episode_length"] + [key for key in NAVIGATION_METRIC_KEYS if key not in RATE_KEYS]

EPISODE_FIELDNAMES = [
    "checkpoint_step",
    "checkpoint_name",
    "checkpoint_path",
    "episode",
    "seed",
    "controller",
    "episode_reward",
    "episode_length",
] + NAVIGATION_METRIC_KEYS

SUMMARY_FIELDNAMES = [
    "checkpoint_step",
    "checkpoint_name",
    "checkpoint_path",
    "controller",
    "episodes_completed",
] + [f"{key.lower()}_rate" for key in RATE_KEYS] + [f"mean_{key.lower()}" for key in MEAN_KEYS]

COMPARISON_FIELDNAMES = [
    "checkpoint_step",
    "checkpoint_name",
    "checkpoint_path",
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

def resolve_checkpoints(run_dir: Path, eval_cfg: Dict[str, Any]) -> List[Path]:
    checkpoint_dir = Path(run_dir) / "checkpoints"
    source = eval_cfg.get("checkpoint_source", "all")
    if source == "all":
        checkpoints = list(checkpoint_dir.glob("ppo_step_*.zip")) + list(checkpoint_dir.glob("ppo_final_step_*.zip"))
    elif source == "step":
        checkpoints = list(checkpoint_dir.glob("ppo_step_*.zip"))
    elif source == "final":
        checkpoints = list(checkpoint_dir.glob("ppo_final_step_*.zip"))
    elif source == "list":
        checkpoints = [checkpoint_dir / name for name in eval_cfg.get("checkpoint_filenames", [])]
    else:
        raise ValueError('offline_evaluation.checkpoint_source must be "all", "step", "final", or "list".')

    missing = [path for path in checkpoints if not path.exists()]
    if missing:
        names = ", ".join(path.name for path in missing)
        raise FileNotFoundError(f"Checkpoint file(s) not found: {names}")
    return sorted(set(checkpoints), key=lambda path: (checkpoint_step(path), path.name))


def offline_seeds(eval_cfg: Dict[str, Any]) -> List[Optional[int]]:
    episodes = int(eval_cfg.get("n_eval_episodes", 100))
    if not eval_cfg.get("fixed_episode_seeds", True):
        return [None for _ in range(episodes)]
    base = int(eval_cfg.get("seed_base", 11042))
    return [base + index for index in range(episodes)]


def run_offline_evaluation(config_path: str) -> Dict[str, Any]:
    wrapper_config = load_yaml(config_path)
    eval_cfg = wrapper_config.get("offline_evaluation", {})
    training_config = load_yaml(eval_cfg.get("training_config_path", "training_pipeline/config.yaml"))
    run_dir = Path(eval_cfg["run_dir"])
    output_dir = run_dir / eval_cfg.get("output_dir", "offline_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_csv = output_dir / eval_cfg.get("episode_csv", "checkpoint_episode_metrics.csv")
    summary_csv = output_dir / eval_cfg.get("summary_csv", "checkpoint_summary_metrics.csv")
    comparison_csv = output_dir / eval_cfg.get("comparison_csv", "checkpoint_orca_comparison.csv")
    mode = eval_cfg.get("mode", "evaluate_and_plot")

    if mode not in {"evaluate_and_plot", "evaluate_only", "plot_only"}:
        raise ValueError('offline_evaluation.mode must be "evaluate_and_plot", "evaluate_only", or "plot_only".')

    if mode != "plot_only":
        if eval_cfg.get("overwrite_existing", False):
            clear_outputs([episode_csv, summary_csv, comparison_csv])
        evaluated = evaluated_checkpoints(summary_csv)
        checkpoints = [path for path in resolve_checkpoints(run_dir, eval_cfg) if str(path) not in evaluated]
        evaluate_checkpoints(training_config, eval_cfg, checkpoints, episode_csv, summary_csv, comparison_csv)

    if mode != "evaluate_only":
        generate_plots(eval_cfg, output_dir)

    return {"output_dir": str(output_dir), "summary_csv": str(summary_csv), "episode_csv": str(episode_csv)}


def evaluate_checkpoints(
    training_config: Dict[str, Any],
    eval_cfg: Dict[str, Any],
    checkpoints: Iterable[Path],
    episode_csv: Path,
    summary_csv: Path,
    comparison_csv: Path,
) -> None:
    seeds = offline_seeds(eval_cfg)
    include_orca = bool(eval_cfg.get("include_orca", True))
    progress_cfg = eval_cfg.get("progress", {})
    show_progress = progress_cfg.get("enabled", True)

    episode_writer = CSVMetricWriter(episode_csv, EPISODE_FIELDNAMES)
    summary_writer = CSVMetricWriter(summary_csv, SUMMARY_FIELDNAMES)
    comparison_writer = CSVMetricWriter(comparison_csv, COMPARISON_FIELDNAMES) if include_orca else None

    checkpoints = list(checkpoints)
    started_at = time.time()
    for index, checkpoint in enumerate(checkpoints, start=1):
        step = checkpoint_step(checkpoint)
        if show_progress:
            print(f"[{index}/{len(checkpoints)}] Evaluating {checkpoint.name} on {len(seeds)} episode(s).")
        agent_env = make_eval_env(training_config)
        try:
            policy = load_learned_agent_policy(checkpoint, agent_env, eval_cfg, training_config)
            agent_rows = run_policy_episodes(agent_env, policy, seeds, checkpoint, csv_path=None)
        finally:
            agent_env.close()
        write_checkpoint_rows(episode_writer, checkpoint, step, agent_rows)
        summary_writer.write(summary_row(checkpoint, step, "learned_agent", agent_rows))

        if include_orca:
            valid_seeds = [row["seed"] for row in agent_rows]
            orca_env = make_eval_env(training_config)
            try:
                orca_rows = run_policy_episodes(
                    orca_env,
                    ORCARobotPolicy(),
                    valid_seeds,
                    checkpoint_path=None,
                    csv_path=None,
                    fill_valid_episodes=False,
                )
            finally:
                orca_env.close()
            write_checkpoint_rows(episode_writer, checkpoint, step, orca_rows)
            summary_writer.write(summary_row(checkpoint, step, "orca", orca_rows))
            for row in paired_comparison_rows(agent_rows, orca_rows):
                row.update(checkpoint_metadata(checkpoint, step))
                comparison_writer.write(row)
        if show_progress:
            print_progress_eta(index, len(checkpoints), started_at)


def write_checkpoint_rows(writer: CSVMetricWriter, checkpoint: Path, step: int, rows: Iterable[Dict[str, Any]]) -> None:
    for row in rows:
        output = dict(row)
        output.update(checkpoint_metadata(checkpoint, step))
        writer.write(output)


def checkpoint_metadata(checkpoint: Path, step: int) -> Dict[str, Any]:
    return {
        "checkpoint_step": int(step),
        "checkpoint_name": Path(checkpoint).name,
        "checkpoint_path": str(checkpoint),
    }


def summary_row(checkpoint: Path, step: int, controller: str, rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(rows)
    row = checkpoint_metadata(checkpoint, step)
    row["controller"] = controller
    row["episodes_completed"] = len(rows)
    for key in RATE_KEYS:
        row[f"{key.lower()}_rate"] = mean_or_none(numeric_values(rows, key))
    for key in MEAN_KEYS:
        row[f"mean_{key.lower()}"] = mean_or_none(numeric_values(rows, key))
    return row


def evaluated_checkpoints(summary_csv: Path) -> set:
    rows = read_csv_rows(summary_csv)
    return {row["checkpoint_path"] for row in rows if row.get("controller") == "learned_agent" and row.get("checkpoint_path")}


def clear_outputs(paths: Iterable[Path]) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def print_progress_eta(done: int, total: int, started_at: float) -> None:
    elapsed = max(time.time() - started_at, 0.0)
    if done <= 0 or total <= 0:
        return
    remaining = max(total - done, 0)
    seconds_per_checkpoint = elapsed / done
    eta = remaining * seconds_per_checkpoint
    print(
        f"Completed {done}/{total} checkpoint(s). "
        f"Elapsed: {format_seconds(elapsed)}. ETA: {format_seconds(eta)}."
    )


def format_seconds(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def generate_plots(eval_cfg: Dict[str, Any], output_dir: Path) -> None:
    plot_cfg = eval_cfg.get("plots", {})
    from testing_pipeline.evaluation_plots import generate_evaluation_plots

    generate_evaluation_plots(plot_cfg, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved PPO or stateful PPO checkpoints and plot metrics.")
    parser.add_argument("--config", default="testing_pipeline/offline_evaluation_config.yaml", help="Offline evaluation YAML config.")
    args = parser.parse_args()
    result = run_offline_evaluation(args.config)
    print(f"Offline evaluation complete. Outputs: {result}")


if __name__ == "__main__":
    main()
