import numpy as np

from environment_inspection.episode_analysis import collect_analysis_episodes
from environment_inspection.failure_analysis import failure_type


def run_stall_analysis(env, policy, config):
    cfg = config["stall_analysis"]
    base_seed = int(cfg["seed"])
    episodes = int(cfg["episodes"])
    max_steps = int(cfg["max_steps"])
    window_steps = [int(value) for value in cfg["window_steps"]]
    thresholds = [float(value) for value in cfg["displacement_thresholds"]]

    records = []
    print(
        "Stall analysis started: "
        f"policy={config['policy']['type']} seed={base_seed} episodes={episodes} "
        f"max_steps={max_steps} windows={window_steps} thresholds={thresholds}"
    )

    collection = collect_analysis_episodes(
        env,
        policy,
        base_seed=base_seed,
        episode_count=episodes,
        max_steps=max_steps,
        start_trace=lambda active_env: [_robot_position(active_env)],
        update_trace=lambda positions, active_env: positions.append(_robot_position(active_env)),
    )
    for episode in collection.episodes:
        records.append(
            {
                "seed": episode.seed,
                "success": bool(episode.final_info.get("SUCCESS", False)),
                "failure_type": failure_type(
                    episode.final_info,
                    episode.terminated,
                    episode.truncated,
                ),
                "steps": episode.steps,
                "positions": np.asarray(episode.trace, dtype=np.float32),
            }
        )

    print_stall_summary(
        records,
        requested=episodes,
        completed=len(collection.episodes),
        skipped_resets=collection.skipped_resets,
        window_steps=window_steps,
        thresholds=thresholds,
    )


def _robot_position(env):
    robot = env.unwrapped.robot
    return float(robot.x), float(robot.y)


def rolling_displacements(positions, window_steps):
    if len(positions) <= window_steps:
        return np.zeros((0,), dtype=np.float32)
    deltas = positions[window_steps:] - positions[:-window_steps]
    return np.linalg.norm(deltas, axis=1)


def longest_true_run(values):
    longest = 0
    current = 0
    for value in values:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def print_stall_summary(records, requested, completed, skipped_resets, window_steps, thresholds):
    failed = [record for record in records if not record["success"]]
    successful = [record for record in records if record["success"]]

    print("\nStall analysis summary")
    print(f"  requested_episodes: {requested}")
    print(f"  completed_episodes: {completed}")
    print(f"  skipped_reset_failures: {skipped_resets}")
    print(f"  successful_episodes: {len(successful)}")
    print(f"  failed_episodes: {len(failed)}")

    if not records:
        print("  no completed episodes recorded.")
        return

    print("\nDisplacement summary")
    print("  window | failed_min_m median/p25 | success_min_m median/p25")
    for window in window_steps:
        failed_stats = _min_displacement_stats(failed, window)
        success_stats = _min_displacement_stats(successful, window)
        print(
            f"  {window:>6} | "
            f"{_format_stats(failed_stats):>22} | "
            f"{_format_stats(success_stats):>22}"
        )

    rows = _threshold_rows(failed, successful, window_steps, thresholds)
    print("\nThreshold hit summary")
    print("  window threshold | failed_hit_rate | success_hit_rate | failed_longest median/max")
    for row in rows:
        print(
            f"  {row['window']:>6} {row['threshold']:>9.4f} | "
            f"{row['failed_hits']:>4}/{row['failed_total']:<4} {row['failed_rate']:>7.2%} | "
            f"{row['success_hits']:>4}/{row['success_total']:<4} {row['success_rate']:>7.2%} | "
            f"{row['failed_longest_median']:>6.1f}/{row['failed_longest_max']}"
        )

    _print_recommendation(rows)


def _min_displacement_stats(records, window):
    values = []
    for record in records:
        displacements = rolling_displacements(record["positions"], window)
        if displacements.size == 0:
            continue
        values.append(float(np.min(displacements)))

    if not values:
        return None

    values = np.asarray(values, dtype=np.float32)
    return {
        "count": len(values),
        "median": float(np.median(values)),
        "p25": float(np.percentile(values, 25)),
    }


def _format_stats(stats):
    if stats is None:
        return "n/a"
    return f"{stats['median']:.4f}/{stats['p25']:.4f}"


def _threshold_rows(failed, successful, window_steps, thresholds):
    rows = []
    for window in window_steps:
        for threshold in thresholds:
            failed_stats = _threshold_stats(failed, window, threshold)
            success_stats = _threshold_stats(successful, window, threshold)
            rows.append(
                {
                    "window": window,
                    "threshold": threshold,
                    "failed_hits": failed_stats["hits"],
                    "failed_total": failed_stats["total"],
                    "failed_rate": failed_stats["rate"],
                    "failed_longest_median": failed_stats["longest_median"],
                    "failed_longest_max": failed_stats["longest_max"],
                    "success_hits": success_stats["hits"],
                    "success_total": success_stats["total"],
                    "success_rate": success_stats["rate"],
                }
            )
    return rows


def _threshold_stats(records, window, threshold):
    if not records:
        return {"hits": 0, "total": 0, "rate": 0.0, "longest_median": 0.0, "longest_max": 0}

    hits = 0
    longest_runs = []
    for record in records:
        displacements = rolling_displacements(record["positions"], window)
        if displacements.size == 0:
            continue
        stalled = displacements < threshold
        if np.any(stalled):
            hits += 1
        longest_runs.append(longest_true_run(stalled))

    if not longest_runs:
        return {"hits": 0, "total": 0, "rate": 0.0, "longest_median": 0.0, "longest_max": 0}

    total = len(longest_runs)
    return {
        "hits": hits,
        "total": total,
        "rate": hits / total,
        "longest_median": float(np.median(longest_runs)),
        "longest_max": int(np.max(longest_runs)),
    }


def _print_recommendation(rows):
    candidates = [row for row in rows if row["failed_total"] > 0]
    if not candidates:
        print("\nMost common failed stall range: n/a, no failed episodes with enough steps.")
        return

    best = max(
        candidates,
        key=lambda row: (
            row["failed_rate"],
            row["failed_longest_median"],
            -row["success_rate"],
        ),
    )
    print("\nMost common failed stall range")
    print(
        f"  window_steps={best['window']} min_displacement={best['threshold']:.4f}m "
        f"caught_failed={best['failed_hits']}/{best['failed_total']} ({best['failed_rate']:.2%}) "
        f"caught_success={best['success_hits']}/{best['success_total']} ({best['success_rate']:.2%})"
    )
