"""Replay stateful checkpoints and produce synchronized attention diagnostics."""

import argparse
import csv
import math
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment_inspection.attention_visualization import render_labelled_frame, save_episode_outputs
from environment_inspection.output_utils import outcome_from_info, write_csv
from testing_pipeline.policy_loading import load_learned_agent_policy
from training_pipeline.episode_runtime import is_planner_reset_failure, reset_env
from training_pipeline.env_factory import make_eval_env
from training_pipeline.utils import load_yaml


TIMESTEP_FIELDS = [
    "seed", "step", "simulation_time", "slot", "label", "entity_key", "entity_index",
    "valid", "attention", "relative_x", "relative_y", "world_x", "world_y", "radius",
    "distance", "clearance", "is_nearest", "is_top_attention", "linear_action", "angular_action",
    "success", "collision", "timeout", "outcome",
]

SUMMARY_FIELDS = [
    "seed", "steps", "success", "collision", "timeout", "outcome", "peak_attention",
    "peak_attention_step", "minimum_clearance", "minimum_clearance_step",
    "largest_attention_switch", "attention_switch_step", "mean_normalized_attention_entropy",
    "nearest_entity_top_attention_fraction", "mean_nearest_entity_attention",
    "spearman_attention_inverse_distance",
]


def run_attention_analysis(config_path: str) -> Path:
    settings = load_yaml(config_path)["attention_analysis"]
    pipeline_config = load_yaml(settings["training_config_path"])
    checkpoint = Path(settings["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Stateful checkpoint not found: {checkpoint}")

    cases = resolve_analysis_cases(settings)
    if not cases:
        raise ValueError("No attention-analysis seeds were configured or selected.")

    output_dir = Path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    timestep_path = output_dir / "attention_timesteps.csv"
    summary_path = output_dir / "attention_summary.csv"
    all_records = []
    summaries = []

    env = make_eval_env(pipeline_config)
    try:
        policy = load_learned_agent_policy(
            checkpoint,
            env,
            {
                "policy_type": "stateful_ppo",
                "deterministic": bool(settings.get("deterministic", True)),
                "device": settings.get("device", "auto"),
            },
            pipeline_config,
        )
        for index, case in enumerate(cases, start=1):
            seed = case["seed"]
            print(f"[{index}/{len(cases)}] Replaying seed {seed} for attention analysis.")
            records, summary, events = collect_episode_trace(
                env,
                policy,
                seed,
                int(settings.get("max_steps", 401)),
                bool(settings.get("frames", {}).get("enabled", True)),
                bool(settings.get("frames", {}).get("label_entities", True)),
            )
            all_records.extend(records)
            summaries.append(summary)
            folder_label = case["category"] or summary["outcome"]
            save_episode_outputs(
                seed,
                records,
                events,
                output_dir,
                settings.get("plots", {}),
                folder_label=folder_label,
            )
    finally:
        env.close()

    write_csv(timestep_path, TIMESTEP_FIELDS, all_records)
    write_csv(summary_path, SUMMARY_FIELDS, summaries)
    print(f"Attention analysis complete: {output_dir}")
    return output_dir


def resolve_analysis_seeds(settings: Mapping) -> List[int]:
    return [case["seed"] for case in resolve_analysis_cases(settings)]


def resolve_analysis_cases(settings: Mapping) -> List[Dict]:
    explicit = [int(seed) for seed in settings.get("seeds", [])]
    if explicit:
        return [{"seed": seed, "category": None} for seed in _unique(explicit)]

    selection = settings.get("selection", {})
    comparison_path = Path(selection.get("comparison_csv", ""))
    if not comparison_path.is_file():
        raise FileNotFoundError(
            "Explicit attention-analysis seeds are empty and the final-test comparison CSV "
            f"does not exist: {comparison_path}"
        )
    with open(comparison_path, newline="") as file:
        rows = list(csv.DictReader(file))

    selected = []
    seen = set()
    per_category = int(selection.get("per_category", 1))
    selection_mode = selection.get("mode", "first")
    if selection_mode not in {"first", "random"}:
        raise ValueError("attention selection.mode must be 'first' or 'random'.")
    randomizer = random.Random(int(selection.get("random_seed", 0)))
    for category in selection.get("categories", []):
        matches = sorted(
            (row for row in rows if _matches_category(row, category)),
            key=lambda row: int(row["seed"]),
        )
        candidates = [int(row["seed"]) for row in matches if int(row["seed"]) not in seen]
        if selection_mode == "random":
            randomizer.shuffle(candidates)
        chosen = candidates[:per_category]
        if not chosen:
            print(f"No final-test seed matched attention category '{category}'.")
        for seed in chosen:
            if seed not in seen:
                selected.append({"seed": seed, "category": category})
                seen.add(seed)
    return selected


def collect_episode_trace(env, policy, seed: int, max_steps: int, capture_frames: bool, label_entities: bool):
    try:
        observation, _ = reset_env(env, seed)
    except RuntimeError as exc:
        if is_planner_reset_failure(exc):
            raise RuntimeError(f"Selected seed {seed} no longer produces a valid A* path.") from exc
        raise
    policy.reset()
    base_env = env.unwrapped
    records = []
    events: Dict[str, Dict] = {}
    previous_attention = None
    entropy_values = []
    nearest_attention_values = []
    nearest_top_hits = 0
    valid_steps = 0
    correlation_attention = []
    correlation_inverse_distance = []
    final_info = {}

    for step in range(max_steps):
        entities = extract_entity_rows(observation, policy.model.policy.entity_keys, policy.model.policy.entity_feature_dim, base_env)
        action = np.asarray(policy.predict(observation, env), dtype=np.float32).reshape(-1)
        attention = policy.model.policy.architecture.last_attention_weights
        if attention is None:
            raise RuntimeError("The stateful architecture did not expose attention weights after prediction.")
        attention = attention.detach().cpu().numpy().reshape(-1)
        if len(attention) != len(entities):
            raise RuntimeError(f"Attention/entity mismatch: {len(attention)} weights for {len(entities)} slots.")

        valid_indices = [index for index, entity in enumerate(entities) if entity["valid"]]
        top_slot = max(valid_indices, key=lambda index: attention[index]) if valid_indices else None
        nearest_slot = min(valid_indices, key=lambda index: entities[index]["clearance"]) if valid_indices else None
        for index, entity in enumerate(entities):
            entity["attention"] = float(attention[index])
            entity["is_top_attention"] = index == top_slot
            entity["is_nearest"] = index == nearest_slot

        frame = render_labelled_frame(base_env, entities if label_entities else (), top_slot) if capture_frames else None
        simulation_time = float(getattr(base_env, "ticks", step)) * float(base_env.TIMESTEP)
        _update_events(events, step, simulation_time, entities, attention, previous_attention, frame)

        if valid_indices:
            valid_weights = attention[valid_indices]
            entropy_values.append(_normalized_entropy(valid_weights))
            nearest_attention_values.append(float(attention[nearest_slot]))
            nearest_top_hits += int(nearest_slot == top_slot)
            valid_steps += 1
            for index in valid_indices:
                correlation_attention.append(float(attention[index]))
                correlation_inverse_distance.append(1.0 / max(float(entities[index]["distance"]), 1e-8))

        for entity in entities:
            records.append(
                {
                    "seed": seed,
                    "step": step,
                    "simulation_time": simulation_time,
                    **entity,
                    "linear_action": float(action[0]),
                    "angular_action": float(action[-1]),
                }
            )

        next_observation, _, terminated, truncated, final_info = env.step(action)
        previous_attention = attention.copy()
        if terminated or truncated:
            if frame is not None:
                events["pre_terminal"] = _event(step, simulation_time, frame, 0.0)
            break
        observation = next_observation
    else:
        final_info = {"SUCCESS": False, "COLLISION": False, "TIMEOUT": False}

    outcome = outcome_from_info(final_info)
    for row in records:
        row.update(
            success=bool(final_info.get("SUCCESS", False)),
            collision=bool(final_info.get("COLLISION", False)),
            timeout=bool(final_info.get("TIMEOUT", False)),
            outcome=outcome,
        )

    summary = {
        "seed": seed,
        "steps": 0 if not records else max(int(row["step"]) for row in records) + 1,
        "success": bool(final_info.get("SUCCESS", False)),
        "collision": bool(final_info.get("COLLISION", False)),
        "timeout": bool(final_info.get("TIMEOUT", False)),
        "outcome": outcome,
        "peak_attention": _event_value(events, "peak_attention"),
        "peak_attention_step": _event_step(events, "peak_attention"),
        "minimum_clearance": _event_value(events, "minimum_clearance"),
        "minimum_clearance_step": _event_step(events, "minimum_clearance"),
        "largest_attention_switch": _event_value(events, "attention_switch"),
        "attention_switch_step": _event_step(events, "attention_switch"),
        "mean_normalized_attention_entropy": _mean_or_none(entropy_values),
        "nearest_entity_top_attention_fraction": nearest_top_hits / valid_steps if valid_steps else None,
        "mean_nearest_entity_attention": _mean_or_none(nearest_attention_values),
        "spearman_attention_inverse_distance": _spearman(correlation_attention, correlation_inverse_distance),
    }
    return records, summary, events


def extract_entity_rows(observation: Mapping, entity_keys: Iterable[str], feature_dim: int, base_env) -> List[Dict]:
    """Reproduce the policy's concatenation order and mask directly from observations."""
    output = []
    robot_radius = float(np.asarray(observation["robot"]).reshape(-1)[-1])
    theta = float(base_env.robot.orientation)
    cos_theta, sin_theta = math.cos(theta), math.sin(theta)
    slot = 0
    for key in entity_keys:
        values = np.asarray(observation[key], dtype=np.float32).reshape(-1, feature_dim)
        for entity_index, row in enumerate(values):
            valid = bool(np.sum(np.abs(row)) > 1e-8)
            relative_x, relative_y = float(row[6]), float(row[7])
            world_x = float(base_env.robot.x) + cos_theta * relative_x - sin_theta * relative_y
            world_y = float(base_env.robot.y) + sin_theta * relative_x + cos_theta * relative_y
            distance = math.hypot(relative_x, relative_y)
            radius = abs(float(row[10]))
            output.append(
                {
                    "slot": slot,
                    "label": f"E{slot + 1:02d}",
                    "entity_key": key,
                    "entity_index": entity_index,
                    "valid": valid,
                    "relative_x": relative_x if valid else None,
                    "relative_y": relative_y if valid else None,
                    "world_x": world_x if valid else None,
                    "world_y": world_y if valid else None,
                    "radius": radius if valid else None,
                    "distance": distance if valid else None,
                    "clearance": distance - robot_radius - radius if valid else None,
                }
            )
            slot += 1
    return output


def _update_events(events, step, simulation_time, entities, attention, previous_attention, frame):
    if frame is None:
        return
    valid = [index for index, entity in enumerate(entities) if entity["valid"]]
    if not valid:
        return
    peak = max(float(attention[index]) for index in valid)
    if "peak_attention" not in events or peak > events["peak_attention"]["value"]:
        events["peak_attention"] = _event(step, simulation_time, frame, peak)

    clearance = min(float(entities[index]["clearance"]) for index in valid)
    if "minimum_clearance" not in events or clearance < events["minimum_clearance"]["value"]:
        events["minimum_clearance"] = _event(step, simulation_time, frame, clearance)

    if previous_attention is not None:
        switch = float(np.sum(np.abs(attention - previous_attention)))
        if "attention_switch" not in events or switch > events["attention_switch"]["value"]:
            events["attention_switch"] = _event(step, simulation_time, frame, switch)


def _event(step, simulation_time, frame, value):
    return {"step": int(step), "simulation_time": float(simulation_time), "frame": frame.copy(), "value": float(value)}


def _matches_category(row: Mapping[str, str], category: str) -> bool:
    agent_success = _csv_bool(row.get("agent_SUCCESS"))
    orca_success = _csv_bool(row.get("orca_SUCCESS"))
    agent_collision = _csv_bool(row.get("agent_COLLISION"))
    agent_timeout = _csv_bool(row.get("agent_TIMEOUT"))
    rules = {
        "agent_success_orca_failure": agent_success and not orca_success,
        "agent_collision": agent_collision,
        "agent_timeout": agent_timeout,
        "both_success": agent_success and orca_success,
        "both_failure": not agent_success and not orca_success,
    }
    if category not in rules:
        raise ValueError(f"Unknown attention seed-selection category: {category}")
    return rules[category]


def _csv_bool(value) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true", "yes"}


def _normalized_entropy(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights[weights > 0]
    if len(weights) <= 1:
        return 0.0
    return float(-np.sum(weights * np.log(weights)) / np.log(len(weights)))


def _spearman(left: Sequence[float], right: Sequence[float]):
    if len(left) < 2 or len(right) != len(left):
        return None
    left_rank = _rankdata(np.asarray(left, dtype=np.float64))
    right_rank = _rankdata(np.asarray(right, dtype=np.float64))
    if np.std(left_rank) <= 1e-12 or np.std(right_rank) <= 1e-12:
        return None
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _unique(values: Iterable[int]) -> List[int]:
    return list(dict.fromkeys(int(value) for value in values))


def _mean_or_none(values):
    return None if not values else float(np.mean(values))


def _event_value(events, name):
    return None if name not in events else events[name]["value"]


def _event_step(events, name):
    return None if name not in events else events[name]["step"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create stateful social-attention heatmaps and synchronized frames.")
    parser.add_argument(
        "--config",
        default="environment_inspection/attention_analysis_config.yaml",
        help="Attention-analysis YAML configuration.",
    )
    arguments = parser.parse_args()
    run_attention_analysis(arguments.config)


if __name__ == "__main__":
    main()
