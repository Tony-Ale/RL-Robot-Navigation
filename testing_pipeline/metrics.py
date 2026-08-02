from numbers import Real
from typing import Any, Dict, Iterable, List

from training_pipeline.metrics import NAVIGATION_METRIC_KEYS, mean_or_none, numeric_values


SUMMARY_RATE_KEYS = ["SUCCESS", "COLLISION", "TIMEOUT"]
SUMMARY_MEAN_KEYS = [
    "episode_reward",
    "episode_length",
    "PATH_LENGTH",
    "A_STAR_PATH_LENGTH",
    "A_STAR_SPL",
    "SPL",
    "STL",
    "TIME_TO_REACH_GOAL",
    "MINIMUM_DISTANCE_TO_HUMAN",
    "PERSONAL_SPACE_COMPLIANCE",
]


def build_episode_row(
    episode: int,
    seed,
    controller: str,
    episode_reward: float,
    episode_length: int,
    final_info: Dict[str, Any],
    checkpoint_path=None,
) -> Dict[str, Any]:
    row = {
        "episode": int(episode),
        "seed": seed,
        "controller": controller,
        "episode_reward": float(episode_reward),
        "episode_length": int(episode_length),
        "checkpoint_path": "" if checkpoint_path is None else str(checkpoint_path),
    }
    for key in NAVIGATION_METRIC_KEYS:
        row[key] = final_info.get(key)
    return row


def summarize_rows(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(rows)
    summary = {
        "episodes_requested": len(rows),
        "episodes_completed": len(rows),
    }
    for key in SUMMARY_RATE_KEYS:
        values = numeric_values(rows, key)
        summary[f"{key.lower()}_rate"] = mean_or_none(values)
    for key in SUMMARY_MEAN_KEYS:
        values = numeric_values(rows, key)
        summary[f"mean_{key.lower()}"] = mean_or_none(values)
    return summary


def paired_comparison_rows(agent_rows: List[Dict[str, Any]], orca_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    agent_by_seed = {row["seed"]: row for row in agent_rows}
    orca_by_seed = {row["seed"]: row for row in orca_rows}
    rows = []
    for seed in sorted(set(agent_by_seed) & set(orca_by_seed)):
        agent = agent_by_seed[seed]
        orca = orca_by_seed[seed]
        row = {
            "seed": seed,
            "agent_SUCCESS": agent.get("SUCCESS"),
            "orca_SUCCESS": orca.get("SUCCESS"),
            "agent_COLLISION": agent.get("COLLISION"),
            "orca_COLLISION": orca.get("COLLISION"),
            "agent_TIMEOUT": agent.get("TIMEOUT"),
            "orca_TIMEOUT": orca.get("TIMEOUT"),
            "agent_PATH_LENGTH": agent.get("PATH_LENGTH"),
            "orca_PATH_LENGTH": orca.get("PATH_LENGTH"),
            "agent_A_STAR_PATH_LENGTH": agent.get("A_STAR_PATH_LENGTH"),
            "orca_A_STAR_PATH_LENGTH": orca.get("A_STAR_PATH_LENGTH"),
            "agent_A_STAR_SPL": agent.get("A_STAR_SPL"),
            "orca_A_STAR_SPL": orca.get("A_STAR_SPL"),
            "agent_TIME_TO_REACH_GOAL": agent.get("TIME_TO_REACH_GOAL"),
            "orca_TIME_TO_REACH_GOAL": orca.get("TIME_TO_REACH_GOAL"),
            "agent_SPL": agent.get("SPL"),
            "orca_SPL": orca.get("SPL"),
            "agent_STL": agent.get("STL"),
            "orca_STL": orca.get("STL"),
        }
        for key in ["PATH_LENGTH", "TIME_TO_REACH_GOAL", "A_STAR_SPL", "SPL", "STL"]:
            row[f"delta_{key}"] = _numeric_delta(agent.get(key), orca.get(key))
        row["winner"] = _winner(agent, orca)
        rows.append(row)
    return rows


def summarize_comparison(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(rows)
    summary = {
        "paired_episodes": len(rows),
        "agent_win_count": sum(1 for row in rows if row.get("winner") == "agent"),
        "orca_win_count": sum(1 for row in rows if row.get("winner") == "orca"),
        "tie_count": sum(1 for row in rows if row.get("winner") == "tie"),
        "unknown_count": sum(1 for row in rows if row.get("winner") == "unknown"),
    }
    for key in ["delta_PATH_LENGTH", "delta_TIME_TO_REACH_GOAL", "delta_A_STAR_SPL", "delta_SPL", "delta_STL"]:
        summary[f"mean_{key.lower()}"] = mean_or_none(numeric_values(rows, key))
    return summary


def _numeric_delta(agent_value, orca_value):
    if isinstance(agent_value, Real) and isinstance(orca_value, Real):
        return float(agent_value) - float(orca_value)
    return None


def _winner(agent: Dict[str, Any], orca: Dict[str, Any]) -> str:
    agent_success = _metric_bool(agent, "SUCCESS")
    orca_success = _metric_bool(orca, "SUCCESS")
    if agent_success is None or orca_success is None:
        return "unknown"
    if agent_success != orca_success:
        return "agent" if agent_success else "orca"

    agent_spl = agent.get("A_STAR_SPL")
    orca_spl = orca.get("A_STAR_SPL")
    if isinstance(agent_spl, Real) and isinstance(orca_spl, Real):
        if abs(float(agent_spl) - float(orca_spl)) <= 1e-9:
            return "tie"
        return "agent" if float(agent_spl) > float(orca_spl) else "orca"
    return "unknown"


def _metric_bool(row: Dict[str, Any], key: str):
    value = row.get(key)
    if value is None:
        return None
    return bool(value)
