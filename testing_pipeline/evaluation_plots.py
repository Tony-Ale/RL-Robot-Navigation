import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from training_pipeline.utils import configure_matplotlib_cache

configure_matplotlib_cache()


METRIC_LABELS = {
    "success_rate": "Success Rate",
    "collision_rate": "Collision Rate",
    "timeout_rate": "Timeout Rate",
    "mean_episode_reward": "Mean Episode Reward",
    "mean_a_star_spl": "Mean A*-Referenced SPL",
    "mean_spl": "Mean SPL",
    "mean_stl": "Mean STL",
    "A_STAR_SPL": "A*-Referenced SPL",
    "PATH_LENGTH": "Path Length",
    "TIME_TO_REACH_GOAL": "Time to Reach Goal",
    "MINIMUM_DISTANCE_TO_HUMAN": "Minimum Distance to Entity",
    "PERSONAL_SPACE_COMPLIANCE": "Personal-Space Compliance",
    "episode_reward": "Episode Reward",
    "SUCCESS": "Success Rate",
    "COLLISION": "Collision Rate",
}

RATE_METRICS = {
    "success_rate",
    "collision_rate",
    "timeout_rate",
    "mean_a_star_spl",
    "mean_spl",
    "mean_stl",
    "A_STAR_SPL",
    "PERSONAL_SPACE_COMPLIANCE",
    "SUCCESS",
    "COLLISION",
}


def generate_evaluation_plots(plot_cfg: Dict[str, Any], default_output_dir: Path) -> List[Path]:
    sections = [
        plot_cfg.get("architecture_comparison", {}),
        plot_cfg.get("final_outcomes", {}),
        plot_cfg.get("training_curves", {}),
    ]
    if not any(section.get("enabled", False) for section in sections):
        return []

    import matplotlib.pyplot as plt

    output_dir = Path(default_output_dir) / plot_cfg.get("output_dir", "plots")
    save = bool(plot_cfg.get("save", True))
    show = bool(plot_cfg.get("show", False))
    if save:
        output_dir.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    if sections[0].get("enabled", False):
        for name, figure in architecture_figures(sections[0]):
            _finish_figure(figure, output_dir / "architecture_comparison" / name, plot_cfg, save, show, written)
    if sections[1].get("enabled", False):
        for name, figure in final_outcome_figures(sections[1]):
            _finish_figure(figure, output_dir / "final_outcomes" / name, plot_cfg, save, show, written)
    if sections[2].get("enabled", False):
        for name, figure in training_curve_figures(sections[2]):
            _finish_figure(figure, output_dir / "training_curves" / name, plot_cfg, save, show, written)

    if not show:
        plt.close("all")
    return written


def architecture_figures(section: Dict[str, Any]):
    series, common_steps = load_architecture_series(section)
    metrics = section.get(
        "metrics",
        ["success_rate", "collision_rate", "mean_episode_reward", "mean_a_star_spl"],
    )
    figures = []
    for metric in metrics:
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(8, 4.5))
        plotted = False
        for label, rows in series.items():
            points = [
                (int(row["checkpoint_step"]), _float(row.get(metric)))
                for row in rows
                if int(row["checkpoint_step"]) in common_steps
            ]
            points = [(step, value) for step, value in points if value is not None]
            if not points:
                continue
            axis.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                marker="o",
                label=label,
            )
            plotted = True
        if not plotted:
            plt.close(figure)
            continue
        _format_axis(axis, metric, "Training Steps", _label(metric))
        axis.set_title(f"Architecture Comparison: {_label(metric)}")
        axis.legend()
        figures.append((f"{_file_name(metric)}.png", figure))
    return figures


def load_architecture_series(section: Dict[str, Any]) -> Tuple[Dict[str, List[Dict[str, str]]], set]:
    models = _models(section, "architecture_comparison")
    max_step = section.get("max_step")
    if max_step is not None:
        max_step = int(max_step)
        if max_step <= 0:
            raise ValueError("plots.architecture_comparison.max_step must be positive.")

    series: Dict[str, List[Dict[str, str]]] = {}
    step_sets = []
    for model in models:
        label = model["label"]
        controller = model.get("controller", "learned_agent")
        rows = [
            row
            for row in _read_csv(Path(model["summary_csv"]))
            if row.get("controller") == controller
            and _integer(row.get("checkpoint_step")) is not None
            and (max_step is None or int(row["checkpoint_step"]) <= max_step)
        ]
        if not rows:
            raise ValueError(f"No architecture-comparison rows found for '{label}'.")
        steps = [int(row["checkpoint_step"]) for row in rows]
        if len(steps) != len(set(steps)):
            raise ValueError(f"Architecture-comparison data for '{label}' contains duplicate checkpoint steps.")
        rows.sort(key=lambda row: int(row["checkpoint_step"]))
        series[label] = rows
        step_sets.append(set(steps))

    common_steps = set.intersection(*step_sets)
    if not common_steps:
        raise ValueError("Architecture-comparison models have no common checkpoint steps.")
    return series, common_steps


def final_outcome_figures(section: Dict[str, Any]):
    model_rows = load_final_model_rows(section)
    figures = [("outcome_rates.png", _outcome_rate_figure(model_rows))]
    for metric_cfg in section.get("metrics", []):
        if isinstance(metric_cfg, str):
            metric_cfg = {"name": metric_cfg}
        metric = metric_cfg["name"]
        successful_only = bool(metric_cfg.get("successful_only", False))
        values = []
        labels = []
        for label, rows in model_rows.items():
            selected = [row for row in rows if not successful_only or _boolean(row.get("SUCCESS")) is True]
            metric_values = [_float(row.get(metric)) for row in selected]
            metric_values = [value for value in metric_values if value is not None]
            if metric_values:
                labels.append(label)
                values.append(metric_values)
        if not values:
            continue

        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(8, 4.5))
        display_labels = [label.replace(": ", "\n", 1) for label in labels]
        axis.bar(display_labels, [float(np.mean(metric_values)) for metric_values in values])
        suffix = " (Successful Episodes)" if successful_only else ""
        axis.set_title(f"Mean {metric_cfg.get('title', _label(metric))}{suffix}")
        _format_axis(axis, metric, "", metric_cfg.get("ylabel", _label(metric)))
        figures.append((f"{_file_name(metric)}.png", figure))
    return figures


def load_final_model_rows(section: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]:
    models = _models(section, "final_outcomes")
    result: Dict[str, List[Dict[str, str]]] = {}
    seed_sets = []
    for model in models:
        label = model["label"]
        controller = model.get("controller", "learned_agent")
        rows = [row for row in _read_csv(Path(model["episode_csv"])) if row.get("controller") == controller]
        requested_step = model.get("checkpoint_step")
        available_steps = {
            int(row["checkpoint_step"])
            for row in rows
            if _integer(row.get("checkpoint_step")) is not None
        }
        if requested_step is None:
            if len(available_steps) != 1:
                raise ValueError(
                    f"Final-outcome model '{label}' must set checkpoint_step because its CSV contains "
                    f"{len(available_steps)} checkpoint steps."
                )
            requested_step = next(iter(available_steps))
        rows = [row for row in rows if _integer(row.get("checkpoint_step")) == int(requested_step)]
        if not rows:
            raise ValueError(f"No final-outcome rows found for '{label}' at checkpoint {requested_step}.")

        seeds = [row.get("seed") for row in rows]
        if any(seed in (None, "") for seed in seeds):
            raise ValueError(f"Final-outcome rows for '{label}' contain missing seeds.")
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Final-outcome rows for '{label}' contain duplicate seeds.")
        result[label] = sorted(rows, key=lambda row: int(row["seed"]))
        seed_sets.append(set(seeds))

    if section.get("require_matching_seeds", True):
        reference = seed_sets[0]
        mismatched = [label for label, seeds in zip(result, seed_sets) if seeds != reference]
        if mismatched:
            raise ValueError(
                "Final-outcome comparisons require identical seed sets; mismatch for "
                + ", ".join(mismatched)
                + "."
            )
    return result


def _outcome_rate_figure(model_rows: Dict[str, List[Dict[str, str]]]):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    outcomes = ["SUCCESS", "COLLISION", "TIMEOUT"]
    labels = list(model_rows)
    x_positions = np.arange(len(outcomes), dtype=float)
    width = 0.8 / len(labels)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for index, label in enumerate(labels):
        rows = model_rows[label]
        rates = []
        for outcome in outcomes:
            values = [_boolean(row.get(outcome)) for row in rows]
            if any(value is None for value in values):
                raise ValueError(f"Final-outcome rows for '{label}' contain missing {outcome} values.")
            rates.append(float(np.mean(values)))
        positions = x_positions - 0.4 + width / 2 + index * width
        axis.bar(positions, rates, width, label=label)
    axis.set_xticks(x_positions, ["Success", "Collision", "Timeout"])
    axis.set_ylabel("Episode Rate")
    axis.set_title("Final Held-Out Outcomes")
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_ylim(0.0, 1.05)
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend()
    return figure


def training_curve_figures(section: Dict[str, Any]):
    model_rows = load_training_series(section)
    window = int(section.get("rolling_window_episodes", 100))
    if window <= 0:
        raise ValueError("plots.training_curves.rolling_window_episodes must be positive.")
    metrics = section.get("metrics", ["episode_reward", "SUCCESS", "COLLISION"])
    figures = []
    for metric in metrics:
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(8, 4.5))
        plotted = False
        for label, rows in model_rows.items():
            points = [
                (
                    int(row["global_step"]),
                    _boolean_number(row.get(metric)) if metric in {"SUCCESS", "COLLISION"} else _float(row.get(metric)),
                )
                for row in rows
            ]
            points = [(step, value) for step, value in points if value is not None]
            if len(points) < window:
                continue
            steps = [point[0] for point in points][window - 1 :]
            values = np.convolve(
                np.asarray([point[1] for point in points], dtype=float),
                np.ones(window, dtype=float) / window,
                mode="valid",
            )
            axis.plot(steps, values, label=label)
            plotted = True
        if not plotted:
            plt.close(figure)
            continue
        _format_axis(axis, metric, "Training Steps", f"{window}-Episode Rolling {_label(metric)}")
        axis.set_title(f"Training: {window}-Episode Rolling {_label(metric)}")
        axis.legend()
        figures.append((f"{_file_name(metric)}.png", figure))
    return figures


def load_training_series(section: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]:
    result = {}
    for model in _models(section, "training_curves"):
        label = model["label"]
        combined = []
        occupied_ranges = []
        segments = model.get("segments", [])
        if not segments:
            raise ValueError(f"Training-curve model '{label}' must define at least one segment.")
        for segment in segments:
            csv_path = segment.get("csv")
            if csv_path is None:
                if "run_dir" not in segment:
                    raise ValueError(f"A training segment for '{label}' must define csv or run_dir.")
                csv_path = Path(segment["run_dir"]) / "metrics" / "navigation_training_metrics.csv"
            rows = []
            for row in _read_csv(Path(csv_path)):
                step = _integer(row.get("global_step"))
                if step is None:
                    continue
                start = segment.get("start_step")
                end = segment.get("end_step")
                if start is not None and step < int(start):
                    continue
                if end is not None and step > int(end):
                    continue
                rows.append(row)
            if not rows:
                raise ValueError(f"Training segment '{csv_path}' produced no rows for '{label}'.")
            current_range = (
                min(int(row["global_step"]) for row in rows),
                max(int(row["global_step"]) for row in rows),
            )
            if any(current_range[0] <= end and start <= current_range[1] for start, end in occupied_ranges):
                raise ValueError(f"Training segments for '{label}' overlap in global-step range.")
            occupied_ranges.append(current_range)
            combined.extend(rows)

        combined.sort(key=lambda row: int(row["global_step"]))
        steps = [int(row["global_step"]) for row in combined]
        if len(steps) != len(set(steps)):
            raise ValueError(f"Training data for '{label}' contains duplicate global steps.")
        result[label] = combined
    return result


def _models(section: Dict[str, Any], name: str) -> List[Dict[str, Any]]:
    models = section.get("models", [])
    if not models:
        raise ValueError(f"plots.{name}.models must contain at least one model.")
    labels = [model.get("label") for model in models]
    if any(not label for label in labels):
        raise ValueError(f"Every plots.{name} model must have a label.")
    if len(labels) != len(set(labels)):
        raise ValueError(f"plots.{name} model labels must be unique.")
    return models


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Plot data CSV not found: {path}")
    with open(path, newline="") as file:
        return list(csv.DictReader(file))


def _finish_figure(
    figure,
    path: Path,
    plot_cfg: Dict[str, Any],
    save: bool,
    show: bool,
    written: List[Path],
) -> None:
    import matplotlib.pyplot as plt

    figure.tight_layout()
    if save:
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=int(plot_cfg.get("dpi", 180)), bbox_inches="tight")
        written.append(path)
    if show:
        plt.show()
    plt.close(figure)


def _format_axis(axis, metric: str, xlabel: str, ylabel: str) -> None:
    from matplotlib.ticker import PercentFormatter

    if xlabel:
        axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)
    if metric in RATE_METRICS:
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))


def _integer(value) -> Optional[int]:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _float(value) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _boolean(value) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1"}:
            return True
        if lowered in {"false", "0"}:
            return False
    return None


def _boolean_number(value) -> Optional[float]:
    boolean = _boolean(value)
    return float(boolean) if boolean is not None else None


def _label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric.replace("_", " ").title())


def _file_name(metric: str) -> str:
    return metric.lower().replace("*", "star").replace(" ", "_")
