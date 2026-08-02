"""Create top-down trajectory snapshots from recorded world-coordinate paths."""

from pathlib import Path
from typing import Mapping, Sequence

import cv2

from training_pipeline.utils import configure_matplotlib_cache

configure_matplotlib_cache()

import matplotlib.pyplot as plt


OUTCOME_STYLES = {
    "success": ("tab:green", "o"),
    "collision": ("tab:red", "X"),
    "timeout": ("tab:orange", "s"),
    "max_steps_reached": ("tab:gray", "s"),
    "other": ("tab:gray", "s"),
}


def save_trajectory_snapshot(
    initial_frame,
    robot_path: Sequence[Sequence[float]],
    goal: Sequence[float],
    summary: Mapping,
    output_path: Path,
    dpi: int = 180,
) -> None:
    """Plot the actual robot path over the initial physical scene."""
    map_width = float(summary["map_width"])
    map_height = float(summary["map_height"])
    figure, axis = plt.subplots(figsize=(7.2, 7.2 * map_height / map_width))
    axis.imshow(
        cv2.cvtColor(initial_frame, cv2.COLOR_BGR2RGB),
        extent=(-map_width / 2.0, map_width / 2.0, -map_height / 2.0, map_height / 2.0),
    )

    axis.plot(
        [point[0] for point in robot_path],
        [point[1] for point in robot_path],
        color="tab:blue",
        linewidth=2.2,
        label="Robot trajectory",
        zorder=5,
    )
    axis.scatter(
        robot_path[0][0],
        robot_path[0][1],
        color="limegreen",
        edgecolor="black",
        s=65,
        label="Start",
        zorder=6,
    )
    axis.scatter(
        goal[0],
        goal[1],
        color="gold",
        edgecolor="black",
        marker="*",
        s=150,
        label="Goal",
        zorder=6,
    )
    terminal_colour, terminal_marker = OUTCOME_STYLES.get(
        str(summary["outcome"]),
        OUTCOME_STYLES["other"],
    )
    axis.scatter(
        robot_path[-1][0],
        robot_path[-1][1],
        color=terminal_colour,
        edgecolor="black",
        marker=terminal_marker,
        s=80,
        label=f"Final position ({summary['outcome'].replace('_', ' ')})",
        zorder=7,
    )

    axis.set_xlim(-map_width / 2.0, map_width / 2.0)
    axis.set_ylim(-map_height / 2.0, map_height / 2.0)
    axis.set_aspect("equal")
    axis.set_xlabel("World x (m)")
    axis.set_ylabel("World y (m)")
    axis.set_title(
        f"Trajectory snapshot: seed {summary['seed']}\n"
        "Entities shown at episode start"
    )
    axis.legend(loc="upper right", fontsize=8)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)
