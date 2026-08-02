"""Plot synchronized stateful-attention traces and SocNavGym frames."""

from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import cv2
from training_pipeline.utils import configure_matplotlib_cache

configure_matplotlib_cache()

import matplotlib.pyplot as plt
import numpy as np


EVENT_TITLES = {
    "peak_attention": "Peak attention",
    "attention_switch": "Largest attention switch",
    "minimum_clearance": "Minimum clearance",
    "pre_terminal": "Before terminal event",
}


def render_labelled_frame(base_env, entities: Sequence[Mapping], top_slot: int | None) -> np.ndarray:
    """Render the current physical state without opening SocNavGym's GUI window."""
    image = render_environment_frame(base_env)

    for entity in entities:
        if not entity["valid"]:
            continue
        x_px = _world_to_pixel_x(base_env, entity["world_x"])
        y_px = _world_to_pixel_y(base_env, entity["world_y"])
        selected = int(entity["slot"]) == top_slot
        colour = (0, 0, 220) if selected else (30, 30, 30)
        thickness = 2 if selected else 1
        cv2.circle(image, (x_px, y_px), 12, colour, thickness)
        cv2.putText(
            image,
            entity["label"],
            (x_px + 8, y_px - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            colour,
            1,
            cv2.LINE_AA,
        )
    return image


def render_environment_frame(base_env, include_callbacks: bool = True) -> np.ndarray:
    """Render the physical environment into an image without opening a GUI window."""
    image = np.full(
        (int(base_env.RESOLUTION_Y), int(base_env.RESOLUTION_X), 3),
        255,
        dtype=np.uint8,
    )
    draw_args = (
        image,
        base_env.PIXEL_TO_WORLD_X,
        base_env.PIXEL_TO_WORLD_Y,
        base_env.MAP_X,
        base_env.MAP_Y,
    )

    for obj in base_env.walls + base_env.tables + base_env.laptops + base_env.plants:
        obj.draw(*draw_args)

    _draw_circle(base_env, image, base_env.robot.goal_x, base_env.robot.goal_y, base_env.GOAL_RADIUS, (0, 180, 0))
    for human in base_env.dynamic_humans:
        _draw_circle(base_env, image, human.goal_x, human.goal_y, base_env.HUMAN_GOAL_RADIUS, (120, 0, 0))
    for interaction in base_env.moving_interactions:
        _draw_circle(base_env, image, interaction.goal_x, interaction.goal_y, interaction.goal_radius, (0, 0, 180))

    for human in base_env.static_humans + base_env.dynamic_humans:
        human.draw(*draw_args)
    base_env.robot.draw(*draw_args)
    for interaction in base_env.moving_interactions + base_env.static_interactions + base_env.h_l_interactions:
        interaction.draw(*draw_args)
    if include_callbacks:
        for callback in getattr(base_env, "render_callbacks", []):
            callback(image, base_env)
    return image


def save_episode_outputs(
    seed: int,
    records: Sequence[Mapping],
    events: Mapping[str, Mapping],
    output_dir: Path,
    plot_config: Mapping,
    folder_label: str | None = None,
) -> None:
    """Save heatmaps, action profile, selected frames, and an overview figure."""
    suffix = f"_{folder_label}" if folder_label else ""
    episode_dir = Path(output_dir) / f"seed_{seed}{suffix}"
    frame_dir = episode_dir / "frames"
    episode_dir.mkdir(parents=True, exist_ok=True)
    frame_dir.mkdir(parents=True, exist_ok=True)

    matrices = _trace_matrices(records)
    ordered_events = _ordered_unique_events(events)
    letters = {name: chr(ord("A") + index) for index, (name, _) in enumerate(ordered_events)}
    dpi = int(plot_config.get("dpi", 180))
    attention_min = float(plot_config.get("attention_min", 0.0))
    attention_max = float(plot_config.get("attention_max", 1.0))
    if attention_min >= attention_max:
        raise ValueError("plots.attention_min must be smaller than plots.attention_max.")

    if plot_config.get("save_individual", True):
        _save_heatmap(
            matrices["attention"],
            matrices["times"],
            matrices["labels"],
            ordered_events,
            letters,
            "Attention weight",
            episode_dir / "attention_heatmap.png",
            "viridis",
            attention_min,
            attention_max,
            dpi,
            extend="max",
        )
        _save_heatmap(
            matrices["distance"],
            matrices["times"],
            matrices["labels"],
            ordered_events,
            letters,
            "Robot-entity centre distance (m)",
            episode_dir / "distance_heatmap.png",
            "magma_r",
            None,
            None,
            dpi,
        )
        _save_actions(matrices, ordered_events, letters, episode_dir / "action_profile.png", dpi)

    for name, event in ordered_events:
        cv2.imwrite(str(frame_dir / f"{letters[name]}_{name}_step_{event['step']:03d}.png"), event["frame"])

    if plot_config.get("save_overview", True):
        _save_overview(
            seed,
            matrices,
            ordered_events,
            letters,
            episode_dir / "episode_overview.png",
            dpi,
            attention_min,
            attention_max,
        )


def _trace_matrices(records: Sequence[Mapping]) -> Dict[str, np.ndarray | List[str]]:
    if not records:
        raise ValueError("Cannot plot an empty attention trace.")
    steps = sorted({int(row["step"]) for row in records})
    slots = sorted({int(row["slot"]) for row in records})
    step_index = {step: index for index, step in enumerate(steps)}
    slot_index = {slot: index for index, slot in enumerate(slots)}
    attention = np.full((len(slots), len(steps)), np.nan)
    distance = np.full_like(attention, np.nan)
    labels = [""] * len(slots)
    times = np.zeros(len(steps), dtype=np.float64)
    linear = np.zeros(len(steps), dtype=np.float64)
    angular = np.zeros(len(steps), dtype=np.float64)

    for row in records:
        x = step_index[int(row["step"])]
        y = slot_index[int(row["slot"])]
        labels[y] = str(row["label"])
        times[x] = float(row["simulation_time"])
        linear[x] = float(row["linear_action"])
        angular[x] = float(row["angular_action"])
        if bool(row["valid"]):
            attention[y, x] = float(row["attention"])
            distance[y, x] = float(row["distance"])
    return {
        "attention": attention,
        "distance": distance,
        "labels": labels,
        "times": times,
        "linear": linear,
        "angular": angular,
    }


def _save_heatmap(
    matrix,
    times,
    labels,
    events,
    letters,
    colourbar_label,
    path,
    cmap_name,
    vmin,
    vmax,
    dpi,
    extend="neither",
):
    figure, axis = plt.subplots(figsize=(11, max(4.2, len(labels) * 0.34)))
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("#d0d0d0")
    image = axis.imshow(matrix, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_xlabel("Simulation time (s)")
    axis.set_ylabel("Entity slot")
    axis.set_yticks(np.arange(len(labels)), labels)
    _set_time_ticks(axis, times)
    _mark_events(axis, events, letters)
    figure.colorbar(image, ax=axis, label=colourbar_label, extend=extend)
    figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _save_actions(matrices, events, letters, path, dpi):
    figure, axis = plt.subplots(figsize=(11, 3.8))
    axis.plot(matrices["times"], matrices["linear"], label="Linear action")
    axis.plot(matrices["times"], matrices["angular"], label="Angular action")
    for name, event in events:
        axis.axvline(event["simulation_time"], color="black", alpha=0.35, linewidth=1)
        axis.text(event["simulation_time"], 1.02, letters[name], transform=axis.get_xaxis_transform())
    axis.set_xlabel("Simulation time (s)")
    axis.set_ylabel("Normalized action")
    axis.set_ylim(-1.05, 1.05)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _save_overview(seed, matrices, events, letters, path, dpi, attention_min, attention_max):
    frame_count = max(len(events), 1)
    figure = plt.figure(figsize=(4.1 * frame_count, 8.0))
    grid = figure.add_gridspec(2, frame_count, height_ratios=(1.2, 1.0))
    heat_axis = figure.add_subplot(grid[0, :])
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#d0d0d0")
    image = heat_axis.imshow(
        matrices["attention"],
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=attention_min,
        vmax=attention_max,
    )
    heat_axis.set_xlabel("Simulation time (s)")
    heat_axis.set_ylabel("Entity slot")
    heat_axis.set_yticks(np.arange(len(matrices["labels"])), matrices["labels"])
    _set_time_ticks(heat_axis, matrices["times"])
    _mark_events(heat_axis, events, letters)
    figure.colorbar(image, ax=heat_axis, label="Attention weight", fraction=0.025, extend="max")

    for index, (name, event) in enumerate(events):
        axis = figure.add_subplot(grid[1, index])
        axis.imshow(cv2.cvtColor(event["frame"], cv2.COLOR_BGR2RGB))
        axis.set_title(f"{letters[name]}: {EVENT_TITLES[name]}\nstep {event['step']}")
        axis.axis("off")
    figure.suptitle(f"Stateful attention analysis: seed {seed}")
    figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _ordered_unique_events(events: Mapping[str, Mapping]):
    ordered = []
    used_steps = set()
    for name in ("peak_attention", "attention_switch", "minimum_clearance", "pre_terminal"):
        event = events.get(name)
        if event is None or int(event["step"]) in used_steps:
            continue
        used_steps.add(int(event["step"]))
        ordered.append((name, event))
    return ordered


def _mark_events(axis, events, letters):
    for name, event in events:
        x = int(event["step"])
        axis.axvline(x, color="white", linewidth=1.2, linestyle="--")
        axis.text(x + 0.2, -0.75, letters[name], color="black", weight="bold")


def _set_time_ticks(axis, times):
    if len(times) == 0:
        return
    positions = np.linspace(0, len(times) - 1, min(7, len(times)), dtype=int)
    axis.set_xticks(positions, [f"{times[index]:.1f}" for index in positions])


def _draw_circle(base_env, image, x, y, radius, colour):
    centre = (_world_to_pixel_x(base_env, x), _world_to_pixel_y(base_env, y))
    edge = _world_to_pixel_x(base_env, x + radius)
    cv2.circle(image, centre, abs(edge - centre[0]), colour, 2)


def _world_to_pixel_x(base_env, x):
    return int(base_env.PIXEL_TO_WORLD_X * (float(x) + float(base_env.MAP_X) / 2.0))


def _world_to_pixel_y(base_env, y):
    return int(base_env.PIXEL_TO_WORLD_Y * (float(base_env.MAP_Y) / 2.0 - float(y)))
