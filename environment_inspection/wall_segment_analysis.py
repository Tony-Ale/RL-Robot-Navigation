import math
import sys
import tempfile
from pathlib import Path

import gym
import numpy as np

from navigation_features.wall_geometry import is_boundary_wall
from training_pipeline.utils import load_yaml


ENTITY_FEATURE_DIM = 14


def calculate_wall_segment_capacity(config, segment_size, include_boundary_walls=True):
    """Return the maximum segment count for a square/rectangular SocNavGym room."""
    env_cfg = config["env"]
    shape = env_cfg["set_shape"]
    if shape == "no-walls":
        return 0
    if shape not in ("square", "rectangle"):
        raise ValueError("Wall capacity analysis supports square and rectangle rooms.")

    segment_size = float(segment_size)
    if segment_size <= 0:
        raise ValueError("wall_segment_size must be greater than zero.")

    map_x = float(env_cfg["max_map_x"])
    map_y = map_x if shape == "square" else float(env_cfg["max_map_y"])
    boundary_segments = 2 * _segment_count(map_x, segment_size) + 2 * _segment_count(map_y, segment_size)

    if not env_cfg.get("add_corridors", False):
        return boundary_segments if include_boundary_walls else 0

    robot_diameter = 2.0 * float(config["robot"]["robot_radius"])
    human_diameter = float(config["human"]["human_diameter"])
    minimum_gap = max(robot_diameter, human_diameter) + 0.5
    solid_length = map_x - minimum_gap
    if solid_length <= 0:
        raise ValueError("The minimum corridor opening leaves no valid corridor wall.")

    # Each divider is split into two positive pieces around its opening. A split
    # can add one partial segment beyond segmenting the same solid length whole.
    segments_per_divider = _segment_count(solid_length, segment_size) + 1
    corridor_segments = 2 * segments_per_divider
    return corridor_segments + (boundary_segments if include_boundary_walls else 0)


def count_live_wall_segments(env, include_boundary_walls=True):
    """Count the exact wall rows produced by SocNavGym for the current episode."""
    base_env = env.unwrapped
    rows = 0
    for wall in base_env.walls:
        if not include_boundary_walls and is_boundary_wall(wall, base_env.MAP_X, base_env.MAP_Y):
            continue
        values = np.asarray(base_env._get_entity_obs(wall), dtype=np.float32).reshape(-1)
        if values.size % ENTITY_FEATURE_DIM != 0:
            raise ValueError("SocNavGym wall observation size is not divisible by 14.")
        rows += values.size // ENTITY_FEATURE_DIM
    return rows


def run_wall_segment_analysis(config, pipeline_config):
    settings = config["wall_segment_analysis"]
    config_path = settings.get("config_path") or _target_environment_config(pipeline_config)
    source_config = load_yaml(config_path)
    segment_sizes = [float(value) for value in settings["segment_sizes"]]
    seed_start = int(settings["seed_start"])
    episodes = int(settings["episodes"])
    include_boundary_walls = bool(settings.get("include_boundary_walls", True))
    if episodes <= 0:
        raise ValueError("wall_segment_analysis.episodes must be greater than zero.")

    _configure_socnavgym_import(pipeline_config)
    import socnavgym  # noqa: F401 - registers SocNavGym-v1

    print(
        f"Wall segment analysis: config={config_path} seeds={seed_start}-{seed_start + episodes - 1} "
        f"boundary_walls={'included' if include_boundary_walls else 'excluded'}"
    )
    for segment_size in segment_sizes:
        capacity = calculate_wall_segment_capacity(source_config, segment_size, include_boundary_walls)
        maximum_observed = -1
        maximum_seed = None

        env_config = _with_segment_size(source_config, segment_size)
        with tempfile.TemporaryDirectory() as directory:
            generated_path = Path(directory) / "wall_segment_analysis.yaml"
            _write_yaml(generated_path, env_config)
            env = gym.make(pipeline_config["environment"]["id"], config=str(generated_path))
            try:
                for seed in range(seed_start, seed_start + episodes):
                    env.reset(seed=seed)
                    observed = count_live_wall_segments(env, include_boundary_walls)
                    if observed > capacity:
                        raise AssertionError(
                            f"Seed {seed} produced {observed} wall segments, exceeding capacity {capacity}."
                        )
                    if observed > maximum_observed:
                        maximum_observed = observed
                        maximum_seed = seed
            finally:
                env.close()

        print(
            f"  segment_size={segment_size:g} calculated={capacity} "
            f"observed_max={maximum_observed} seed={maximum_seed}"
        )


def _segment_count(length, segment_size):
    return int(math.ceil(float(length) / float(segment_size) - 1e-12))


def _target_environment_config(pipeline_config):
    fixed = pipeline_config.get("wrappers", {}).get("fixed_observation_space", {})
    if fixed.get("enabled", False):
        return fixed["config_path"]
    return pipeline_config["environment"]["config_path"]


def _configure_socnavgym_import(pipeline_config):
    environment = pipeline_config["environment"]
    if not environment.get("use_socnavgym_clone", False):
        return
    clone_path = str(Path(environment["socnavgym_clone_path"]).resolve())
    if clone_path not in sys.path:
        sys.path.insert(0, clone_path)


def _with_segment_size(config, segment_size):
    import copy

    updated = copy.deepcopy(config)
    updated["env"]["wall_segment_size"] = float(segment_size)
    return updated


def _write_yaml(path, config):
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required for wall segment analysis.") from exc
    path.write_text(yaml.safe_dump(config, sort_keys=False))
